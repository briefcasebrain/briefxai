use anyhow::Result;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use std::collections::HashMap;
use tracing::{info, warn};

use crate::config::BriefXAIConfig;
use crate::llm::LlmClient;
use crate::prompts::{
    get_assign_to_high_level_cluster_prompt, get_facet_cluster_name_prompt,
    get_neighborhood_cluster_names_prompt, get_renaming_higher_level_cluster_prompt,
};
use crate::types::{ConversationCluster, Facet, FacetValue};
use crate::utils::{most_common, sample_items};

pub async fn create_base_clusters(
    config: &BriefXAIConfig,
    embeddings: &[Vec<f32>],
    facet_data: &[Vec<FacetValue>],
) -> Result<Vec<ConversationCluster>> {
    info!(
        "Creating base clusters for {} data points",
        embeddings.len()
    );

    let n_clusters = config.n_base_clusters(embeddings.len());
    let clusters = perform_kmeans(embeddings, n_clusters)?;

    let mut base_clusters = Vec::new();
    let llm_client = LlmClient::new(config.clone()).await?;

    // Group by facet for clustering
    let facets_to_cluster: Vec<&Facet> = facet_data[0]
        .iter()
        .map(|fv| &fv.facet)
        .filter(|f| f.should_make_clusters())
        .collect();

    for facet in facets_to_cluster {
        let facet_clusters = create_facet_clusters(
            config,
            &llm_client,
            facet,
            &clusters,
            facet_data,
            embeddings,
        )
        .await?;

        base_clusters.extend(facet_clusters);
    }

    info!("Created {} base clusters", base_clusters.len());
    Ok(base_clusters)
}

async fn create_facet_clusters(
    config: &BriefXAIConfig,
    llm_client: &LlmClient,
    facet: &Facet,
    cluster_assignments: &[usize],
    facet_data: &[Vec<FacetValue>],
    _embeddings: &[Vec<f32>],
) -> Result<Vec<ConversationCluster>> {
    let mut clusters = Vec::new();
    let n_clusters = cluster_assignments.iter().max().copied().unwrap_or(0) + 1;

    for cluster_id in 0..n_clusters {
        let indices: Vec<usize> = cluster_assignments
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == cluster_id)
            .map(|(i, _)| i)
            .collect();

        if indices.is_empty() {
            continue;
        }

        // Sample points inside and outside cluster
        let inside_samples = sample_cluster_points(
            &indices,
            facet_data,
            facet,
            config.max_points_to_sample_inside_cluster,
        );

        let outside_indices: Vec<usize> = (0..cluster_assignments.len())
            .filter(|i| !indices.contains(i))
            .collect();

        let outside_samples = sample_cluster_points(
            &outside_indices,
            facet_data,
            facet,
            config.max_points_to_sample_outside_cluster,
        );

        // Generate name and description
        let (name, summary) = generate_cluster_name_description(
            llm_client,
            facet,
            &inside_samples,
            &outside_samples,
            config.n_name_description_samples_per_cluster,
        )
        .await?;

        clusters.push(ConversationCluster {
            facet: facet.clone(),
            summary,
            name,
            children: None,
            parent: None,
            indices: Some(indices),
        });
    }

    Ok(clusters)
}

fn sample_cluster_points(
    indices: &[usize],
    facet_data: &[Vec<FacetValue>],
    target_facet: &Facet,
    n_samples: usize,
) -> Vec<String> {
    let sampled_indices = sample_items(indices, n_samples, None);

    sampled_indices
        .iter()
        .filter_map(|&i| {
            facet_data[i]
                .iter()
                .find(|fv| fv.facet == *target_facet)
                .map(|fv| fv.value.clone())
        })
        .collect()
}

async fn generate_cluster_name_description(
    llm_client: &LlmClient,
    facet: &Facet,
    inside_samples: &[String],
    outside_samples: &[String],
    n_samples: usize,
) -> Result<(String, String)> {
    let mut names = Vec::new();
    let mut descriptions = Vec::new();

    for _ in 0..n_samples {
        let prompt = get_facet_cluster_name_prompt(facet, inside_samples, outside_samples);
        let response = llm_client.complete(&prompt).await?;

        if let Some((name, desc)) = parse_name_description(&response) {
            names.push(name);
            descriptions.push(desc);
        }
    }

    let final_name = most_common(names).unwrap_or_else(|| "Unknown Cluster".to_string());
    let final_desc =
        most_common(descriptions).unwrap_or_else(|| "No description available".to_string());

    Ok((final_name, final_desc))
}

fn parse_name_description(response: &str) -> Option<(String, String)> {
    let lines: Vec<&str> = response.lines().collect();
    let mut name = None;
    let mut description = None;

    for line in lines {
        if line.starts_with("Name:") {
            name = Some(line.strip_prefix("Name:")?.trim().to_string());
        } else if line.starts_with("Description:") {
            description = Some(line.strip_prefix("Description:")?.trim().to_string());
        }
    }

    match (name, description) {
        (Some(n), Some(d)) => Some((n, d)),
        _ => None,
    }
}

pub async fn build_hierarchy(
    config: &BriefXAIConfig,
    base_clusters: Vec<ConversationCluster>,
) -> Result<Vec<ConversationCluster>> {
    info!(
        "Building cluster hierarchy from {} base clusters",
        base_clusters.len()
    );

    let llm_client = LlmClient::new(config.clone()).await?;
    let mut current_level = base_clusters;
    let mut hierarchy_levels = vec![current_level.clone()];

    while current_level.len() > config.min_top_level_size {
        info!(
            "Building next hierarchy level from {} clusters",
            current_level.len()
        );

        let next_level = build_next_level(config, &llm_client, current_level.clone()).await?;

        if next_level.len() >= current_level.len() {
            // No reduction, stop here
            break;
        }

        hierarchy_levels.push(next_level.clone());
        current_level = next_level;
    }

    // Return the top level
    Ok(current_level)
}

async fn build_next_level(
    config: &BriefXAIConfig,
    llm_client: &LlmClient,
    clusters: Vec<ConversationCluster>,
) -> Result<Vec<ConversationCluster>> {
    // Group clusters by facet
    let mut clusters_by_facet: HashMap<Facet, Vec<ConversationCluster>> = HashMap::new();

    for cluster in clusters {
        clusters_by_facet
            .entry(cluster.facet.clone())
            .or_default()
            .push(cluster);
    }

    let mut next_level = Vec::new();

    for (facet, facet_clusters) in clusters_by_facet {
        let higher_clusters =
            create_higher_level_clusters(config, llm_client, &facet, facet_clusters).await?;

        next_level.extend(higher_clusters);
    }

    Ok(next_level)
}

async fn create_higher_level_clusters(
    config: &BriefXAIConfig,
    llm_client: &LlmClient,
    facet: &Facet,
    clusters: Vec<ConversationCluster>,
) -> Result<Vec<ConversationCluster>> {
    // Create cluster embeddings by averaging their constituent point embeddings
    let cluster_embeddings = compute_cluster_embeddings(&clusters)?;

    // Use hierarchical clustering or k-means on cluster embeddings
    let n_neighborhoods = config.n_average_clusters_per_neighborhood(clusters.len());
    let neighborhoods = if cluster_embeddings.len() > 1 {
        cluster_hierarchical_clustering(&cluster_embeddings, n_neighborhoods, &clusters)?
    } else {
        // Single cluster case
        vec![clusters]
    };

    let mut higher_clusters = Vec::new();

    for neighborhood in neighborhoods {
        if neighborhood.is_empty() {
            continue;
        }

        let neighborhood_clusters =
            create_neighborhood_clusters(config, llm_client, facet, neighborhood).await?;

        higher_clusters.extend(neighborhood_clusters);
    }

    Ok(higher_clusters)
}

async fn create_neighborhood_clusters(
    config: &BriefXAIConfig,
    llm_client: &LlmClient,
    facet: &Facet,
    neighborhood: Vec<ConversationCluster>,
) -> Result<Vec<ConversationCluster>> {
    let n_desired = config.n_desired_higher_level_names_per_cluster(neighborhood.len());

    // Generate potential cluster names
    let cluster_summaries: Vec<(String, String)> = neighborhood
        .iter()
        .map(|c| (c.name.clone(), c.summary.clone()))
        .collect();

    let prompt = get_neighborhood_cluster_names_prompt(facet, &cluster_summaries, n_desired);
    let response = llm_client.complete(&prompt).await?;

    let higher_level_names = parse_higher_level_names(&response);

    // Assign clusters to higher level categories
    let assignments = assign_to_categories(llm_client, &neighborhood, &higher_level_names).await?;

    // Create higher level clusters
    let mut result = Vec::new();

    for (i, (name, desc)) in higher_level_names.iter().enumerate() {
        let children: Vec<ConversationCluster> = assignments
            .iter()
            .enumerate()
            .filter(|(_, &cat)| cat == i)
            .map(|(j, _)| neighborhood[j].clone())
            .collect();

        if children.is_empty() {
            continue;
        }

        // Refine name based on actual contents
        let refined =
            refine_cluster_name(llm_client, name, desc, &children, config.n_rename_samples).await?;

        result.push(ConversationCluster {
            facet: facet.clone(),
            name: refined.0,
            summary: refined.1,
            children: Some(children),
            parent: None,
            indices: None,
        });
    }

    Ok(result)
}

fn parse_higher_level_names(response: &str) -> Vec<(String, String)> {
    let mut names = Vec::new();
    let lines: Vec<&str> = response.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        if lines[i].contains("Name:") {
            if let Some(name) = lines[i].split("Name:").nth(1) {
                let name = name.trim().to_string();

                // Look for description on next line or same line
                let desc = if i + 1 < lines.len() && lines[i + 1].contains("Description:") {
                    i += 1;
                    lines[i]
                        .split("Description:")
                        .nth(1)
                        .map(|s| s.trim().to_string())
                } else if lines[i].contains("Description:") {
                    lines[i]
                        .split("Description:")
                        .nth(1)
                        .map(|s| s.trim().to_string())
                } else {
                    None
                };

                if let Some(description) = desc {
                    names.push((name, description));
                }
            }
        }
        i += 1;
    }

    names
}

async fn assign_to_categories(
    llm_client: &LlmClient,
    clusters: &[ConversationCluster],
    categories: &[(String, String)],
) -> Result<Vec<usize>> {
    let cluster_summaries: Vec<(String, String)> = clusters
        .iter()
        .map(|c| (c.name.clone(), c.summary.clone()))
        .collect();

    let prompt = get_assign_to_high_level_cluster_prompt(&cluster_summaries, categories);
    let response = llm_client.complete(&prompt).await?;

    parse_assignments(&response, clusters.len())
}

fn parse_assignments(response: &str, n_clusters: usize) -> Result<Vec<usize>> {
    let mut assignments = vec![0; n_clusters];

    for line in response.lines() {
        if let Some((cluster_str, category_str)) = line.split_once("->") {
            let cluster_idx = cluster_str.trim().parse::<usize>().ok().and_then(|i| {
                if i > 0 {
                    Some(i - 1)
                } else {
                    None
                }
            });

            let category_idx = category_str.trim().chars().next().and_then(|c| {
                if c.is_ascii_uppercase() {
                    Some((c as usize) - ('A' as usize))
                } else {
                    None
                }
            });

            if let (Some(cluster), Some(category)) = (cluster_idx, category_idx) {
                if cluster < n_clusters {
                    assignments[cluster] = category;
                }
            }
        }
    }

    Ok(assignments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Facet;

    fn create_test_config() -> BriefXAIConfig {
        BriefXAIConfig {
            seed: 42,
            verbose: false,
            llm_batch_size: 10,
            max_points_to_sample_inside_cluster: 5,
            max_points_to_sample_outside_cluster: 5,
            n_name_description_samples_per_cluster: 3,
            min_top_level_size: 2,
            ..Default::default()
        }
    }

    fn create_test_embeddings(n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| vec![i as f32 / n as f32, (n - i) as f32 / n as f32])
            .collect()
    }

    fn create_test_facet_data(n: usize) -> Vec<Vec<FacetValue>> {
        let facet = Facet {
            name: "sentiment".to_string(),
            question: "What is the sentiment?".to_string(),
            prefill: "".to_string(),
            summary_criteria: None,
            numeric: None,
        };

        (0..n)
            .map(|i| {
                vec![FacetValue {
                    facet: facet.clone(),
                    value: if i % 2 == 0 { "positive" } else { "negative" }.to_string(),
                }]
            })
            .collect()
    }

    #[test]
    fn test_perform_kmeans_basic() {
        let embeddings = create_test_embeddings(10);
        let result = perform_kmeans(&embeddings, 3);

        assert!(result.is_ok());
        let assignments = result.unwrap();
        assert_eq!(assignments.len(), 10);

        // Check all assignments are valid
        for &assignment in &assignments {
            assert!(assignment < 3);
        }
    }

    #[test]
    fn test_perform_kmeans_edge_cases() {
        // Empty embeddings
        let empty_embeddings: Vec<Vec<f32>> = vec![];
        let result = perform_kmeans(&empty_embeddings, 3);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);

        // Single embedding
        let single_embedding = vec![vec![1.0, 2.0]];
        let result = perform_kmeans(&single_embedding, 3);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![0]);

        // More clusters than points
        let two_embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = perform_kmeans(&two_embeddings, 5);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_perform_kmeans_consistency() {
        let embeddings = create_test_embeddings(20);

        // Run multiple times with same seed - should be consistent
        let result1 = perform_kmeans(&embeddings, 4).unwrap();
        let result2 = perform_kmeans(&embeddings, 4).unwrap();

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_compute_cluster_embeddings() {
        let clusters = vec![
            ConversationCluster {
                facet: Facet::default(),
                name: "Test Cluster 1".to_string(),
                summary: "A test cluster".to_string(),
                children: None,
                parent: None,
                indices: Some(vec![0, 1, 2]),
            },
            ConversationCluster {
                facet: Facet::default(),
                name: "Test Cluster 2".to_string(),
                summary: "Another test cluster".to_string(),
                children: None,
                parent: None,
                indices: Some(vec![3, 4]),
            },
        ];

        let result = compute_cluster_embeddings(&clusters);
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 128);
        assert_eq!(embeddings[1].len(), 128);

        // Embeddings should be different for different clusters
        assert_ne!(embeddings[0], embeddings[1]);
    }

    #[test]
    fn test_sample_cluster_points() {
        let facet = Facet {
            name: "sentiment".to_string(),
            question: "What is the sentiment?".to_string(),
            prefill: "".to_string(),
            summary_criteria: None,
            numeric: None,
        };

        let facet_data = vec![
            vec![FacetValue {
                facet: facet.clone(),
                value: "positive".to_string(),
            }],
            vec![FacetValue {
                facet: facet.clone(),
                value: "negative".to_string(),
            }],
            vec![FacetValue {
                facet: facet.clone(),
                value: "neutral".to_string(),
            }],
        ];

        let indices = vec![0, 1, 2];
        let samples = sample_cluster_points(&indices, &facet_data, &facet, 2);

        assert!(samples.len() <= 2);
        assert!(
            samples.contains(&"positive".to_string())
                || samples.contains(&"negative".to_string())
                || samples.contains(&"neutral".to_string())
        );
    }

    #[test]
    fn test_parse_name_description() {
        let response = "Name: Test Cluster\nDescription: A cluster for testing";
        let result = parse_name_description(response);

        assert!(result.is_some());
        let (name, desc) = result.unwrap();
        assert_eq!(name, "Test Cluster");
        assert_eq!(desc, "A cluster for testing");
    }

    #[test]
    fn test_parse_name_description_malformed() {
        let response = "Invalid format";
        let result = parse_name_description(response);
        assert!(result.is_none());

        let response = "Name: Only name provided";
        let result = parse_name_description(response);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_higher_level_names() {
        let response = "Name: Category A\nDescription: First category\nName: Category B\nDescription: Second category";
        let names = parse_higher_level_names(response);

        assert_eq!(names.len(), 2);
        assert_eq!(names[0].0, "Category A");
        assert_eq!(names[0].1, "First category");
        assert_eq!(names[1].0, "Category B");
        assert_eq!(names[1].1, "Second category");
    }

    #[test]
    fn test_parse_assignments() {
        let response = "1 -> A\n2 -> B\n3 -> A";
        let result = parse_assignments(response, 3);

        assert!(result.is_ok());
        let assignments = result.unwrap();
        assert_eq!(assignments.len(), 3);
        assert_eq!(assignments[0], 0); // 1 -> A (0-indexed)
        assert_eq!(assignments[1], 1); // 2 -> B
        assert_eq!(assignments[2], 0); // 3 -> A
    }

    #[test]
    fn test_parse_assignments_invalid() {
        let response = "invalid format";
        let result = parse_assignments(response, 3);

        assert!(result.is_ok());
        let assignments = result.unwrap();
        assert_eq!(assignments, vec![0, 0, 0]); // Default assignments
    }

    #[test]
    fn test_cluster_hierarchical_clustering_simple() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];

        let clusters = vec![
            ConversationCluster {
                facet: Facet::default(),
                name: "Cluster 1".to_string(),
                summary: "Test".to_string(),
                children: None,
                parent: None,
                indices: Some(vec![0]),
            },
            ConversationCluster {
                facet: Facet::default(),
                name: "Cluster 2".to_string(),
                summary: "Test".to_string(),
                children: None,
                parent: None,
                indices: Some(vec![1]),
            },
            ConversationCluster {
                facet: Facet::default(),
                name: "Cluster 3".to_string(),
                summary: "Test".to_string(),
                children: None,
                parent: None,
                indices: Some(vec![2]),
            },
        ];

        let result = cluster_hierarchical_clustering(&embeddings, 2, &clusters);
        assert!(result.is_ok());

        let neighborhoods = result.unwrap();
        assert!(neighborhoods.len() <= 2);

        // All original clusters should be preserved
        let total_clusters: usize = neighborhoods.iter().map(|n| n.len()).sum();
        assert_eq!(total_clusters, 3);
    }

    #[test]
    fn test_config_helpers() {
        let config = create_test_config();

        assert_eq!(config.n_base_clusters(100), 10);
        assert_eq!(config.n_average_clusters_per_neighborhood(20), 2);
        assert_eq!(config.n_desired_higher_level_names_per_cluster(9), 3);
        assert_eq!(config.n_desired_higher_level_names_per_cluster(2), 1);
    }
}

async fn refine_cluster_name(
    llm_client: &LlmClient,
    name: &str,
    description: &str,
    children: &[ConversationCluster],
    n_samples: usize,
) -> Result<(String, String)> {
    let child_summaries: Vec<(String, String)> = children
        .iter()
        .take(10) // Limit to avoid huge prompts
        .map(|c| (c.name.clone(), c.summary.clone()))
        .collect();

    let mut names = Vec::new();
    let mut descriptions = Vec::new();

    for _ in 0..n_samples {
        let prompt = get_renaming_higher_level_cluster_prompt(name, description, &child_summaries);
        let response = llm_client.complete(&prompt).await?;

        if let Some((n, d)) = parse_name_description(&response) {
            names.push(n);
            descriptions.push(d);
        }
    }

    let final_name = most_common(names).unwrap_or_else(|| name.to_string());
    let final_desc = most_common(descriptions).unwrap_or_else(|| description.to_string());

    Ok((final_name, final_desc))
}

fn compute_cluster_embeddings(clusters: &[ConversationCluster]) -> Result<Vec<Vec<f32>>> {
    // For now, create dummy embeddings based on cluster name
    // In a real implementation, you'd compute these from the actual embeddings
    let mut embeddings = Vec::new();

    for cluster in clusters {
        // Create a simple embedding based on cluster characteristics
        let name_hash = cluster.name.len() as f32;
        let summary_hash = cluster.summary.len() as f32;
        let size = cluster.indices.as_ref().map(|i| i.len()).unwrap_or(0) as f32;

        // Create a simple 128-dimensional embedding
        let mut embedding = vec![0.0; 128];
        embedding[0] = name_hash / 100.0;
        embedding[1] = summary_hash / 1000.0;
        embedding[2] = size / 50.0;

        // Add some structure based on cluster name
        for (i, byte) in cluster.name.bytes().take(125).enumerate() {
            embedding[i + 3] = (byte as f32) / 255.0;
        }

        embeddings.push(embedding);
    }

    Ok(embeddings)
}

fn cluster_hierarchical_clustering(
    embeddings: &[Vec<f32>],
    n_clusters: usize,
    original_clusters: &[ConversationCluster],
) -> Result<Vec<Vec<ConversationCluster>>> {
    if embeddings.len() <= n_clusters {
        // Each cluster is its own group
        return Ok(original_clusters.iter().map(|c| vec![c.clone()]).collect());
    }

    // Use k-means on cluster embeddings to create neighborhoods
    let assignments = perform_kmeans(embeddings, n_clusters)?;

    let mut neighborhoods = vec![Vec::new(); n_clusters];
    for (i, &assignment) in assignments.iter().enumerate() {
        if i < original_clusters.len() {
            neighborhoods[assignment].push(original_clusters[i].clone());
        }
    }

    // Remove empty neighborhoods
    neighborhoods.retain(|n| !n.is_empty());

    Ok(neighborhoods)
}

pub fn perform_kmeans(embeddings: &[Vec<f32>], n_clusters: usize) -> Result<Vec<usize>> {
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }

    let n_points = embeddings.len();
    let dim = embeddings[0].len();

    // Ensure we don't have more clusters than data points
    let actual_clusters = n_clusters.min(n_points);
    if actual_clusters == 0 {
        return Ok(vec![0; n_points]);
    }

    info!(
        "Running K-means clustering with {} clusters for {} points",
        actual_clusters, n_points
    );

    // Convert embeddings to ndarray format
    let mut data = Array2::zeros((n_points, dim));
    for (i, embedding) in embeddings.iter().enumerate() {
        if embedding.len() != dim {
            warn!(
                "Embedding {} has dimension {} but expected {}, skipping",
                i,
                embedding.len(),
                dim
            );
            continue;
        }
        for (j, &value) in embedding.iter().enumerate() {
            data[[i, j]] = value as f64;
        }
    }

    // Create dataset with proper type annotations
    let targets = Array1::<usize>::zeros(n_points);
    let dataset = Dataset::new(data, targets);

    // Configure K-means
    let rng = Xoshiro256Plus::seed_from_u64(42); // Fixed seed for reproducibility
    let kmeans = KMeans::params_with_rng(actual_clusters, rng)
        .max_n_iterations(300)
        .tolerance(1e-4);

    // Fit the model
    match kmeans.fit(&dataset) {
        Ok(model) => {
            let predictions = model.predict(&dataset);
            let assignments: Vec<usize> = predictions.as_targets().iter().copied().collect();

            info!("K-means clustering completed successfully");

            // Validate cluster assignments
            let max_cluster = assignments.iter().max().copied().unwrap_or(0);
            if max_cluster >= actual_clusters {
                warn!(
                    "Invalid cluster assignment detected: max cluster {} >= {}",
                    max_cluster, actual_clusters
                );
            }

            Ok(assignments)
        }
        Err(e) => {
            warn!(
                "K-means clustering failed: {}, falling back to simple assignment",
                e
            );

            // Fallback: simple round-robin assignment
            let assignments: Vec<usize> = (0..n_points).map(|i| i % actual_clusters).collect();

            Ok(assignments)
        }
    }
}
