use anyhow::{Result, Context};
use tracing::{info, debug, warn};
use ndarray::{Array2, Array1, Axis};
use linfa::prelude::*;
use linfa_reduction::Pca;
use rand::{SeedableRng, Rng};
use rand_xoshiro::Xoshiro256Plus;

use crate::config::BriefXAIConfig;

pub async fn generate_umap(
    config: &BriefXAIConfig,
    embeddings: &[Vec<f32>],
) -> Result<Vec<(f32, f32)>> {
    info!("Generating UMAP-like visualization for {} points", embeddings.len());
    
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }
    
    let n_points = embeddings.len();
    let dim = embeddings[0].len();
    
    if n_points < 3 {
        warn!("Too few points for meaningful dimensionality reduction, using simple projection");
        return generate_simple_2d_projection(embeddings, config.seed).await;
    }
    
    debug!("Input: {} points with {} dimensions", n_points, dim);
    
    // Try PCA first as it's more robust
    match generate_pca_projection(embeddings, config.seed).await {
        Ok(coords) => {
            info!("Generated PCA-based 2D projection");
            Ok(coords)
        }
        Err(e) => {
            warn!("PCA failed: {}, falling back to simple projection", e);
            generate_simple_2d_projection(embeddings, config.seed).await
        }
    }
}

pub async fn generate_simple_2d_projection(
    embeddings: &[Vec<f32>],
    seed: u64,
) -> Result<Vec<(f32, f32)>> {
    use rand::{SeedableRng, Rng};
    use rand::rngs::StdRng;
    
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }
    
    let dim = embeddings[0].len();
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Generate two random projection vectors
    let proj1: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let proj2: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    // Normalize projection vectors
    let norm1: f32 = proj1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = proj2.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    let proj1: Vec<f32> = proj1.iter().map(|x| x / norm1).collect();
    let proj2: Vec<f32> = proj2.iter().map(|x| x / norm2).collect();
    
    // Project embeddings
    let coords: Vec<(f32, f32)> = embeddings
        .iter()
        .map(|emb| {
            let x: f32 = emb.iter().zip(&proj1).map(|(a, b)| a * b).sum();
            let y: f32 = emb.iter().zip(&proj2).map(|(a, b)| a * b).sum();
            (x, y)
        })
        .collect();
    
    // Normalize to [-1, 1] range
    let (min_x, max_x) = coords
        .iter()
        .map(|(x, _)| x)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), x| {
            (min.min(*x), max.max(*x))
        });
    
    let (min_y, max_y) = coords
        .iter()
        .map(|(_, y)| y)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), y| {
            (min.min(*y), max.max(*y))
        });
    
    let range_x = max_x - min_x;
    let range_y = max_y - min_y;
    
    let normalized_coords: Vec<(f32, f32)> = coords
        .iter()
        .map(|(x, y)| {
            let norm_x = if range_x > 0.0 {
                2.0 * (x - min_x) / range_x - 1.0
            } else {
                0.0
            };
            let norm_y = if range_y > 0.0 {
                2.0 * (y - min_y) / range_y - 1.0
            } else {
                0.0
            };
            (norm_x, norm_y)
        })
        .collect();
    
    Ok(normalized_coords)
}

pub async fn generate_pca_projection(
    embeddings: &[Vec<f32>],
    seed: u64,
) -> Result<Vec<(f32, f32)>> {
    let n_points = embeddings.len();
    let dim = embeddings[0].len();
    
    debug!("Running PCA on {} points with {} dimensions", n_points, dim);
    
    // Convert embeddings to ndarray format
    let mut data = Array2::zeros((n_points, dim));
    for (i, embedding) in embeddings.iter().enumerate() {
        if embedding.len() != dim {
            return Err(anyhow::anyhow!("Inconsistent embedding dimensions"));
        }
        for (j, &value) in embedding.iter().enumerate() {
            data[[i, j]] = value as f64;
        }
    }
    
    // Center the data (subtract mean)
    let mean = data.mean_axis(Axis(0)).context("Failed to compute mean")?;
    for mut row in data.rows_mut() {
        row -= &mean;
    }
    
    // Create dataset (PCA doesn't need targets)
    let targets = Array1::<usize>::zeros(n_points);
    let dataset = Dataset::new(data, targets);
    
    // Configure and fit PCA
    let pca = Pca::params(2); // Project to 2D
    let pca_model = pca.fit(&dataset)
        .context("Failed to fit PCA model")?;
    
    // Transform the data
    let transformed = pca_model.predict(&dataset);
    
    // Extract 2D coordinates directly from the transformed array
    let mut coords = Vec::new();
    for i in 0..n_points {
        let x = transformed[[i, 0]] as f32;
        let y = transformed[[i, 1]] as f32;
        coords.push((x, y));
    }
    
    // Normalize coordinates to [-1, 1] range
    let normalized_coords = normalize_coordinates(&coords)?;
    
    debug!("PCA projection completed successfully");
    Ok(normalized_coords)
}

pub async fn generate_tsne_like_projection(
    embeddings: &[Vec<f32>],
    seed: u64,
) -> Result<Vec<(f32, f32)>> {
    // Simplified t-SNE inspired approach using local neighborhood preservation
    let n_points = embeddings.len();
    let rng = Xoshiro256Plus::seed_from_u64(seed);
    
    debug!("Generating t-SNE-like projection for {} points", n_points);
    
    // Initialize with PCA as starting point
    let mut coords = generate_pca_projection(embeddings, seed).await?;
    
    // Simple neighborhood-preserving iterations
    let n_iterations = 100;
    let learning_rate = 0.1;
    
    for iteration in 0..n_iterations {
        let mut forces = vec![(0.0f32, 0.0f32); n_points];
        
        // Compute pairwise forces (simplified)
        for i in 0..n_points {
            for j in (i + 1)..n_points {
                // High-dimensional distance
                let high_dist = euclidean_distance(&embeddings[i], &embeddings[j]);
                
                // Low-dimensional distance
                let dx = coords[i].0 - coords[j].0;
                let dy = coords[i].1 - coords[j].1;
                let low_dist = (dx * dx + dy * dy).sqrt();
                
                if low_dist > 0.0 {
                    // Force proportional to distance difference
                    let force_magnitude = (high_dist - low_dist) * learning_rate;
                    let force_x = force_magnitude * dx / low_dist;
                    let force_y = force_magnitude * dy / low_dist;
                    
                    forces[i].0 += force_x;
                    forces[i].1 += force_y;
                    forces[j].0 -= force_x;
                    forces[j].1 -= force_y;
                }
            }
        }
        
        // Apply forces with momentum
        for i in 0..n_points {
            coords[i].0 += forces[i].0 * 0.1;
            coords[i].1 += forces[i].1 * 0.1;
        }
        
        // Decay learning rate
        if iteration % 20 == 0 {
            debug!("t-SNE iteration {}/{}", iteration, n_iterations);
        }
    }
    
    // Final normalization
    let normalized_coords = normalize_coordinates(&coords)?;
    
    debug!("t-SNE-like projection completed");
    Ok(normalized_coords)
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn normalize_coordinates(coords: &[(f32, f32)]) -> Result<Vec<(f32, f32)>> {
    if coords.is_empty() {
        return Ok(Vec::new());
    }
    
    // Find bounds
    let (min_x, max_x) = coords
        .iter()
        .map(|(x, _)| x)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), x| {
            (min.min(*x), max.max(*x))
        });
    
    let (min_y, max_y) = coords
        .iter()
        .map(|(_, y)| y)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), y| {
            (min.min(*y), max.max(*y))
        });
    
    let range_x = max_x - min_x;
    let range_y = max_y - min_y;
    
    // Normalize to [-1, 1] range
    let normalized_coords: Vec<(f32, f32)> = coords
        .iter()
        .map(|(x, y)| {
            let norm_x = if range_x > 0.0 {
                2.0 * (x - min_x) / range_x - 1.0
            } else {
                0.0
            };
            let norm_y = if range_y > 0.0 {
                2.0 * (y - min_y) / range_y - 1.0
            } else {
                0.0
            };
            (norm_x, norm_y)
        })
        .collect();
    
    Ok(normalized_coords)
}

// Advanced UMAP-inspired projection using approximate nearest neighbors
pub async fn generate_advanced_umap_projection(
    embeddings: &[Vec<f32>],
    seed: u64,
    n_neighbors: usize,
) -> Result<Vec<(f32, f32)>> {
    let n_points = embeddings.len();
    let n_neighbors = n_neighbors.min(n_points - 1).max(2);
    
    info!("Generating advanced UMAP-like projection for {} points with {} neighbors", 
          n_points, n_neighbors);
    
    // Step 1: Find k-nearest neighbors for each point
    let neighbors = find_k_nearest_neighbors(embeddings, n_neighbors)?;
    
    // Step 2: Build graph with weights based on distances
    let graph = build_neighborhood_graph(embeddings, &neighbors)?;
    
    // Step 3: Initialize low-dimensional embedding
    let mut coords = initialize_embedding(n_points, seed)?;
    
    // Step 4: Optimize embedding using simplified UMAP objective
    optimize_embedding(&mut coords, &graph, seed)?;
    
    // Step 5: Final normalization
    let normalized_coords = normalize_coordinates(&coords)?;
    
    info!("Advanced UMAP-like projection completed");
    Ok(normalized_coords)
}

fn find_k_nearest_neighbors(
    embeddings: &[Vec<f32>],
    k: usize,
) -> Result<Vec<Vec<usize>>> {
    let n_points = embeddings.len();
    let mut neighbors = Vec::new();
    
    for i in 0..n_points {
        let mut distances: Vec<(f32, usize)> = (0..n_points)
            .filter(|&j| j != i)
            .map(|j| (euclidean_distance(&embeddings[i], &embeddings[j]), j))
            .collect();
        
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let neighbor_indices: Vec<usize> = distances
            .into_iter()
            .take(k)
            .map(|(_, idx)| idx)
            .collect();
        
        neighbors.push(neighbor_indices);
    }
    
    Ok(neighbors)
}

fn build_neighborhood_graph(
    embeddings: &[Vec<f32>],
    neighbors: &[Vec<usize>],
) -> Result<Vec<Vec<(usize, f32)>>> {
    let n_points = embeddings.len();
    let mut graph = vec![Vec::new(); n_points];
    
    for (i, neighbor_list) in neighbors.iter().enumerate() {
        for &j in neighbor_list {
            let distance = euclidean_distance(&embeddings[i], &embeddings[j]);
            let weight = (-distance).exp(); // Exponential decay
            graph[i].push((j, weight));
        }
    }
    
    Ok(graph)
}

fn initialize_embedding(n_points: usize, seed: u64) -> Result<Vec<(f32, f32)>> {
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);
    let coords: Vec<(f32, f32)> = (0..n_points)
        .map(|_| {
            let x = rng.gen_range(-1.0..1.0);
            let y = rng.gen_range(-1.0..1.0);
            (x, y)
        })
        .collect();
    
    Ok(coords)
}

fn optimize_embedding(
    coords: &mut [(f32, f32)],
    graph: &[Vec<(usize, f32)>],
    seed: u64,
) -> Result<()> {
    let n_iterations = 200;
    let learning_rate = 1.0;
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);
    
    for iteration in 0..n_iterations {
        let mut forces = vec![(0.0f32, 0.0f32); coords.len()];
        
        // Attractive forces (for neighbors)
        for (i, neighbors) in graph.iter().enumerate() {
            for &(j, weight) in neighbors {
                let dx = coords[j].0 - coords[i].0;
                let dy = coords[j].1 - coords[i].1;
                let dist_sq = dx * dx + dy * dy + 1e-8;
                let dist = dist_sq.sqrt();
                
                // Attractive force proportional to weight
                let force = weight * learning_rate / (1.0 + dist_sq);
                forces[i].0 += force * dx / dist;
                forces[i].1 += force * dy / dist;
            }
        }
        
        // Repulsive forces (sample random pairs)
        let n_negative_samples = coords.len().min(100);
        for _ in 0..n_negative_samples {
            let i = rng.gen_range(0..coords.len());
            let j = rng.gen_range(0..coords.len());
            if i != j {
                let dx = coords[j].0 - coords[i].0;
                let dy = coords[j].1 - coords[i].1;
                let dist_sq = dx * dx + dy * dy + 1e-8;
                let dist = dist_sq.sqrt();
                
                // Repulsive force
                let force = learning_rate / (1.0 + dist_sq);
                forces[i].0 -= force * dx / dist;
                forces[i].1 -= force * dy / dist;
            }
        }
        
        // Apply forces
        for i in 0..coords.len() {
            coords[i].0 += forces[i].0 * 0.1;
            coords[i].1 += forces[i].1 * 0.1;
        }
        
        if iteration % 50 == 0 {
            debug!("UMAP optimization iteration {}/{}", iteration, n_iterations);
        }
    }
    
    Ok(())
}