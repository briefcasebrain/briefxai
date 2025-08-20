use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use briefxai::{
    BriefXAIConfig,
    clustering::{create_base_clusters, build_hierarchy, perform_kmeans},
    embeddings::{generate_embeddings, normalize_embeddings, cosine_similarity, euclidean_distance},
    facets::extract_facets,
    types::{ConversationData, Message, Facet, FacetValue},
    umap::generate_umap,
};
use std::collections::HashMap;
use tokio::runtime::Runtime;

// Helper functions for creating test data
fn create_test_conversations(n: usize) -> Vec<ConversationData> {
    (0..n).map(|i| {
        ConversationData {
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: format!("This is test conversation number {} with some content to analyze", i),
                },
                Message {
                    role: "assistant".to_string(),
                    content: format!("This is a response to conversation {} with helpful information", i),
                },
            ],
            metadata: HashMap::new(),
        }
    }).collect()
}

fn create_test_embeddings(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n).map(|i| {
        (0..dim).map(|j| {
            // Create somewhat realistic embeddings with some structure
            ((i * j) as f32).sin() * 0.5 + 0.5
        }).collect()
    }).collect()
}

fn create_test_facet_data(n: usize) -> Vec<Vec<FacetValue>> {
    let facet = Facet {
        name: "sentiment".to_string(),
        question: "What is the sentiment?".to_string(),
        prefill: "".to_string(),
        summary_criteria: None,
        numeric: None,
    };
    
    (0..n).map(|i| {
        vec![FacetValue {
            facet: facet.clone(),
            value: match i % 3 {
                0 => "positive",
                1 => "negative",
                _ => "neutral",
            }.to_string(),
        }]
    }).collect()
}

fn create_test_config() -> BriefXAIConfig {
    BriefXAIConfig {
        seed: 42,
        verbose: false,
        llm_batch_size: 100,
        embed_batch_size: 100,
        max_points_to_sample_inside_cluster: 10,
        max_points_to_sample_outside_cluster: 10,
        n_name_description_samples_per_cluster: 3,
        min_top_level_size: 5,
        umap_n_neighbors: 15,
        umap_min_dist: 0.1,
        umap_n_components: 2,
        llm_provider: briefxai::config::LlmProvider::OpenAI,
        llm_model: "gpt-4o-mini".to_string(),
        llm_api_key: Some("test-key".to_string()),
        ..Default::default()
    }
}

// Clustering benchmarks
fn benchmark_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_clustering");
    
    for size in [100, 500, 1000, 2000].iter() {
        let embeddings = create_test_embeddings(*size, 384);
        let n_clusters = size / 10;
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("kmeans", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    perform_kmeans(black_box(&embeddings), black_box(n_clusters))
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_clustering_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("clustering_pipeline");
    let rt = Runtime::new().unwrap();
    
    for size in [50, 100, 200].iter() {
        let embeddings = create_test_embeddings(*size, 384);
        let facet_data = create_test_facet_data(*size);
        let config = create_test_config();
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("base_clusters", size),
            size,
            |b, &_size| {
                b.to_async(&rt).iter(|| async {
                    create_base_clusters(
                        black_box(&config),
                        black_box(&embeddings),
                        black_box(&facet_data),
                    ).await
                });
            },
        );
    }
    
    group.finish();
}

// Embedding benchmarks
fn benchmark_embedding_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_operations");
    
    for size in [100, 1000, 5000, 10000].iter() {
        let mut embeddings = create_test_embeddings(*size, 384);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("normalize", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    normalize_embeddings(black_box(&mut embeddings.clone()));
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_distance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_metrics");
    
    for dim in [128, 384, 768, 1536].iter() {
        let a = create_test_embeddings(1, *dim)[0].clone();
        let b = create_test_embeddings(1, *dim)[0].clone();
        
        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(
            BenchmarkId::new("cosine_similarity", dim),
            dim,
            |b_bench, &_dim| {
                b_bench.iter(|| {
                    cosine_similarity(black_box(&a), black_box(&b))
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("euclidean_distance", dim),
            dim,
            |b_bench, &_dim| {
                b_bench.iter(|| {
                    euclidean_distance(black_box(&a), black_box(&b))
                });
            },
        );
    }
    
    group.finish();
}

// UMAP benchmarks
fn benchmark_umap_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("umap_generation");
    let rt = Runtime::new().unwrap();
    
    for size in [100, 500, 1000].iter() {
        let embeddings = create_test_embeddings(*size, 384);
        let config = create_test_config();
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("umap", size),
            size,
            |b, &_size| {
                b.to_async(&rt).iter(|| async {
                    generate_umap(black_box(&config), black_box(&embeddings)).await
                });
            },
        );
    }
    
    group.finish();
}

// Memory and scalability benchmarks
fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    for size in [1000, 5000, 10000, 20000].iter() {
        let embeddings = create_test_embeddings(*size, 384);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("large_dataset_kmeans", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    let n_clusters = (*size / 50).max(2);
                    perform_kmeans(black_box(&embeddings), black_box(n_clusters))
                });
            },
        );
    }
    
    group.finish();
}

// Clio features benchmarks
fn benchmark_clio_features(c: &mut Criterion) {
    use briefxai::{
        visualization::interactive_map::InteractiveMap,
        privacy::threshold_protection::PrivacyFilter,
        investigation::targeted_search::InvestigationEngine,
        discovery::serendipitous::DiscoveryEngine,
        types_extended::AnalysisCluster,
    };
    
    let mut group = c.benchmark_group("clio_features");
    let rt = Runtime::new().unwrap();
    
    for size in [100, 500, 1000].iter() {
        let conversations = create_test_conversations(*size);
        let facet_data = create_test_facet_data(*size);
        let umap_coords: Vec<(f32, f32)> = (0..*size).map(|i| {
            (i as f32 / *size as f32, (i * 2) as f32 / *size as f32)
        }).collect();
        
        // Create mock clusters
        let clusters: Vec<AnalysisCluster> = (0..(*size / 10)).map(|i| {
            AnalysisCluster {
                conversation_ids: vec![i * 10, i * 10 + 1, i * 10 + 2],
                name: format!("Cluster {}", i),
                description: format!("Test cluster {}", i),
                children: vec![],
            }
        }).collect();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("interactive_map_creation", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    InteractiveMap::new(
                        black_box(clusters.clone()),
                        black_box(umap_coords.clone()),
                        black_box(&facet_data),
                    )
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("privacy_filtering", size),
            size,
            |b, &_size| {
                b.to_async(&rt).iter(|| async {
                    let config = briefxai::privacy::threshold_protection::PrivacyConfig {
                        min_cluster_size: 10,
                        merge_small_clusters: true,
                        ..Default::default()
                    };
                    let mut filter = PrivacyFilter::new(config);
                    filter.filter_clusters(black_box(clusters.clone()), black_box(&conversations)).await
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("investigation_engine_creation", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    InvestigationEngine::new(
                        black_box(clusters.clone()),
                        black_box(conversations.clone()),
                        black_box(facet_data.clone()),
                        black_box(None),
                    )
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("discovery_engine_creation", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    DiscoveryEngine::new(
                        black_box(clusters.clone()),
                        black_box(facet_data.clone()),
                    )
                });
            },
        );
    }
    
    group.finish();
}

// Comprehensive end-to-end benchmark
fn benchmark_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    let rt = Runtime::new().unwrap();
    
    for size in [50, 100, 200].iter() {
        let conversations = create_test_conversations(*size);
        let config = create_test_config();
        
        group.throughput(Throughput::Elements(*size as u64));
        group.sample_size(10); // Smaller sample size for expensive operations
        
        group.bench_with_input(
            BenchmarkId::new("end_to_end", size),
            size,
            |b, &_size| {
                b.to_async(&rt).iter(|| async {
                    // Simulate the full pipeline without LLM calls
                    let embeddings = create_test_embeddings(*size, 384);
                    let facet_data = create_test_facet_data(*size);
                    
                    // Clustering
                    let base_clusters = create_base_clusters(
                        black_box(&config),
                        black_box(&embeddings),
                        black_box(&facet_data),
                    ).await;
                    
                    // UMAP
                    let _umap_coords = generate_umap(
                        black_box(&config),
                        black_box(&embeddings),
                    ).await;
                    
                    black_box(base_clusters)
                });
            },
        );
    }
    
    group.finish();
}

// Stress testing for large datasets
fn benchmark_stress_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_test");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));
    
    for size in [5000, 10000, 20000].iter() {
        let embeddings = create_test_embeddings(*size, 384);
        let n_clusters = size / 100;
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("large_scale_kmeans", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    perform_kmeans(black_box(&embeddings), black_box(n_clusters))
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_kmeans,
    benchmark_clustering_pipeline,
    benchmark_embedding_operations,
    benchmark_distance_metrics,
    benchmark_umap_generation,
    benchmark_memory_usage,
    benchmark_clio_features,
    benchmark_full_pipeline,
    benchmark_stress_test
);

criterion_main!(benches);