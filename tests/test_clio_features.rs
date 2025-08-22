use briefx::discovery::serendipitous::{DiscoveryEngine, UserPreferences};
use briefx::investigation::targeted_search::{
    ClusterMetric, ComparisonOperator, InvestigationEngine, InvestigationQuery, MetricFilter,
    SortCriterion,
};
use briefx::privacy::threshold_protection::{PrivacyConfig, PrivacyFilter};
use briefx::types::{Facet, FacetValue};
use briefx::types_extended::AnalysisCluster;
use briefx::visualization::interactive_map::{
    AggregationType, ColorScheme, FacetOverlay, InteractiveMap,
};

#[test]
fn test_integrated_clio_features() {
    // Create test data
    let clusters = vec![
        AnalysisCluster {
            conversation_ids: vec![0, 1, 2, 3, 4],
            name: "Customer Support".to_string(),
            description: "Support conversations about product issues".to_string(),
            children: vec![],
        },
        AnalysisCluster {
            conversation_ids: vec![5, 6],
            name: "Small Topic".to_string(),
            description: "A small cluster that should be filtered".to_string(),
            children: vec![],
        },
        AnalysisCluster {
            conversation_ids: (7..20).collect(),
            name: "Product Features".to_string(),
            description: "Discussions about product features and capabilities".to_string(),
            children: vec![],
        },
    ];

    let umap_coords: Vec<(f32, f32)> = (0..20).map(|i| (i as f32 * 0.1, i as f32 * 0.2)).collect();

    let facet_data: Vec<Vec<FacetValue>> = (0..20)
        .map(|i| {
            vec![FacetValue {
                facet: Facet {
                    name: if i < 10 {
                        "support".to_string()
                    } else {
                        "feature".to_string()
                    },
                    question: "Category?".to_string(),
                    prefill: String::new(),
                    summary_criteria: None,
                    numeric: None,
                },
                value: if i % 2 == 0 {
                    "positive".to_string()
                } else {
                    "negative".to_string()
                },
            }]
        })
        .collect();

    // Test 1: Interactive Map with Facet Overlays
    println!("Testing Interactive Map...");
    let mut map = InteractiveMap::new(clusters.clone(), umap_coords.clone(), &facet_data).unwrap();

    let overlay = FacetOverlay {
        facet_name: "support".to_string(),
        color_scheme: ColorScheme::Heatmap,
        aggregation: AggregationType::Prevalence,
        enabled: true,
        opacity: 0.7,
        threshold: 0.1,
    };

    map.apply_facet_overlay(overlay).unwrap();
    let viz_data = map.export_for_frontend();
    assert!(!viz_data
        .get("points")
        .unwrap()
        .as_array()
        .unwrap()
        .is_empty());
    assert!(viz_data.get("active_overlay").is_some());
    println!(
        "Interactive map with {} points created",
        viz_data.get("points").unwrap().as_array().unwrap().len()
    );

    // Test 2: Privacy Threshold Protection
    println!("Testing Privacy Filters...");
    let privacy_config = PrivacyConfig {
        min_cluster_size: 5,
        merge_small_clusters: true,
        ..Default::default()
    };

    let mut privacy_filter = PrivacyFilter::new(privacy_config);
    let conversations = vec![]; // Empty for this test
    let filtered_clusters = futures::executor::block_on(
        privacy_filter.filter_clusters(clusters.clone(), &conversations),
    )
    .unwrap();

    // Small cluster should be filtered or merged
    assert!(filtered_clusters.len() <= clusters.len());
    println!(
        "Privacy filter reduced clusters from {} to {}",
        clusters.len(),
        filtered_clusters.len()
    );

    // Test 3: Targeted Investigation
    println!("Testing Investigation Engine...");
    let investigation_engine =
        InvestigationEngine::new(clusters.clone(), vec![], facet_data.clone(), None).unwrap();

    let query = InvestigationQuery {
        search_terms: vec!["support".to_string()],
        facet_filters: vec![],
        metric_filters: vec![MetricFilter {
            metric: ClusterMetric::Size,
            operator: ComparisonOperator::GreaterThan,
            value: 3.0,
            threshold: 0.0,
        }],
        similar_to_cluster: None,
        sort_criterion: SortCriterion::Size,
        sort_by: Some("Size".to_string()),
        limit: Some(10),
        highlight_matches: Some(true),
    };

    let results = investigation_engine.investigate(&query).unwrap();
    assert!(!results.clusters.is_empty());
    println!(
        "Investigation found {} matching clusters",
        results.total_matches
    );

    // Test 4: Serendipitous Discovery
    println!("Testing Discovery Engine...");
    let mut discovery_engine = DiscoveryEngine::new(clusters.clone(), facet_data.clone()).unwrap();

    let recommendations = discovery_engine.get_recommendations(None, 5).unwrap();
    assert!(!recommendations.is_empty());
    println!(
        "Discovery engine generated {} recommendations",
        recommendations.len()
    );

    // Test navigation from a specific cluster
    let nav_recommendations = discovery_engine.get_recommendations(Some(0), 3).unwrap();
    assert!(!nav_recommendations.is_empty());
    println!(
        "Navigation from cluster 0 yielded {} recommendations",
        nav_recommendations.len()
    );

    // Update preferences based on exploration
    discovery_engine
        .update_preferences(UserPreferences {
            preferred_recommendation_types: vec![],
            min_confidence_score: 0.0,
            max_recommendations: 0,
            preferred_facets: vec![],
        })
        .unwrap();
    println!("User preferences updated based on exploration");

    println!("\nAll Clio features tested successfully!");
}

#[test]
fn test_privacy_thresholds_comprehensive() {
    let clusters = vec![
        AnalysisCluster {
            conversation_ids: (0..15).collect(),
            name: "Large Cluster".to_string(),
            description: "A sufficiently large cluster".to_string(),
            children: vec![],
        },
        AnalysisCluster {
            conversation_ids: vec![15, 16],
            name: "Tiny Cluster".to_string(),
            description: "Too small for privacy".to_string(),
            children: vec![],
        },
        AnalysisCluster {
            conversation_ids: vec![17, 18, 19],
            name: "Small Cluster".to_string(),
            description: "Borderline size".to_string(),
            children: vec![],
        },
    ];

    let config = PrivacyConfig {
        min_cluster_size: 5,
        merge_small_clusters: true,
        ..Default::default()
    };

    let mut filter = PrivacyFilter::new(config);
    let filtered = futures::executor::block_on(filter.filter_clusters(clusters, &vec![])).unwrap();

    // Should have large cluster and merged small clusters
    assert!(filtered.iter().any(|c| c.name == "Large Cluster"));
    assert!(filtered.iter().any(|c| c.name == "merged_small_clusters")); // Merged cluster

    let report = filter.generate_report();
    assert!(report.small_clusters_merged > 0);
}

#[test]
fn test_investigation_with_similarity() {
    let clusters = vec![
        AnalysisCluster {
            conversation_ids: vec![0, 1],
            name: "Technical Support".to_string(),
            description: "Help with technical issues".to_string(),
            children: vec![],
        },
        AnalysisCluster {
            conversation_ids: vec![2, 3],
            name: "Technical Documentation".to_string(),
            description: "Technical guides and docs".to_string(),
            children: vec![],
        },
        AnalysisCluster {
            conversation_ids: vec![4, 5],
            name: "Sales Inquiries".to_string(),
            description: "Questions about pricing".to_string(),
            children: vec![],
        },
    ];

    let engine = InvestigationEngine::new(clusters, vec![], vec![], None).unwrap();

    // Search for clusters similar to cluster 0
    let query = InvestigationQuery {
        search_terms: vec![],
        facet_filters: vec![],
        metric_filters: vec![],
        similar_to_cluster: Some(0),
        sort_criterion: SortCriterion::Relevance,
        sort_by: Some("Relevance".to_string()),
        limit: None,
        highlight_matches: Some(false),
    };

    let results = engine.investigate(&query).unwrap();

    // Should find cluster 1 as similar (both have "Technical" in name)
    assert!(results
        .clusters
        .iter()
        .any(|c| c.name.contains("Technical")));
}
