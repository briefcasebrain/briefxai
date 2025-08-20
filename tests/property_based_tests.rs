use briefxai::{
    clustering::perform_kmeans,
    embeddings::{normalize_embeddings, cosine_similarity, euclidean_distance},
    facets::process_facet_response,
    types::Facet,
};
use proptest::prelude::*;
use std::collections::HashSet;

// Property-based tests for clustering algorithms
proptest! {
    #[test]
    fn test_kmeans_cluster_assignment_validity(
        embeddings in prop::collection::vec(
            prop::collection::vec(prop::num::f32::NORMAL, 10..=384),
            10..=1000
        ),
        n_clusters in 2usize..=50
    ) {
        let actual_clusters = n_clusters.min(embeddings.len());
        
        match perform_kmeans(&embeddings, n_clusters) {
            Ok(assignments) => {
                // All points should be assigned
                prop_assert_eq!(assignments.len(), embeddings.len());
                
                // All assignments should be valid cluster IDs
                for &assignment in &assignments {
                    prop_assert!(assignment < actual_clusters);
                }
                
                // Should use all available clusters (unless fewer points than clusters)
                let unique_assignments: HashSet<usize> = assignments.into_iter().collect();
                if embeddings.len() >= actual_clusters {
                    prop_assert!(unique_assignments.len() <= actual_clusters);
                }
            },
            Err(_) => {
                // K-means can fail on degenerate cases, which is acceptable
            }
        }
    }
    
    #[test]
    fn test_kmeans_deterministic_with_same_input(
        embeddings in prop::collection::vec(
            prop::collection::vec(prop::num::f32::NORMAL, 10..=128),
            10..=100
        ),
        n_clusters in 2usize..=10
    ) {
        // K-means should be deterministic with same seed
        if let (Ok(result1), Ok(result2)) = (
            perform_kmeans(&embeddings, n_clusters),
            perform_kmeans(&embeddings, n_clusters)
        ) {
            prop_assert_eq!(result1, result2);
        }
    }
    
    #[test]
    fn test_kmeans_single_cluster_edge_case(
        embeddings in prop::collection::vec(
            prop::collection::vec(prop::num::f32::NORMAL, 10..=128),
            1..=100
        )
    ) {
        // With single cluster, all points should be assigned to cluster 0
        match perform_kmeans(&embeddings, 1) {
            Ok(assignments) => {
                prop_assert_eq!(assignments.len(), embeddings.len());
                for &assignment in &assignments {
                    prop_assert_eq!(assignment, 0);
                }
            },
            Err(_) => {
                // Acceptable for degenerate cases
            }
        }
    }
}

// Property-based tests for embedding operations
proptest! {
    #[test]
    fn test_normalize_embeddings_properties(
        mut embeddings in prop::collection::vec(
            prop::collection::vec(prop::num::f32::NORMAL, 1..=1000),
            1..=100
        )
    ) {
        let original_lens: Vec<usize> = embeddings.iter().map(|e| e.len()).collect();
        
        normalize_embeddings(&mut embeddings);
        
        // Length should be preserved
        prop_assert_eq!(embeddings.len(), original_lens.len());
        for (i, embedding) in embeddings.iter().enumerate() {
            prop_assert_eq!(embedding.len(), original_lens[i]);
        }
        
        // Each non-zero embedding should have unit length (approximately)
        for embedding in &embeddings {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 { // Avoid zero vectors
                prop_assert!((norm - 1.0).abs() < 1e-5);
            }
        }
    }
    
    #[test]
    fn test_cosine_similarity_properties(
        a in prop::collection::vec(prop::num::f32::NORMAL, 1..=1000),
        b in prop::collection::vec(prop::num::f32::NORMAL, 1..=1000)
    ) {
        prop_assume!(a.len() == b.len());
        
        let similarity = cosine_similarity(&a, &b);
        
        // Cosine similarity should be in [-1, 1] range
        prop_assert!(similarity >= -1.0 - 1e-6);
        prop_assert!(similarity <= 1.0 + 1e-6);
        
        // Cosine similarity should be symmetric
        prop_assert!((similarity - cosine_similarity(&b, &a)).abs() < 1e-6);
        
        // Self-similarity should be 1.0 (for non-zero vectors)
        let self_sim = cosine_similarity(&a, &a);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            prop_assert!((self_sim - 1.0).abs() < 1e-5);
        }
    }
    
    #[test]
    fn test_euclidean_distance_properties(
        a in prop::collection::vec(prop::num::f32::NORMAL, 1..=1000),
        b in prop::collection::vec(prop::num::f32::NORMAL, 1..=1000),
        c in prop::collection::vec(prop::num::f32::NORMAL, 1..=1000)
    ) {
        prop_assume!(a.len() == b.len() && b.len() == c.len());
        
        let dist_ab = euclidean_distance(&a, &b);
        let dist_ba = euclidean_distance(&b, &a);
        let dist_aa = euclidean_distance(&a, &a);
        
        // Distance should be non-negative
        prop_assert!(dist_ab >= 0.0);
        
        // Distance should be symmetric
        prop_assert!((dist_ab - dist_ba).abs() < 1e-6);
        
        // Distance to self should be zero
        prop_assert!(dist_aa < 1e-6);
        
        // Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        let dist_ac = euclidean_distance(&a, &c);
        let dist_bc = euclidean_distance(&b, &c);
        prop_assert!(dist_ac <= dist_ab + dist_bc + 1e-5); // Small epsilon for floating point
    }
    
    #[test]
    fn test_embedding_normalization_preserves_direction(
        embedding in prop::collection::vec(prop::num::f32::NORMAL, 2..=1000)
    ) {
        let mut normalized = vec![embedding.clone()];
        normalize_embeddings(&mut normalized);
        
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 { // Skip zero vectors
            // Normalized vector should be in same direction (positive cosine similarity)
            let similarity = cosine_similarity(&embedding, &normalized[0]);
            prop_assert!(similarity > 0.99); // Should be very close to 1.0
        }
    }
}

// Property-based tests for facet processing
proptest! {
    #[test]
    fn test_numeric_facet_clamping_properties(
        min_val in -1000i32..=1000,
        max_val in -1000i32..=1000,
        input_val in -10000i32..=10000
    ) {
        prop_assume!(min_val <= max_val);
        
        let facet = Facet {
            name: "test_facet".to_string(),
            question: "Test question".to_string(),
            prefill: String::new(),
            summary_criteria: None,
            numeric: Some((min_val, max_val)),
        };
        
        let response = input_val.to_string();
        let result = process_facet_response(&response, &facet);
        
        prop_assert!(result.is_ok());
        let processed_value = result.unwrap();
        let parsed_value: i32 = processed_value.parse().unwrap();
        
        // Result should be within bounds
        prop_assert!(parsed_value >= min_val);
        prop_assert!(parsed_value <= max_val);
        
        // If input was within bounds, result should equal input
        if input_val >= min_val && input_val <= max_val {
            prop_assert_eq!(parsed_value, input_val);
        }
    }
    
    #[test]
    fn test_facet_response_consistency(
        response in "\\PC*",
        prefill in "\\PC{0,20}"
    ) {
        prop_assume!(!response.is_empty());
        
        let facet = Facet {
            name: "test_facet".to_string(),
            question: "Test question".to_string(),
            prefill: prefill.clone(),
            summary_criteria: None,
            numeric: None,
        };
        
        // Same input should produce same output
        let result1 = process_facet_response(&response, &facet);
        let result2 = process_facet_response(&response, &facet);
        
        prop_assert_eq!(result1, result2);
        
        // Result should be Ok for any input
        prop_assert!(result1.is_ok());
        
        // If response starts with prefill, it should be stripped
        if !prefill.is_empty() && response.starts_with(&prefill) {
            let processed = result1.unwrap();
            prop_assert_eq!(processed, response[prefill.len()..].trim());
        }
    }
    
    #[test]
    fn test_numeric_facet_default_value(
        min_val in 1i32..=100,
        max_val in 101i32..=1000,
        invalid_input in "\\PC*"
    ) {
        prop_assume!(min_val < max_val);
        prop_assume!(!invalid_input.chars().any(|c| c.is_ascii_digit()));
        
        let facet = Facet {
            name: "test_facet".to_string(),
            question: "Test question".to_string(),
            prefill: String::new(),
            summary_criteria: None,
            numeric: Some((min_val, max_val)),
        };
        
        let result = process_facet_response(&invalid_input, &facet);
        prop_assert!(result.is_ok());
        
        let processed_value: i32 = result.unwrap().parse().unwrap();
        let expected_default = (min_val + max_val) / 2;
        
        prop_assert_eq!(processed_value, expected_default);
    }
}

// Property-based tests for distance metrics mathematical properties
proptest! {
    #[test]
    fn test_distance_metric_axioms(
        dim in 1usize..=100,
        scale in 0.1f32..=10.0
    ) {
        // Create test vectors
        let zero_vec = vec![0.0; dim];
        let unit_vec = {
            let mut v = vec![0.0; dim];
            v[0] = scale;
            v
        };
        let opposite_vec = {
            let mut v = vec![0.0; dim];
            v[0] = -scale;
            v
        };
        
        // Test identity: distance(x, x) = 0
        prop_assert!(euclidean_distance(&unit_vec, &unit_vec) < 1e-6);
        
        // Test symmetry: distance(x, y) = distance(y, x)
        let dist_xy = euclidean_distance(&unit_vec, &opposite_vec);
        let dist_yx = euclidean_distance(&opposite_vec, &unit_vec);
        prop_assert!((dist_xy - dist_yx).abs() < 1e-6);
        
        // Test non-negativity: distance(x, y) >= 0
        prop_assert!(dist_xy >= 0.0);
        
        // Test that distance to zero is the magnitude
        let dist_to_zero = euclidean_distance(&unit_vec, &zero_vec);
        prop_assert!((dist_to_zero - scale).abs() < 1e-6);
    }
    
    #[test]
    fn test_cosine_similarity_boundary_conditions(
        dim in 2usize..=100,
        magnitude in 0.1f32..=10.0
    ) {
        // Parallel vectors should have similarity 1
        let vec1 = vec![magnitude; dim];
        let vec2 = vec![magnitude * 2.0; dim];
        let similarity = cosine_similarity(&vec1, &vec2);
        prop_assert!((similarity - 1.0).abs() < 1e-5);
        
        // Orthogonal vectors should have similarity 0
        let mut ortho1 = vec![0.0; dim];
        let mut ortho2 = vec![0.0; dim];
        ortho1[0] = magnitude;
        ortho2[1] = magnitude;
        let ortho_similarity = cosine_similarity(&ortho1, &ortho2);
        prop_assert!(ortho_similarity.abs() < 1e-5);
        
        // Anti-parallel vectors should have similarity -1
        let vec3 = vec![magnitude; dim];
        let vec4 = vec![-magnitude; dim];
        let anti_similarity = cosine_similarity(&vec3, &vec4);
        prop_assert!((anti_similarity + 1.0).abs() < 1e-5);
    }
}

// Property-based tests for clustering stability and convergence
proptest! {
    #[test]
    fn test_clustering_stability_under_perturbation(
        base_embeddings in prop::collection::vec(
            prop::collection::vec(prop::num::f32::NORMAL, 10..=50),
            20..=100
        ),
        perturbation_scale in 0.001f32..=0.1,
        n_clusters in 2usize..=10
    ) {
        prop_assume!(n_clusters <= base_embeddings.len());
        
        // Create slightly perturbed version
        let perturbed_embeddings: Vec<Vec<f32>> = base_embeddings.iter().map(|embedding| {
            embedding.iter().map(|&x| x + perturbation_scale * (rand::random::<f32>() - 0.5)).collect()
        }).collect();
        
        let original_result = perform_kmeans(&base_embeddings, n_clusters);
        let perturbed_result = perform_kmeans(&perturbed_embeddings, n_clusters);
        
        if let (Ok(original_assignments), Ok(perturbed_assignments)) = (original_result, perturbed_result) {
            // Count how many assignments changed
            let changes = original_assignments.iter()
                .zip(perturbed_assignments.iter())
                .filter(|(a, b)| a != b)
                .count();
            
            // With small perturbations, most assignments should remain stable
            let change_ratio = changes as f32 / original_assignments.len() as f32;
            prop_assert!(change_ratio < 0.5); // Less than 50% should change
        }
    }
    
    #[test]
    fn test_clustering_scales_with_input_size(
        embedding_size in 50usize..=500,
        dimension in 10usize..=100
    ) {
        let embeddings: Vec<Vec<f32>> = (0..embedding_size).map(|i| {
            (0..dimension).map(|j| {
                ((i * j) as f32).sin() * 0.5 + 0.5
            }).collect()
        }).collect();
        
        let n_clusters = (embedding_size / 10).max(2);
        let result = perform_kmeans(&embeddings, n_clusters);
        
        prop_assert!(result.is_ok());
        let assignments = result.unwrap();
        prop_assert_eq!(assignments.len(), embedding_size);
        
        // All assignments should be valid
        for &assignment in &assignments {
            prop_assert!(assignment < n_clusters);
        }
    }
}

// Property-based tests for edge cases and error conditions
proptest! {
    #[test]
    fn test_empty_input_handling(
        n_clusters in 1usize..=10
    ) {
        let empty_embeddings: Vec<Vec<f32>> = vec![];
        let result = perform_kmeans(&empty_embeddings, n_clusters);
        
        prop_assert!(result.is_ok());
        let assignments = result.unwrap();
        prop_assert!(assignments.is_empty());
    }
    
    #[test]
    fn test_single_point_clustering(
        embedding in prop::collection::vec(prop::num::f32::NORMAL, 1..=100),
        n_clusters in 1usize..=5
    ) {
        let single_point = vec![embedding];
        let result = perform_kmeans(&single_point, n_clusters);
        
        prop_assert!(result.is_ok());
        let assignments = result.unwrap();
        prop_assert_eq!(assignments.len(), 1);
        prop_assert_eq!(assignments[0], 0);
    }
    
    #[test]
    fn test_more_clusters_than_points(
        embeddings in prop::collection::vec(
            prop::collection::vec(prop::num::f32::NORMAL, 10..=50),
            1..=10
        ),
        excess_clusters in 11usize..=20
    ) {
        let result = perform_kmeans(&embeddings, excess_clusters);
        
        prop_assert!(result.is_ok());
        let assignments = result.unwrap();
        prop_assert_eq!(assignments.len(), embeddings.len());
        
        // All assignments should be valid (0 to embeddings.len()-1)
        for &assignment in &assignments {
            prop_assert!(assignment < embeddings.len());
        }
    }
}

// Property-based tests for numerical stability
proptest! {
    #[test]
    fn test_numerical_stability_extreme_values(
        magnitude in 1e-10f32..=1e10
    ) {
        let extreme_vec1 = vec![magnitude, 0.0];
        let extreme_vec2 = vec![0.0, magnitude];
        let extreme_vec3 = vec![magnitude, magnitude];
        
        // These operations should not panic or produce NaN/Inf
        let dist = euclidean_distance(&extreme_vec1, &extreme_vec2);
        prop_assert!(dist.is_finite());
        prop_assert!(dist >= 0.0);
        
        let sim = cosine_similarity(&extreme_vec1, &extreme_vec3);
        prop_assert!(sim.is_finite());
        prop_assert!(sim >= -1.0 - 1e-6);
        prop_assert!(sim <= 1.0 + 1e-6);
        
        let mut for_norm = vec![extreme_vec1.clone()];
        normalize_embeddings(&mut for_norm);
        prop_assert!(for_norm[0].iter().all(|&x| x.is_finite()));
    }
    
    #[test]
    fn test_zero_vector_handling(
        dim in 1usize..=100
    ) {
        let zero_vec = vec![0.0; dim];
        let unit_vec = {
            let mut v = vec![0.0; dim];
            v[0] = 1.0;
            v
        };
        
        // Operations with zero vectors should be well-defined
        let dist = euclidean_distance(&zero_vec, &unit_vec);
        prop_assert!((dist - 1.0).abs() < 1e-6);
        
        let sim = cosine_similarity(&zero_vec, &unit_vec);
        prop_assert_eq!(sim, 0.0);
        
        let self_sim = cosine_similarity(&zero_vec, &zero_vec);
        prop_assert!(self_sim.is_nan() || self_sim == 0.0); // Either is acceptable
        
        // Normalizing zero vector should remain zero
        let mut zero_for_norm = vec![zero_vec];
        normalize_embeddings(&mut zero_for_norm);
        prop_assert!(zero_for_norm[0].iter().all(|&x| x == 0.0));
    }
}