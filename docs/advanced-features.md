# BriefXAI Clio Features Documentation

This document describes the advanced Clio features implemented in BriefXAI, inspired by recent research on privacy-preserving insights into real-world AI use.

## Overview

The Clio features provide four main capabilities:

1. **Interactive 2D Map with Facet Overlays** - Explore clusters visually with customizable overlays
2. **Minimum Threshold Privacy Protection** - Automatically protect user privacy through clustering thresholds
3. **Targeted Investigation Interface** - Search and filter clusters using advanced criteria
4. **Serendipitous Discovery** - Get personalized recommendations for exploration

## Features

### 1. Interactive 2D Map with Facet Overlays

**Location**: `src/visualization/interactive_map.rs`

The interactive map allows users to explore conversation clusters in a 2D space with customizable facet overlays.

#### Key Components:
- **MapPoint**: Represents cluster positions with metadata
- **FacetOverlay**: Configurable overlay system with multiple color schemes
- **InteractiveMap**: Main controller for map functionality

#### Color Schemes:
- **Sequential**: For continuous values (0 to max)
- **Diverging**: For values with meaningful midpoint
- **Categorical**: For discrete categories
- **Heatmap**: For intensity visualization

#### API Endpoints:
```
POST /api/clio/map/create          # Create new interactive map
POST /api/clio/map/overlay/:id     # Apply facet overlay
GET  /api/clio/map/export/:id      # Export map data
```

#### Usage Example:
```rust
use briefxai::visualization::interactive_map::{InteractiveMap, FacetOverlay, ColorScheme};

let map = InteractiveMap::new(clusters, umap_coords, &facet_data)?;
let overlay = FacetOverlay {
    facet_name: "sentiment".to_string(),
    color_scheme: ColorScheme::Heatmap,
    aggregation: AggregationType::Prevalence,
    threshold: Some(0.1),
};
map.apply_facet_overlay(overlay)?;
```

### 2. Minimum Threshold Privacy Protection

**Location**: `src/privacy/threshold_protection.rs`

Implements privacy protection through configurable minimum thresholds, following best practices for privacy-preserving analysis.

#### Key Components:
- **PrivacyConfig**: Configurable privacy settings
- **PrivacyFilter**: Applies privacy rules to clusters
- **PrivacyReport**: Reports on filtering actions taken

#### Privacy Features:
- **Minimum Cluster Size**: Clusters below threshold are merged/filtered
- **Facet Prevalence Thresholds**: Rare facet values are filtered out
- **Sensitive Facet Detection**: Higher thresholds for sensitive categories
- **Privacy Level Assessment**: Automatic assessment of protection level

#### API Endpoints:
```
POST /api/clio/privacy/filter      # Apply privacy filtering
GET  /api/clio/privacy/config      # Get privacy configuration
```

#### Configuration Options:
```yaml
privacy:
  min_cluster_size: 10
  min_unique_sources: 3
  merge_small_clusters: true
  min_facet_prevalence: 0.05
  sensitive_facets:
    - "personal_info"
    - "health_info" 
    - "financial_info"
  sensitive_facet_threshold: 50
```

#### Usage Example:
```rust
use briefxai::privacy::threshold_protection::{PrivacyFilter, PrivacyConfig};

let config = PrivacyConfig {
    min_cluster_size: 10,
    merge_small_clusters: true,
    ..Default::default()
};

let mut filter = PrivacyFilter::new(config);
let filtered_clusters = filter.filter_clusters(clusters, &conversations)?;
let report = filter.generate_report();
```

### 3. Targeted Investigation Interface

**Location**: `src/investigation/targeted_search.rs`

Provides powerful search and filtering capabilities for cluster analysis.

#### Key Components:
- **InvestigationEngine**: Main search engine
- **InvestigationQuery**: Query specification with multiple filter types
- **InvestigationResult**: Search results with scoring and explanations

#### Search Capabilities:
- **Text Search**: Search cluster names and descriptions
- **Facet Filters**: Filter by facet values with operators
- **Metric Filters**: Filter by cluster metrics (size, refusal rate, sentiment)
- **Similarity Search**: Find clusters similar to a given cluster
- **Combined Queries**: Multiple filter types in single query

#### Supported Metrics:
- **Size**: Number of conversations in cluster
- **Refusal Rate**: Percentage of assistant refusals
- **Sentiment Score**: Average sentiment (if available)
- **Conversation Length**: Average conversation length
- **Diversity**: Topic diversity within cluster
- **Coherence**: How well conversations fit cluster

#### API Endpoints:
```
POST /api/clio/investigate         # Run investigation query
GET  /api/clio/investigate/suggest # Get query suggestions
```

#### Usage Example:
```rust
use briefxai::investigation::targeted_search::*;

let engine = InvestigationEngine::new(clusters, conversations, facet_data, embeddings)?;

let query = InvestigationQuery {
    search_terms: vec!["support".to_string()],
    facet_filters: vec![FacetFilter {
        facet_name: "sentiment".to_string(),
        operator: FilterOperator::GreaterThan,
        value: "0.5".to_string(),
        threshold: None,
    }],
    metric_filters: vec![MetricFilter {
        metric: ClusterMetric::RefusalRate,
        operator: ComparisonOperator::LessThan,
        value: 0.1,
    }],
    sort_by: SortCriterion::Relevance,
    limit: Some(10),
    highlight_matches: true,
};

let results = engine.investigate(&query)?;
```

### 4. Serendipitous Discovery

**Location**: `src/discovery/serendipitous.rs`

Provides personalized recommendations for exploring cluster relationships and discovering unexpected connections.

#### Key Components:
- **DiscoveryEngine**: Main recommendation engine
- **ClusterGraph**: Graph representation of cluster relationships
- **ExplorationPath**: Suggested paths through clusters
- **DiscoveryPattern**: Detected interesting patterns

#### Discovery Types:
- **NextInPath**: Logical next step in exploration
- **Surprise**: Unexpected but interesting connection
- **DeepDive**: More detail on current topic
- **BranchOut**: Explore related but different topic
- **ReturnToInterest**: Go back to previously interesting area
- **BridgeGap**: Connect two explored areas

#### Pattern Detection:
- **Anomaly**: Isolated clusters with unique characteristics
- **Bridge**: Clusters connecting disparate topics
- **Convergence**: Multiple paths leading to same cluster
- **Divergence**: Single topic splitting into multiple clusters

#### API Endpoints:
```
POST /api/clio/discovery/recommendations    # Get recommendations
POST /api/clio/discovery/update_preferences # Update user preferences
GET  /api/clio/discovery/patterns          # Get discovery patterns
```

#### Usage Example:
```rust
use briefxai::discovery::serendipitous::*;

let mut engine = DiscoveryEngine::new(clusters, facet_data)?;

// Get recommendations from current position
let recommendations = engine.get_recommendations(Some(current_cluster), 5)?;

// Update preferences based on user exploration
engine.update_preferences(vec![0, 2, 5]);

// Get recommendations again (now personalized)
let new_recommendations = engine.get_recommendations(None, 5)?;
```

## Integration with Analysis Pipeline

**Location**: `src/analysis_integration.rs`

The Clio features are integrated into the main analysis pipeline through the `EnhancedAnalysisEngine`.

### Enhanced Analysis Flow:

1. **Standard Analysis**: Run normal conversation analysis
2. **Privacy Filtering**: Apply privacy thresholds
3. **Feature Creation**: Create interactive map, investigation engine, discovery engine
4. **Result Packaging**: Package everything into `EnhancedAnalysisResults`

### Usage in Main Pipeline:

```rust
use briefxai::analysis_integration::*;

let engine = EnhancedAnalysisEngine::new(config, clio_config);

let results = engine.analyze_with_clio_features(
    conversations,
    facet_data,
    embeddings,
    umap_coords,
    original_clusters,
).await?;

// Results now include all Clio features
assert!(results.interactive_map.is_some());
assert!(results.investigation_engine.is_some());
assert!(results.discovery_engine.is_some());
```

## Configuration

Add Clio features configuration to your `config.yaml`:

```yaml
# Clio Features
enable_clio_features: true

# Privacy Settings
clio_privacy_min_cluster_size: 10
clio_privacy_merge_small_clusters: true
clio_privacy_facet_threshold: 0.05

# Feature Toggles
enable_interactive_map: true
enable_investigation: true
enable_discovery: true

# Detailed Privacy Configuration
privacy:
  min_cluster_size: 10
  min_unique_sources: 3
  merge_small_clusters: true
  redact_small_cluster_descriptions: true
  min_facet_prevalence: 0.05
  sensitive_facets:
    - "personal_info"
    - "health_info"
    - "financial_info"
  sensitive_facet_threshold: 50
```

## Frontend Integration

### Interactive Map Usage:

```javascript
// Initialize Clio features
const clioFeatures = new ClioFeatures();

// Create interactive map
await clioFeatures.createMap();

// Apply facet overlay
await clioFeatures.applyFacetOverlay();

// Handle cluster selection
clioFeatures.selectCluster(clusterId);
```

### Investigation Usage:

```javascript
// Run investigation query
await clioFeatures.runInvestigation();

// Apply suggestion
clioFeatures.applySuggestion("Show high refusal clusters");
```

### Discovery Usage:

```javascript
// Load discovery recommendations
await clioFeatures.loadDiscoveryRecommendations();

// Follow recommendation
await clioFeatures.followRecommendation(clusterId);
```

## API Reference

### Data Types

#### MapPoint
```json
{
  "id": "cluster_0",
  "x": 0.1,
  "y": 0.2,
  "cluster_id": 0,
  "conversation_ids": [0, 1, 2],
  "facet_values": {"sentiment": {"value": "positive"}},
  "size": 15.0,
  "color": "#ff6b35"
}
```

#### InvestigationResult
```json
{
  "clusters": [...],
  "total_matches": 5,
  "query_time_ms": 150,
  "suggested_queries": ["Find similar clusters", "Show by sentiment"]
}
```

#### DiscoveryRecommendation
```json
{
  "cluster_id": 3,
  "cluster_name": "Product Features",
  "reason": "Related topic to current exploration",
  "discovery_type": "BranchOut",
  "confidence": 0.8,
  "preview": "Discussions about product capabilities...",
  "related_patterns": ["Bridge"]
}
```

#### PrivacyReport
```json
{
  "total_redacted": 2,
  "total_merged": 3,
  "min_cluster_size": 10,
  "min_unique_sources": 3,
  "config": {...}
}
```

## Testing

### Unit Tests
Each module includes comprehensive unit tests:

```bash
# Test specific modules
cargo test visualization::interactive_map::tests
cargo test privacy::threshold_protection::tests
cargo test investigation::targeted_search::tests
cargo test discovery::serendipitous::tests

# Test integration
cargo test analysis_integration::tests
```

### Integration Tests
Full feature integration test:

```bash
cargo test test_integrated_clio_features
```

### E2E Tests
Include Clio features in end-to-end testing:

```bash
./test_e2e.sh
```

## Performance Considerations

### Memory Usage
- Interactive maps cache visualization data
- Discovery engines maintain cluster graphs
- Investigation engines pre-calculate metrics

### Optimization Tips
1. **Dataset Size**: Features work best with 100+ conversations
2. **Privacy Thresholds**: Higher thresholds = better performance
3. **Feature Toggles**: Disable unused features to save memory
4. **Caching**: Map and engine instances are cached for reuse

### Scaling
- Interactive maps support up to 10,000 clusters efficiently
- Investigation can handle complex queries on large datasets
- Discovery recommendations scale with graph complexity

## Troubleshooting

### Common Issues

1. **Map Not Rendering**
   - Check UMAP coordinates are valid
   - Ensure D3.js is loaded
   - Verify cluster data format

2. **Privacy Filtering Too Aggressive**
   - Reduce `min_cluster_size`
   - Disable `merge_small_clusters`
   - Lower `min_facet_prevalence`

3. **Investigation No Results**
   - Check search terms and filters
   - Verify facet names match data
   - Enable debug logging

4. **Discovery No Recommendations**
   - Ensure minimum cluster count (3+)
   - Check cluster relationships exist
   - Verify facet data quality

### Debug Configuration

```yaml
debug:
  log_level: debug
  enable_clio_debug: true
  save_intermediate_results: true
```

## Future Enhancements

### Planned Features
1. **Advanced Visualizations**: 3D maps, timeline views
2. **ML-Powered Discovery**: Learning user preferences
3. **Real-time Collaboration**: Shared exploration sessions
4. **Custom Pattern Detection**: User-defined patterns
5. **Export Capabilities**: PDF reports, data export

### Research Directions
1. **Differential Privacy**: Mathematical privacy guarantees
2. **Federated Analysis**: Multi-organization insights
3. **Adaptive Thresholds**: Dynamic privacy protection
4. **Semantic Similarity**: Better cluster relationships

## Contributing

To contribute to Clio features:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Consider privacy implications
5. Test with real datasets

### Development Setup

```bash
# Install dependencies
cargo build

# Run tests
cargo test

# Start development server
cargo run -- serve --port 8080

# Access Clio features
open http://localhost:8080/clio.html
```

## License

The Clio features are part of BriefXAI and licensed under the MIT License, consistent with the main project.