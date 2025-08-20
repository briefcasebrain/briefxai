// BriefXAI Clio Features - Interactive Components
// Interactive 2D map, targeted investigation, and serendipitous discovery

class ClioFeatures {
    constructor() {
        this.currentMapId = null;
        this.currentDiscoveryEngine = null;
        this.selectedCluster = null;
        this.mapData = null;
        this.svg = null;
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Tab switching
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('clio-tab')) {
                this.switchTab(e.target.dataset.tab);
            }
        });

        // Interactive map controls
        document.addEventListener('change', (e) => {
            if (e.target.id === 'facet-selector') {
                this.applyFacetOverlay();
            }
            if (e.target.id === 'color-scheme-selector') {
                this.applyFacetOverlay();
            }
        });

        // Investigation controls
        document.addEventListener('click', (e) => {
            if (e.target.id === 'run-investigation') {
                this.runInvestigation();
            }
            if (e.target.classList.contains('suggestion-pill')) {
                this.applySuggestion(e.target.textContent);
            }
        });

        // Discovery controls
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('discovery-recommendation')) {
                this.followRecommendation(e.target.dataset.clusterId);
            }
        });
    }

    switchTab(tabName) {
        // Hide all tab contents
        document.querySelectorAll('.clio-tab-content').forEach(content => {
            content.style.display = 'none';
        });

        // Remove active class from all tabs
        document.querySelectorAll('.clio-tab').forEach(tab => {
            tab.classList.remove('active');
        });

        // Show selected tab content
        const targetContent = document.getElementById(`${tabName}-content`);
        if (targetContent) {
            targetContent.style.display = 'block';
        }

        // Add active class to selected tab
        const targetTab = document.querySelector(`[data-tab="${tabName}"]`);
        if (targetTab) {
            targetTab.classList.add('active');
        }

        // Initialize tab-specific content
        this.initializeTab(tabName);
    }

    async initializeTab(tabName) {
        switch (tabName) {
            case 'interactive-map':
                await this.initializeInteractiveMap();
                break;
            case 'investigation':
                await this.initializeInvestigation();
                break;
            case 'discovery':
                await this.initializeDiscovery();
                break;
            case 'privacy':
                await this.initializePrivacy();
                break;
        }
    }

    // Interactive Map Functions
    async initializeInteractiveMap() {
        console.log('Initializing interactive map...');
        
        if (!this.currentMapId) {
            await this.createMap();
        }
        
        if (this.currentMapId) {
            await this.loadMapData();
            this.renderMap();
            await this.loadFacetOptions();
        }
    }

    async createMap() {
        try {
            // Get analysis data from the current session
            const analysisData = await this.getCurrentAnalysisData();
            
            const response = await fetch('/api/clio/map/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(analysisData),
            });

            const result = await response.json();
            if (result.success) {
                this.currentMapId = result.data;
                console.log('Map created with ID:', this.currentMapId);
            } else {
                console.error('Failed to create map:', result.error);
                this.showError('Failed to create interactive map: ' + result.error);
            }
        } catch (error) {
            console.error('Error creating map:', error);
            this.showError('Error creating map: ' + error.message);
        }
    }

    async getCurrentAnalysisData() {
        // This would integrate with the existing analysis system
        // For now, return mock data
        return {
            clusters: [
                {
                    conversation_ids: [0, 1, 2, 3, 4],
                    name: "Customer Support",
                    description: "Support conversations about product issues",
                    children: []
                },
                {
                    conversation_ids: [5, 6, 7, 8],
                    name: "Product Features",
                    description: "Discussions about product capabilities",
                    children: []
                },
                {
                    conversation_ids: [9, 10, 11],
                    name: "Sales Inquiries",
                    description: "Questions about pricing and plans",
                    children: []
                }
            ],
            umap_coords: [
                [0.1, 0.2], [0.15, 0.25], [0.12, 0.22], [0.18, 0.28], [0.14, 0.24],
                [0.6, 0.7], [0.65, 0.75], [0.62, 0.72], [0.68, 0.78],
                [0.3, 0.9], [0.35, 0.95], [0.32, 0.92]
            ],
            facet_data: [
                [{ facet: { name: "sentiment", question: "Sentiment?" }, value: "positive" }],
                [{ facet: { name: "sentiment", question: "Sentiment?" }, value: "neutral" }],
                [{ facet: { name: "sentiment", question: "Sentiment?" }, value: "negative" }],
                [{ facet: { name: "category", question: "Category?" }, value: "technical" }],
                [{ facet: { name: "category", question: "Category?" }, value: "billing" }],
                [{ facet: { name: "sentiment", question: "Sentiment?" }, value: "positive" }],
                [{ facet: { name: "category", question: "Category?" }, value: "feature" }],
                [{ facet: { name: "sentiment", question: "Sentiment?" }, value: "positive" }],
                [{ facet: { name: "category", question: "Category?" }, value: "feature" }],
                [{ facet: { name: "sentiment", question: "Sentiment?" }, value: "neutral" }],
                [{ facet: { name: "category", question: "Category?" }, value: "pricing" }],
                [{ facet: { name: "category", question: "Category?" }, value: "sales" }]
            ]
        };
    }

    async loadMapData() {
        try {
            const response = await fetch(`/api/clio/map/export/${this.currentMapId}`);
            const result = await response.json();
            
            if (result.success) {
                this.mapData = result.data;
                console.log('Map data loaded:', this.mapData);
            } else {
                console.error('Failed to load map data:', result.error);
            }
        } catch (error) {
            console.error('Error loading map data:', error);
        }
    }

    renderMap() {
        if (!this.mapData || !this.mapData.points) {
            console.warn('No map data available for rendering');
            return;
        }

        const container = document.getElementById('interactive-map-container');
        if (!container) {
            console.error('Map container not found');
            return;
        }

        // Clear existing content
        container.innerHTML = '';

        // Set up SVG
        const width = container.clientWidth || 800;
        const height = container.clientHeight || 600;
        
        this.svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        // Create scales
        const xExtent = d3.extent(this.mapData.points, d => d.x);
        const yExtent = d3.extent(this.mapData.points, d => d.y);
        
        const xScale = d3.scaleLinear()
            .domain(xExtent)
            .range([50, width - 50]);
            
        const yScale = d3.scaleLinear()
            .domain(yExtent)
            .range([height - 50, 50]);

        // Create tooltip
        const tooltip = d3.select('body').append('div')
            .attr('class', 'map-tooltip')
            .style('opacity', 0);

        // Render points
        this.svg.selectAll('circle')
            .data(this.mapData.points)
            .enter()
            .append('circle')
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', d => Math.max(5, d.size / 2))
            .attr('fill', d => d.color)
            .attr('stroke', '#333')
            .attr('stroke-width', 1)
            .style('cursor', 'pointer')
            .on('mouseover', (event, d) => {
                tooltip.transition()
                    .duration(200)
                    .style('opacity', .9);
                tooltip.html(`
                    <strong>${this.mapData.cluster_info[d.cluster_id]?.name || 'Unknown'}</strong><br/>
                    Conversations: ${d.conversation_ids.length}<br/>
                    Cluster ID: ${d.cluster_id}
                `)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 28) + 'px');
            })
            .on('mouseout', () => {
                tooltip.transition()
                    .duration(500)
                    .style('opacity', 0);
            })
            .on('click', (event, d) => {
                this.selectCluster(d.cluster_id);
            });

        // Add axes
        const xAxis = d3.axisBottom(xScale);
        const yAxis = d3.axisLeft(yScale);

        this.svg.append('g')
            .attr('transform', `translate(0, ${height - 50})`)
            .call(xAxis);

        this.svg.append('g')
            .attr('transform', 'translate(50, 0)')
            .call(yAxis);

        console.log('Map rendered with', this.mapData.points.length, 'points');
    }

    async loadFacetOptions() {
        if (!this.mapData || !this.mapData.facet_options) {
            return;
        }

        const selector = document.getElementById('facet-selector');
        if (!selector) return;

        selector.innerHTML = '<option value="">Select a facet...</option>';
        
        this.mapData.facet_options.forEach(facet => {
            const option = document.createElement('option');
            option.value = facet;
            option.textContent = facet.charAt(0).toUpperCase() + facet.slice(1);
            selector.appendChild(option);
        });
    }

    async applyFacetOverlay() {
        const facetName = document.getElementById('facet-selector').value;
        const colorScheme = document.getElementById('color-scheme-selector').value || 'heatmap';
        
        if (!facetName || !this.currentMapId) return;

        try {
            const response = await fetch(`/api/clio/map/overlay/${this.currentMapId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    facet_name: facetName,
                    color_scheme: colorScheme,
                    aggregation: 'prevalence',
                    threshold: 0.1
                }),
            });

            const result = await response.json();
            if (result.success) {
                await this.loadMapData();
                this.renderMap();
                console.log('Facet overlay applied:', facetName);
            } else {
                console.error('Failed to apply overlay:', result.error);
            }
        } catch (error) {
            console.error('Error applying overlay:', error);
        }
    }

    selectCluster(clusterId) {
        this.selectedCluster = clusterId;
        console.log('Selected cluster:', clusterId);
        
        // Update UI to show selection
        if (this.svg) {
            this.svg.selectAll('circle')
                .attr('stroke-width', d => d.cluster_id === clusterId ? 3 : 1)
                .attr('stroke', d => d.cluster_id === clusterId ? '#ff6b35' : '#333');
        }

        // Update other tabs with selected cluster
        this.updateInvestigationWithCluster(clusterId);
        this.updateDiscoveryWithCluster(clusterId);
    }

    // Investigation Functions
    async initializeInvestigation() {
        console.log('Initializing investigation interface...');
        await this.loadInvestigationSuggestions();
    }

    async loadInvestigationSuggestions() {
        try {
            const response = await fetch('/api/clio/investigate/suggest');
            const result = await response.json();
            
            if (result.success) {
                this.renderInvestigationSuggestions(result.data);
            }
        } catch (error) {
            console.error('Error loading suggestions:', error);
        }
    }

    renderInvestigationSuggestions(suggestions) {
        const container = document.getElementById('investigation-suggestions');
        if (!container) return;

        container.innerHTML = '';
        
        suggestions.forEach(suggestion => {
            const pill = document.createElement('span');
            pill.className = 'suggestion-pill';
            pill.textContent = suggestion;
            pill.style.cursor = 'pointer';
            container.appendChild(pill);
        });
    }

    async runInvestigation() {
        const searchTerms = document.getElementById('search-terms').value
            .split(',')
            .map(term => term.trim())
            .filter(term => term.length > 0);

        const query = {
            search_terms: searchTerms,
            facet_filters: [],
            metric_filters: [],
            similar_to_cluster: this.selectedCluster,
            sort_by: 'relevance',
            limit: 10,
            highlight_matches: true
        };

        const analysisData = await this.getCurrentAnalysisData();
        
        const requestData = {
            clusters: analysisData.clusters,
            conversations: [],
            facet_data: analysisData.facet_data,
            embeddings: null,
            query: query
        };

        try {
            const response = await fetch('/api/clio/investigate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData),
            });

            const result = await response.json();
            
            if (result.success) {
                this.renderInvestigationResults(result.data);
            } else {
                console.error('Investigation failed:', result.error);
                this.showError('Investigation failed: ' + result.error);
            }
        } catch (error) {
            console.error('Error running investigation:', error);
            this.showError('Investigation error: ' + error.message);
        }
    }

    renderInvestigationResults(results) {
        const container = document.getElementById('investigation-results');
        if (!container) return;

        container.innerHTML = `
            <h4>Investigation Results (${results.total_matches} matches, ${results.query_time_ms}ms)</h4>
        `;

        results.clusters.forEach(match => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'investigation-result';
            resultDiv.innerHTML = `
                <div class="result-header">
                    <h5>${match.cluster.name}</h5>
                    <span class="match-score">Score: ${match.match_score.toFixed(2)}</span>
                </div>
                <p class="result-description">${match.highlighted_text || match.cluster.description}</p>
                <div class="result-metrics">
                    <span>Size: ${match.metrics.size}</span>
                    <span>Refusal Rate: ${(match.metrics.refusal_rate * 100).toFixed(1)}%</span>
                    <span>Sentiment: ${match.metrics.sentiment_score.toFixed(2)}</span>
                </div>
                <p class="result-explanation">${match.explanation}</p>
            `;
            container.appendChild(resultDiv);
        });

        // Show suggested follow-up queries
        if (results.suggested_queries && results.suggested_queries.length > 0) {
            const suggestionsDiv = document.createElement('div');
            suggestionsDiv.className = 'follow-up-suggestions';
            suggestionsDiv.innerHTML = '<h5>Suggested follow-up queries:</h5>';
            
            results.suggested_queries.forEach(suggestion => {
                const suggestionSpan = document.createElement('span');
                suggestionSpan.className = 'suggestion-pill';
                suggestionSpan.textContent = suggestion;
                suggestionsDiv.appendChild(suggestionSpan);
            });
            
            container.appendChild(suggestionsDiv);
        }
    }

    updateInvestigationWithCluster(clusterId) {
        const similarClusterCheck = document.getElementById('similar-cluster-check');
        if (similarClusterCheck) {
            similarClusterCheck.checked = true;
        }
    }

    applySuggestion(suggestionText) {
        document.getElementById('search-terms').value = suggestionText;
        this.runInvestigation();
    }

    // Discovery Functions
    async initializeDiscovery() {
        console.log('Initializing discovery interface...');
        await this.loadDiscoveryRecommendations();
    }

    async loadDiscoveryRecommendations() {
        const analysisData = await this.getCurrentAnalysisData();
        
        const requestData = {
            clusters: analysisData.clusters,
            facet_data: analysisData.facet_data,
            current_cluster: this.selectedCluster,
            limit: 5
        };

        try {
            const response = await fetch('/api/clio/discovery/recommendations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData),
            });

            const result = await response.json();
            
            if (result.success) {
                this.renderDiscoveryRecommendations(result.data);
            } else {
                console.error('Failed to load recommendations:', result.error);
            }
        } catch (error) {
            console.error('Error loading recommendations:', error);
        }
    }

    renderDiscoveryRecommendations(recommendations) {
        const container = document.getElementById('discovery-recommendations');
        if (!container) return;

        container.innerHTML = '<h4>Discovery Recommendations</h4>';

        if (recommendations.length === 0) {
            container.innerHTML += '<p>No recommendations available at this time.</p>';
            return;
        }

        recommendations.forEach(rec => {
            const recDiv = document.createElement('div');
            recDiv.className = 'discovery-recommendation';
            recDiv.dataset.clusterId = rec.cluster_id;
            
            const typeIcon = this.getDiscoveryTypeIcon(rec.discovery_type);
            
            recDiv.innerHTML = `
                <div class="recommendation-header">
                    <span class="type-icon">${typeIcon}</span>
                    <h5>${rec.cluster_name}</h5>
                    <span class="confidence">Confidence: ${(rec.confidence * 100).toFixed(0)}%</span>
                </div>
                <p class="recommendation-reason">${rec.reason}</p>
                <p class="recommendation-preview">${rec.preview}</p>
            `;
            
            container.appendChild(recDiv);
        });
    }

    getDiscoveryTypeIcon(discoveryType) {
        const icons = {
            NextInPath: '➡️',
            Surprise: '🎲',
            DeepDive: '🔍',
            BranchOut: '🌿',
            ReturnToInterest: '🔄',
            BridgeGap: '🌉'
        };
        return icons[discoveryType] || '🎯';
    }

    async followRecommendation(clusterId) {
        this.selectCluster(parseInt(clusterId));
        
        // Update preferences based on this choice
        if (this.currentDiscoveryEngine) {
            try {
                await fetch(`/api/clio/discovery/update_preferences/${this.currentDiscoveryEngine}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify([parseInt(clusterId)]),
                });
            } catch (error) {
                console.error('Error updating preferences:', error);
            }
        }
        
        // Reload recommendations
        await this.loadDiscoveryRecommendations();
    }

    updateDiscoveryWithCluster(clusterId) {
        // Reload recommendations with new current cluster
        this.loadDiscoveryRecommendations();
    }

    // Privacy Functions
    async initializePrivacy() {
        console.log('Initializing privacy interface...');
        await this.loadPrivacyConfig();
    }

    async loadPrivacyConfig() {
        try {
            const response = await fetch('/api/clio/privacy/config');
            const result = await response.json();
            
            if (result.success) {
                this.populatePrivacyForm(result.data);
            }
        } catch (error) {
            console.error('Error loading privacy config:', error);
        }
    }

    populatePrivacyForm(config) {
        const form = document.getElementById('privacy-config-form');
        if (!form) return;

        // Populate form fields with config values
        const minClusterSize = document.getElementById('min-cluster-size');
        if (minClusterSize) minClusterSize.value = config.min_cluster_size;

        const mergeSmall = document.getElementById('merge-small-clusters');
        if (mergeSmall) mergeSmall.checked = config.merge_small_clusters;

        const facetThreshold = document.getElementById('facet-threshold');
        if (facetThreshold) facetThreshold.value = config.min_facet_prevalence;
    }

    // Utility Functions
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ff4444;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            z-index: 1000;
        `;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }

    showSuccess(message) {
        const successDiv = document.createElement('div');
        successDiv.className = 'success-message';
        successDiv.textContent = message;
        successDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #44ff44;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            z-index: 1000;
        `;
        
        document.body.appendChild(successDiv);
        
        setTimeout(() => {
            if (successDiv.parentNode) {
                successDiv.parentNode.removeChild(successDiv);
            }
        }, 3000);
    }
}

// Initialize Clio features when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    window.clioFeatures = new ClioFeatures();
    console.log('Clio features initialized');
});