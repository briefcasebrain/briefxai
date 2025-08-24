// Clio Visualization System
// Implements the zoomable map-like interface with interactive exploration

class ClioVisualization {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        
        // Visualization state
        this.currentLevel = 0;
        this.selectedCluster = null;
        this.zoomLevel = 1;
        this.panX = 0;
        this.panY = 0;
        
        // Data
        this.hierarchy = null;
        this.clusters = [];
        this.conversations = [];
        
        // D3 setup
        this.svg = null;
        this.g = null;
        this.zoom = null;
        
        // Interaction state
        this.isDragging = false;
        this.dragStartX = 0;
        this.dragStartY = 0;
        
        this.init();
    }
    
    init() {
        // Create SVG canvas
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height)
            .attr('class', 'clio-canvas');
        
        // Create main group for transformations
        this.g = this.svg.append('g')
            .attr('class', 'clio-main-group');
        
        // Setup zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => this.handleZoom(event));
        
        this.svg.call(this.zoom);
        
        // Setup interaction handlers
        this.setupInteractions();
        
        // Create UI layers
        this.createLayers();
        
        // Load initial data
        this.loadData();
    }
    
    createLayers() {
        // Background layer for grid/map
        this.backgroundLayer = this.g.append('g')
            .attr('class', 'background-layer');
        
        // Cluster layer
        this.clusterLayer = this.g.append('g')
            .attr('class', 'cluster-layer');
        
        // Connection layer for relationships
        this.connectionLayer = this.g.append('g')
            .attr('class', 'connection-layer');
        
        // Label layer
        this.labelLayer = this.g.append('g')
            .attr('class', 'label-layer');
        
        // Overlay layer for tooltips and interactions
        this.overlayLayer = this.g.append('g')
            .attr('class', 'overlay-layer');
        
        // Create background grid
        this.createGrid();
    }
    
    createGrid() {
        const gridSize = 50;
        const numLinesX = Math.ceil(this.width / gridSize);
        const numLinesY = Math.ceil(this.height / gridSize);
        
        // Vertical lines
        for (let i = 0; i <= numLinesX; i++) {
            this.backgroundLayer.append('line')
                .attr('x1', i * gridSize)
                .attr('y1', 0)
                .attr('x2', i * gridSize)
                .attr('y2', this.height)
                .attr('class', 'grid-line')
                .style('stroke', '#e0e0e0')
                .style('stroke-width', 0.5);
        }
        
        // Horizontal lines
        for (let i = 0; i <= numLinesY; i++) {
            this.backgroundLayer.append('line')
                .attr('x1', 0)
                .attr('y1', i * gridSize)
                .attr('x2', this.width)
                .attr('y2', i * gridSize)
                .attr('class', 'grid-line')
                .style('stroke', '#e0e0e0')
                .style('stroke-width', 0.5);
        }
    }
    
    async loadData() {
        try {
            const response = await fetch('/api/clio/hierarchy');
            const data = await response.json();
            
            this.hierarchy = data.hierarchy;
            this.clusters = data.clusters;
            this.conversations = data.conversations;
            
            this.renderHierarchy();
        } catch (error) {
            console.error('Failed to load Clio data:', error);
            // Use sample data for now
            this.loadSampleData();
        }
    }
    
    loadSampleData() {
        // Create sample hierarchical data
        this.hierarchy = {
            id: 'root',
            name: 'All Conversations',
            level: 0,
            size: 1234,
            children: [
                {
                    id: 'cluster-1',
                    name: 'Technical Issues',
                    level: 1,
                    size: 456,
                    sentiment: -0.6,
                    position: { x: 200, y: 200 },
                    children: [
                        {
                            id: 'cluster-1-1',
                            name: 'Login Problems',
                            level: 2,
                            size: 234,
                            sentiment: -0.8,
                            position: { x: 150, y: 150 }
                        },
                        {
                            id: 'cluster-1-2',
                            name: 'Performance',
                            level: 2,
                            size: 222,
                            sentiment: -0.4,
                            position: { x: 250, y: 150 }
                        }
                    ]
                },
                {
                    id: 'cluster-2',
                    name: 'Feature Requests',
                    level: 1,
                    size: 334,
                    sentiment: 0.3,
                    position: { x: 500, y: 300 },
                    children: [
                        {
                            id: 'cluster-2-1',
                            name: 'UI Improvements',
                            level: 2,
                            size: 167,
                            sentiment: 0.5,
                            position: { x: 450, y: 250 }
                        },
                        {
                            id: 'cluster-2-2',
                            name: 'New Features',
                            level: 2,
                            size: 167,
                            sentiment: 0.1,
                            position: { x: 550, y: 250 }
                        }
                    ]
                },
                {
                    id: 'cluster-3',
                    name: 'Billing',
                    level: 1,
                    size: 444,
                    sentiment: -0.3,
                    position: { x: 350, y: 450 },
                    children: []
                }
            ]
        };
        
        this.renderHierarchy();
    }
    
    renderHierarchy() {
        // Clear existing clusters
        this.clusterLayer.selectAll('.cluster-group').remove();
        this.connectionLayer.selectAll('.cluster-connection').remove();
        
        // Get clusters at current level
        const clustersToRender = this.getClustersByLevel(this.currentLevel);
        
        // Render connections first (so they appear behind clusters)
        this.renderConnections(clustersToRender);
        
        // Render clusters
        const clusterGroups = this.clusterLayer.selectAll('.cluster-group')
            .data(clustersToRender)
            .enter()
            .append('g')
            .attr('class', 'cluster-group')
            .attr('transform', d => `translate(${d.position.x}, ${d.position.y})`);
        
        // Add cluster circles
        clusterGroups.append('circle')
            .attr('class', 'cluster-circle')
            .attr('r', d => this.getClusterRadius(d))
            .style('fill', d => this.getClusterColor(d))
            .style('stroke', '#fff')
            .style('stroke-width', 2)
            .style('cursor', 'pointer')
            .on('click', (event, d) => this.handleClusterClick(d))
            .on('mouseover', (event, d) => this.showClusterTooltip(event, d))
            .on('mouseout', () => this.hideTooltip());
        
        // Add cluster labels
        clusterGroups.append('text')
            .attr('class', 'cluster-label')
            .attr('text-anchor', 'middle')
            .attr('dy', d => this.getClusterRadius(d) + 20)
            .style('font-size', '12px')
            .style('font-weight', 'bold')
            .text(d => d.name);
        
        // Add size indicators
        clusterGroups.append('text')
            .attr('class', 'cluster-size')
            .attr('text-anchor', 'middle')
            .attr('dy', 5)
            .style('font-size', '14px')
            .style('fill', '#fff')
            .text(d => d.size);
        
        // Add sentiment indicators
        clusterGroups.each((d, i, nodes) => {
            const group = d3.select(nodes[i]);
            if (d.sentiment !== undefined) {
                const radius = this.getClusterRadius(d);
                const angle = Math.PI / 4; // 45 degrees
                const iconX = radius * Math.cos(angle);
                const iconY = -radius * Math.sin(angle);
                
                group.append('text')
                    .attr('class', 'sentiment-icon')
                    .attr('x', iconX)
                    .attr('y', iconY)
                    .attr('text-anchor', 'middle')
                    .style('font-size', '16px')
                    .text(d.sentiment > 0 ? '😊' : d.sentiment < -0.5 ? '😠' : '😐');
            }
        });
        
        // Update breadcrumb
        this.updateBreadcrumb();
    }
    
    renderConnections(clusters) {
        // Create connections between parent and child clusters
        const connections = [];
        
        clusters.forEach(cluster => {
            if (cluster.children && cluster.children.length > 0) {
                cluster.children.forEach(child => {
                    if (child.position) {
                        connections.push({
                            source: cluster.position,
                            target: child.position
                        });
                    }
                });
            }
        });
        
        // Draw connection lines
        this.connectionLayer.selectAll('.cluster-connection')
            .data(connections)
            .enter()
            .append('line')
            .attr('class', 'cluster-connection')
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y)
            .style('stroke', '#ccc')
            .style('stroke-width', 1)
            .style('stroke-dasharray', '3,3');
    }
    
    getClustersByLevel(level) {
        const clusters = [];
        
        const traverse = (node, currentLevel) => {
            if (currentLevel === level) {
                clusters.push(node);
            } else if (node.children && currentLevel < level) {
                node.children.forEach(child => traverse(child, currentLevel + 1));
            }
        };
        
        if (level === 0) {
            // Show top-level clusters
            return this.hierarchy.children || [];
        } else {
            // Show clusters at specific level
            traverse(this.hierarchy, 0);
            return clusters;
        }
    }
    
    getClusterRadius(cluster) {
        // Calculate radius based on size and level
        const baseRadius = 30;
        const sizeScale = Math.log(cluster.size + 1) * 5;
        const levelScale = 1 - (cluster.level * 0.1);
        
        return Math.max(20, Math.min(80, baseRadius + sizeScale * levelScale));
    }
    
    getClusterColor(cluster) {
        if (!cluster.sentiment) return '#6366f1';
        
        // Color based on sentiment
        const colorScale = d3.scaleLinear()
            .domain([-1, 0, 1])
            .range(['#ef4444', '#f59e0b', '#10b981']);
        
        return colorScale(cluster.sentiment);
    }
    
    handleClusterClick(cluster) {
        console.log('Clicked cluster:', cluster);
        
        if (cluster.children && cluster.children.length > 0) {
            // Drill down into cluster
            this.drillDown(cluster);
        } else {
            // Show cluster details
            this.showClusterDetails(cluster);
        }
    }
    
    drillDown(cluster) {
        // Animate zoom to cluster
        const radius = this.getClusterRadius(cluster);
        const scale = Math.min(this.width, this.height) / (radius * 4);
        
        this.svg.transition()
            .duration(750)
            .call(
                this.zoom.transform,
                d3.zoomIdentity
                    .translate(this.width / 2, this.height / 2)
                    .scale(scale)
                    .translate(-cluster.position.x, -cluster.position.y)
            );
        
        // Update current level
        setTimeout(() => {
            this.currentLevel = cluster.level + 1;
            this.selectedCluster = cluster;
            this.renderHierarchy();
        }, 750);
    }
    
    drillUp() {
        if (this.currentLevel > 0) {
            this.currentLevel--;
            this.selectedCluster = null;
            
            // Reset zoom
            this.svg.transition()
                .duration(750)
                .call(this.zoom.transform, d3.zoomIdentity);
            
            this.renderHierarchy();
        }
    }
    
    showClusterDetails(cluster) {
        // Create detail panel
        const detailPanel = document.createElement('div');
        detailPanel.className = 'cluster-detail-panel';
        detailPanel.innerHTML = `
            <div class="detail-header">
                <h3>${cluster.name}</h3>
                <button class="close-btn" onclick="clioViz.hideDetails()">×</button>
            </div>
            <div class="detail-content">
                <div class="detail-stat">
                    <span class="stat-label">Conversations:</span>
                    <span class="stat-value">${cluster.size}</span>
                </div>
                <div class="detail-stat">
                    <span class="stat-label">Sentiment:</span>
                    <span class="stat-value">${(cluster.sentiment * 100).toFixed(0)}%</span>
                </div>
                <div class="detail-stat">
                    <span class="stat-label">Level:</span>
                    <span class="stat-value">${cluster.level}</span>
                </div>
            </div>
            <div class="detail-actions">
                <button onclick="clioViz.investigateCluster('${cluster.id}')">Investigate</button>
                <button onclick="clioViz.exportCluster('${cluster.id}')">Export</button>
            </div>
        `;
        
        this.container.appendChild(detailPanel);
    }
    
    hideDetails() {
        const panel = this.container.querySelector('.cluster-detail-panel');
        if (panel) {
            panel.remove();
        }
    }
    
    showClusterTooltip(event, cluster) {
        // Create tooltip
        const tooltip = d3.select('body').append('div')
            .attr('class', 'clio-tooltip')
            .style('position', 'absolute')
            .style('background', 'rgba(0, 0, 0, 0.8)')
            .style('color', 'white')
            .style('padding', '10px')
            .style('border-radius', '5px')
            .style('pointer-events', 'none')
            .style('z-index', 1000);
        
        tooltip.html(`
            <strong>${cluster.name}</strong><br>
            Size: ${cluster.size}<br>
            ${cluster.sentiment ? `Sentiment: ${(cluster.sentiment * 100).toFixed(0)}%` : ''}
        `);
        
        // Position tooltip
        tooltip
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px');
    }
    
    hideTooltip() {
        d3.selectAll('.clio-tooltip').remove();
    }
    
    updateBreadcrumb() {
        // Create breadcrumb navigation
        let breadcrumb = this.container.querySelector('.breadcrumb');
        if (!breadcrumb) {
            breadcrumb = document.createElement('div');
            breadcrumb.className = 'breadcrumb';
            this.container.insertBefore(breadcrumb, this.container.firstChild);
        }
        
        const path = [];
        let current = this.selectedCluster;
        
        path.push({ name: 'All', level: 0 });
        
        if (current) {
            path.push({ name: current.name, level: current.level });
        }
        
        breadcrumb.innerHTML = path.map((item, i) => `
            <span class="breadcrumb-item ${i === path.length - 1 ? 'active' : ''}" 
                  onclick="clioViz.navigateToLevel(${item.level})">
                ${item.name}
            </span>
        `).join(' > ');
    }
    
    navigateToLevel(level) {
        this.currentLevel = level;
        this.renderHierarchy();
        
        // Reset zoom
        this.svg.transition()
            .duration(500)
            .call(this.zoom.transform, d3.zoomIdentity);
    }
    
    handleZoom(event) {
        this.g.attr('transform', event.transform);
        this.zoomLevel = event.transform.k;
        
        // Adjust label visibility based on zoom level
        this.adjustLabelVisibility();
    }
    
    adjustLabelVisibility() {
        // Hide labels when zoomed out too far
        const showLabels = this.zoomLevel > 0.5;
        
        this.labelLayer.selectAll('.cluster-label')
            .style('display', showLabels ? 'block' : 'none');
        
        // Show more detail when zoomed in
        if (this.zoomLevel > 2) {
            this.showDetailedLabels();
        }
    }
    
    showDetailedLabels() {
        // Add additional information when zoomed in
        this.clusterLayer.selectAll('.cluster-group').each((d, i, nodes) => {
            const group = d3.select(nodes[i]);
            
            // Check if detailed labels already exist
            if (!group.select('.detailed-info').node()) {
                group.append('text')
                    .attr('class', 'detailed-info')
                    .attr('text-anchor', 'middle')
                    .attr('dy', -this.getClusterRadius(d) - 10)
                    .style('font-size', '10px')
                    .style('fill', '#666')
                    .text(`${d.children ? d.children.length + ' subclusters' : 'Leaf cluster'}`);
            }
        });
    }
    
    setupInteractions() {
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'Escape':
                    this.drillUp();
                    break;
                case '+':
                case '=':
                    this.zoomIn();
                    break;
                case '-':
                case '_':
                    this.zoomOut();
                    break;
                case 'r':
                    this.resetView();
                    break;
            }
        });
        
        // Double-click to drill up
        this.svg.on('dblclick', () => {
            this.drillUp();
        });
    }
    
    zoomIn() {
        this.svg.transition()
            .duration(300)
            .call(this.zoom.scaleBy, 1.3);
    }
    
    zoomOut() {
        this.svg.transition()
            .duration(300)
            .call(this.zoom.scaleBy, 0.7);
    }
    
    resetView() {
        this.currentLevel = 0;
        this.selectedCluster = null;
        
        this.svg.transition()
            .duration(750)
            .call(this.zoom.transform, d3.zoomIdentity);
        
        this.renderHierarchy();
    }
    
    // API methods for external control
    investigateCluster(clusterId) {
        console.log('Investigating cluster:', clusterId);
        // Trigger investigation mode
        window.postMessage({
            type: 'investigate_cluster',
            clusterId: clusterId
        }, '*');
    }
    
    exportCluster(clusterId) {
        console.log('Exporting cluster:', clusterId);
        // Trigger export
        fetch(`/api/clio/export/${clusterId}`, {
            method: 'POST'
        }).then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `cluster_${clusterId}.json`;
            a.click();
        });
    }
    
    // Update visualization with new data
    updateData(data) {
        this.hierarchy = data.hierarchy;
        this.clusters = data.clusters;
        this.conversations = data.conversations;
        
        this.renderHierarchy();
    }
    
    // Handle window resize
    handleResize() {
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        
        this.svg
            .attr('width', this.width)
            .attr('height', this.height);
        
        this.createGrid();
        this.renderHierarchy();
    }
}

// Initialize when DOM is ready
let clioViz;
document.addEventListener('DOMContentLoaded', () => {
    // Check if we're on a page with Clio visualization
    const vizContainer = document.getElementById('clio-visualization');
    if (vizContainer) {
        clioViz = new ClioVisualization('clio-visualization');
        
        // Handle window resize
        window.addEventListener('resize', () => clioViz.handleResize());
    }
    
    // Add navbar functionality
    setupNavbarInteractions();
});

// Navbar functionality
function setupNavbarInteractions() {
    // Navigation tab switching
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            const view = e.target.getAttribute('data-view');
            switchToView(view);
        });
    });
}

function switchToView(view) {
    console.log('Switching to view:', view);
    
    // Update active tab
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelector(`.nav-tab[data-view="${view}"]`).classList.add('active');
    
    // Show different content based on view
    updateMainContent(view);
}

function updateMainContent(view) {
    const mainViz = document.querySelector('.main-viz');
    const breadcrumb = document.querySelector('.breadcrumb');
    
    // Update breadcrumb
    switch(view) {
        case 'map':
            breadcrumb.innerHTML = '<span class="breadcrumb-item active">Map View</span>';
            showMapView();
            break;
        case 'patterns':
            breadcrumb.innerHTML = '<span class="breadcrumb-item">Map View</span> > <span class="breadcrumb-item active">Pattern Analysis</span>';
            showPatternsView();
            break;
        case 'timeline':
            breadcrumb.innerHTML = '<span class="breadcrumb-item">Map View</span> > <span class="breadcrumb-item active">Timeline View</span>';
            showTimelineView();
            break;
        case 'audit':
            breadcrumb.innerHTML = '<span class="breadcrumb-item">Map View</span> > <span class="breadcrumb-item active">Privacy Audit</span>';
            showAuditView();
            break;
    }
}

function showMapView() {
    const vizCanvas = document.getElementById('clio-visualization');
    vizCanvas.innerHTML = '';
    vizCanvas.style.display = 'block';
    
    // Initialize or refresh the visualization
    if (window.clioViz) {
        window.clioViz.resetView();
    } else {
        window.clioViz = new ClioVisualization('clio-visualization');
    }
}

function showPatternsView() {
    const vizCanvas = document.getElementById('clio-visualization');
    vizCanvas.style.display = 'flex';
    vizCanvas.innerHTML = `
        <div style="width: 100%; padding: 2rem; background: white; border-radius: 8px;">
            <h2>🔍 Pattern Discovery</h2>
            <p style="margin-bottom: 2rem; color: #666;">Discover recurring patterns in your conversation data using advanced AI analysis.</p>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem;">
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #28a745;">
                    <h3 style="color: #28a745; margin: 0 0 1rem 0;">🔄 Recurring Patterns</h3>
                    <p style="margin: 0 0 1rem 0;">Common themes that appear across multiple conversations</p>
                    <div style="background: white; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">"Password reset issues"</div>
                        <div style="font-size: 0.9em; color: #666;">Frequency: 234 conversations (18.9%)</div>
                    </div>
                </div>
                
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #ffc107;">
                    <h3 style="color: #ffc107; margin: 0 0 1rem 0;">📈 Emerging Patterns</h3>
                    <p style="margin: 0 0 1rem 0;">New themes gaining traction recently</p>
                    <div style="background: white; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">"API integration help"</div>
                        <div style="font-size: 0.9em; color: #666;">Frequency: 67 conversations (+156% this week)</div>
                    </div>
                </div>
                
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #dc3545;">
                    <h3 style="color: #dc3545; margin: 0 0 1rem 0;">⚠️ Anomalies</h3>
                    <p style="margin: 0 0 1rem 0;">Unusual patterns requiring attention</p>
                    <div style="background: white; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">"Unusual error messages"</div>
                        <div style="font-size: 0.9em; color: #666;">Frequency: 12 conversations (potential system issue)</div>
                    </div>
                </div>
            </div>
            
            <button style="margin-top: 2rem; padding: 0.75rem 2rem; background: #6366f1; color: white; border: none; border-radius: 6px; cursor: pointer;" onclick="document.getElementById('pattern-modal').style.display = 'block';">
                🔍 Discover New Patterns
            </button>
        </div>
    `;
}

function showTimelineView() {
    const vizCanvas = document.getElementById('clio-visualization');
    vizCanvas.style.display = 'flex';
    vizCanvas.innerHTML = `
        <div style="width: 100%; padding: 2rem; background: white; border-radius: 8px;">
            <h2>📅 Timeline Analysis</h2>
            <p style="margin-bottom: 2rem; color: #666;">Analyze conversation patterns over time to identify trends and seasonal variations.</p>
            
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;">
                <h3 style="margin: 0 0 1rem 0;">📊 Temporal Insights</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #6366f1;">1,234</div>
                        <div style="font-size: 0.9em; color: #666;">Total Conversations</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #10b981;">+23%</div>
                        <div style="font-size: 0.9em; color: #666;">vs Last Week</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #f59e0b;">2.3</div>
                        <div style="font-size: 0.9em; color: #666;">Avg Daily Peak</div>
                    </div>
                </div>
            </div>
            
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px;">
                <h3 style="margin: 0 0 1rem 0;">📈 Trend Analysis</h3>
                <div style="height: 200px; background: white; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #666;">
                    Interactive timeline chart would be rendered here using D3.js
                </div>
            </div>
        </div>
    `;
}

function showAuditView() {
    const vizCanvas = document.getElementById('clio-visualization');
    vizCanvas.style.display = 'flex';
    vizCanvas.innerHTML = `
        <div style="width: 100%; padding: 2rem; background: white; border-radius: 8px;">
            <h2>🛡️ Privacy & Security Audit</h2>
            <p style="margin-bottom: 2rem; color: #666;">Comprehensive privacy analysis and security audit of conversation data.</p>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
                <div style="background: #d1f2eb; padding: 1.5rem; border-radius: 8px; border: 2px solid #10b981;">
                    <h3 style="color: #10b981; margin: 0 0 1rem 0;">✅ PII Detection</h3>
                    <p style="margin: 0 0 1rem 0;">No personally identifiable information detected</p>
                    <div style="font-size: 0.9em; color: #666;">
                        <div>• Email addresses: 0 found</div>
                        <div>• Phone numbers: 0 found</div>
                        <div>• Credit cards: 0 found</div>
                        <div>• SSNs: 0 found</div>
                    </div>
                </div>
                
                <div style="background: #fff3cd; padding: 1.5rem; border-radius: 8px; border: 2px solid #ffc107;">
                    <h3 style="color: #ffc107; margin: 0 0 1rem 0;">⚠️ Data Sensitivity</h3>
                    <p style="margin: 0 0 1rem 0;">Medium sensitivity level detected</p>
                    <div style="font-size: 0.9em; color: #666;">
                        <div>• User preferences: 45 instances</div>
                        <div>• Technical details: 123 instances</div>
                        <div>• Usage patterns: 67 instances</div>
                    </div>
                </div>
                
                <div style="background: #d1f2eb; padding: 1.5rem; border-radius: 8px; border: 2px solid #10b981;">
                    <h3 style="color: #10b981; margin: 0 0 1rem 0;">🔒 Security Status</h3>
                    <p style="margin: 0 0 1rem 0;">All security checks passed</p>
                    <div style="font-size: 0.9em; color: #666;">
                        <div>• Data encryption: ✅ Active</div>
                        <div>• Access logs: ✅ Monitored</div>
                        <div>• Anonymization: ✅ Applied</div>
                    </div>
                </div>
            </div>
            
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px;">
                <h3 style="margin: 0 0 1rem 0;">📋 Audit Report</h3>
                <div style="background: white; padding: 1rem; border-radius: 4px; font-family: monospace; font-size: 0.9em;">
                    <div>Audit completed: ${new Date().toLocaleString()}</div>
                    <div>Total conversations scanned: 1,234</div>
                    <div>Privacy score: 95/100</div>
                    <div>Compliance status: ✅ GDPR Compliant</div>
                </div>
                <button style="margin-top: 1rem; padding: 0.5rem 1rem; background: #6366f1; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    📥 Download Full Report
                </button>
            </div>
        </div>
    `;
}