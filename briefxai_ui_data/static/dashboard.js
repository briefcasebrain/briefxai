// BriefXAI Dashboard JavaScript
// Modern, interactive UI with real-time updates and 3D visualizations

class BriefXAIDashboard {
    constructor() {
        this.currentTheme = localStorage.getItem('theme') || 'light';
        this.ws = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.clusters = [];
        this.selectedCluster = null;
        
        this.init();
    }

    init() {
        this.applyTheme(this.currentTheme);
        this.initWebSocket();
        this.initEventListeners();
        this.init3DVisualization();
        this.initCommandPalette();
        this.loadInitialData();
    }

    // Theme Management
    applyTheme(theme) {
        document.body.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        this.currentTheme = theme;
        
        const themeBtn = document.getElementById('theme-btn');
        themeBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
    }

    toggleTheme() {
        const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        this.applyTheme(newTheme);
    }

    // WebSocket Connection
    initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
            // Attempt to reconnect after 5 seconds
            setTimeout(() => this.initWebSocket(), 5000);
        };
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'progress':
                this.updateProgress(data.percentage);
                break;
            case 'cluster_update':
                this.updateCluster(data.cluster);
                break;
            case 'insight':
                this.addInsight(data.insight);
                break;
            case 'metric_update':
                this.updateMetrics(data.metrics);
                break;
            case 'complete':
                this.onAnalysisComplete(data.results);
                break;
        }
        
        // Add to live updates feed
        this.addLiveUpdate(data);
    }

    updateConnectionStatus(connected) {
        const statusIndicator = document.querySelector('.status-item.success, .status-item.error');
        if (statusIndicator) {
            statusIndicator.textContent = connected ? '● Connected' : '● Disconnected';
            statusIndicator.className = `status-item ${connected ? 'success' : 'error'}`;
        }
    }

    // 3D Visualization
    init3DVisualization() {
        const container = document.getElementById('constellation-view');
        if (!container) return;

        // Three.js setup
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(
            75,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );

        this.renderer = new THREE.WebGLRenderer({ 
            canvas: document.getElementById('3d-canvas'),
            antialias: true,
            alpha: true
        });
        
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(10, 10, 5);
        this.scene.add(directionalLight);

        // Camera position
        this.camera.position.z = 50;

        // Add controls
        this.initControls();

        // Start animation loop
        this.animate();
    }

    initControls() {
        // Add mouse controls for 3D view
        const container = document.getElementById('constellation-view');
        let mouseX = 0;
        let mouseY = 0;

        container.addEventListener('mousemove', (event) => {
            mouseX = (event.clientX / window.innerWidth) * 2 - 1;
            mouseY = -(event.clientY / window.innerHeight) * 2 + 1;
        });

        // Zoom controls
        container.addEventListener('wheel', (event) => {
            event.preventDefault();
            this.camera.position.z += event.deltaY * 0.1;
            this.camera.position.z = Math.max(10, Math.min(100, this.camera.position.z));
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Rotate clusters slowly
        if (this.clusterMeshes) {
            this.clusterMeshes.forEach(mesh => {
                mesh.rotation.x += 0.001;
                mesh.rotation.y += 0.002;
            });
        }

        this.renderer.render(this.scene, this.camera);
    }

    addClusterTo3DView(cluster) {
        const geometry = new THREE.SphereGeometry(
            Math.log(cluster.size + 1) * 2,
            32,
            32
        );
        
        const material = new THREE.MeshPhongMaterial({
            color: this.getClusterColor(cluster.sentiment),
            transparent: true,
            opacity: 0.8
        });
        
        const sphere = new THREE.Mesh(geometry, material);
        
        // Position based on UMAP coordinates
        sphere.position.x = cluster.x * 20;
        sphere.position.y = cluster.y * 20;
        sphere.position.z = cluster.z * 20 || 0;
        
        sphere.userData = cluster;
        this.scene.add(sphere);
        
        if (!this.clusterMeshes) this.clusterMeshes = [];
        this.clusterMeshes.push(sphere);
    }

    getClusterColor(sentiment) {
        if (sentiment > 0.5) return 0x10b981; // Positive - green
        if (sentiment < -0.5) return 0xef4444; // Negative - red
        return 0xf59e0b; // Neutral - yellow
    }

    // Event Listeners
    initEventListeners() {
        // Theme toggle
        document.getElementById('theme-btn')?.addEventListener('click', () => {
            this.toggleTheme();
        });

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.view);
            });
        });

        // Navigation items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                this.switchView(e.target.dataset.view);
            });
        });

        // Natural language query
        const nlInput = document.getElementById('nl-query');
        if (nlInput) {
            nlInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.handleNaturalLanguageQuery(e.target.value);
                }
            });
        }

        // Add facet button
        document.querySelector('.add-facet-btn')?.addEventListener('click', () => {
            this.showFacetBuilder();
        });

        // Canvas controls
        document.querySelectorAll('.control-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.handleCanvasControl(e.target.title);
            });
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });
    }

    switchTab(view) {
        // Update active tab
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`.tab-btn[data-view="${view}"]`)?.classList.add('active');

        // Show corresponding view
        document.querySelectorAll('.viz-view').forEach(v => {
            v.classList.remove('active');
        });
        document.getElementById(`${view}-view`)?.classList.add('active');

        // Load view-specific content
        this.loadViewContent(view);
    }

    switchView(view) {
        // Update active navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`.nav-item[data-view="${view}"]`)?.classList.add('active');

        // Load view content
        this.loadMainView(view);
    }

    // Command Palette
    initCommandPalette() {
        document.addEventListener('keydown', (e) => {
            // Cmd/Ctrl + K to open command palette
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                this.toggleCommandPalette();
            }
            
            // Escape to close
            if (e.key === 'Escape') {
                this.closeCommandPalette();
            }
        });

        const commandInput = document.getElementById('command-input');
        if (commandInput) {
            commandInput.addEventListener('input', (e) => {
                this.searchCommands(e.target.value);
            });
        }
    }

    toggleCommandPalette() {
        const palette = document.getElementById('command-palette');
        if (palette) {
            palette.classList.toggle('hidden');
            if (!palette.classList.contains('hidden')) {
                document.getElementById('command-input')?.focus();
            }
        }
    }

    closeCommandPalette() {
        document.getElementById('command-palette')?.classList.add('hidden');
    }

    searchCommands(query) {
        // Implement fuzzy search for commands
        const commands = [
            { name: 'New Analysis', action: () => this.startNewAnalysis() },
            { name: 'Export to PDF', action: () => this.exportToPDF() },
            { name: 'Share Results', action: () => this.shareResults() },
            { name: 'Toggle Theme', action: () => this.toggleTheme() },
            { name: 'Add Custom Facet', action: () => this.showFacetBuilder() },
            { name: 'View Keyboard Shortcuts', action: () => this.showShortcuts() }
        ];

        const results = commands.filter(cmd => 
            cmd.name.toLowerCase().includes(query.toLowerCase())
        );

        this.displayCommandResults(results);
    }

    displayCommandResults(results) {
        const container = document.getElementById('command-results');
        if (!container) return;

        container.innerHTML = results.map(cmd => `
            <div class="command-result" onclick="dashboard.executeCommand('${cmd.name}')">
                <span class="command-name">${cmd.name}</span>
            </div>
        `).join('');
    }

    // Natural Language Query
    handleNaturalLanguageQuery(query) {
        if (!query.trim()) return;

        // Send query to backend
        fetch('/api/nl-query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        })
        .then(res => res.json())
        .then(data => {
            this.displayQueryResults(data);
        })
        .catch(err => {
            console.error('Query error:', err);
            this.showNotification('Query failed', 'error');
        });
    }

    // Facet Builder
    showFacetBuilder() {
        const modal = document.getElementById('facet-builder-modal');
        if (modal) {
            modal.classList.add('active');
        }
    }

    // Progress Updates
    updateProgress(percentage) {
        const progressBar = document.querySelector('.status-progress');
        if (progressBar) {
            progressBar.value = percentage;
        }
        
        const progressText = document.querySelector('.status-item span');
        if (progressText) {
            progressText.textContent = `${percentage}%`;
        }
    }

    // Live Updates
    addLiveUpdate(data) {
        const feed = document.getElementById('update-feed');
        if (!feed) return;

        const update = document.createElement('div');
        update.className = 'update-item';
        update.innerHTML = `
            <span class="update-time">Just now</span>
            <span class="update-text">${this.formatUpdateMessage(data)}</span>
        `;
        
        feed.insertBefore(update, feed.firstChild);
        
        // Limit to 10 updates
        while (feed.children.length > 10) {
            feed.removeChild(feed.lastChild);
        }
    }

    formatUpdateMessage(data) {
        switch (data.type) {
            case 'cluster_update':
                return `Cluster "${data.cluster.name}" updated`;
            case 'insight':
                return `New insight: ${data.insight.title}`;
            case 'progress':
                return `Analysis ${data.percentage}% complete`;
            default:
                return 'System update';
        }
    }

    // Insights
    addInsight(insight) {
        const container = document.querySelector('.insights-list');
        if (!container) return;

        const insightCard = document.createElement('div');
        insightCard.className = `insight-card ${insight.type}`;
        insightCard.innerHTML = `
            <span class="insight-icon">${this.getInsightIcon(insight.type)}</span>
            <div class="insight-content">
                <h4>${insight.title}</h4>
                <p>${insight.description}</p>
                <button class="insight-action" onclick="dashboard.handleInsightAction('${insight.id}')">
                    ${insight.actionText || 'View'}
                </button>
            </div>
        `;
        
        container.insertBefore(insightCard, container.firstChild);
    }

    getInsightIcon(type) {
        const icons = {
            warning: '⚠️',
            suggestion: '💡',
            info: '📊',
            success: '✅',
            error: '❌'
        };
        return icons[type] || '📌';
    }

    // Metrics
    updateMetrics(metrics) {
        Object.keys(metrics).forEach(key => {
            const element = document.querySelector(`[data-metric="${key}"]`);
            if (element) {
                element.textContent = metrics[key];
            }
        });
    }

    // Data Loading
    loadInitialData() {
        // Simulate loading initial data
        setTimeout(() => {
            this.createSampleClusters();
            this.updateProgress(73);
        }, 1000);
    }

    createSampleClusters() {
        const sampleClusters = [
            {
                id: 1,
                name: 'Password Reset Issues',
                size: 234,
                sentiment: -0.8,
                x: -10,
                y: 5,
                z: 0
            },
            {
                id: 2,
                name: 'Feature Requests',
                size: 156,
                sentiment: 0.3,
                x: 15,
                y: -8,
                z: 5
            },
            {
                id: 3,
                name: 'Billing Questions',
                size: 89,
                sentiment: -0.4,
                x: 0,
                y: 12,
                z: -10
            }
        ];

        sampleClusters.forEach(cluster => {
            this.addClusterTo3DView(cluster);
            this.addClusterCard(cluster);
        });
    }

    addClusterCard(cluster) {
        const grid = document.getElementById('cluster-grid');
        if (!grid) return;

        const card = document.createElement('div');
        card.className = 'cluster-card';
        card.innerHTML = `
            <div class="cluster-header">
                <span class="cluster-icon">🔵</span>
                <h3>${cluster.name}</h3>
                <span class="cluster-badge ${cluster.sentiment < 0 ? 'negative' : 'positive'}">
                    ${Math.abs(cluster.sentiment * 100).toFixed(0)}% ${cluster.sentiment < 0 ? 'negative' : 'positive'}
                </span>
            </div>
            <div class="cluster-stats">
                <span class="stat">${cluster.size} conversations</span>
            </div>
            <div class="cluster-actions">
                <button class="action-btn" onclick="dashboard.viewCluster(${cluster.id})">View All</button>
                <button class="action-btn" onclick="dashboard.exportCluster(${cluster.id})">Export</button>
                <button class="action-btn primary" onclick="dashboard.investigateCluster(${cluster.id})">Investigate</button>
            </div>
        `;
        
        grid.appendChild(card);
    }

    // Cluster Actions
    viewCluster(id) {
        console.log('Viewing cluster', id);
        // Implement cluster detail view
    }

    exportCluster(id) {
        console.log('Exporting cluster', id);
        // Implement export functionality
    }

    investigateCluster(id) {
        console.log('Investigating cluster', id);
        // Implement investigation mode
    }

    // Export Functions
    exportToPDF() {
        console.log('Exporting to PDF...');
        // Implement PDF export
    }

    shareResults() {
        console.log('Sharing results...');
        // Implement sharing functionality
    }

    // Utility Functions
    handleResize() {
        if (this.camera && this.renderer) {
            const container = document.getElementById('constellation-view');
            this.camera.aspect = container.clientWidth / container.clientHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(container.clientWidth, container.clientHeight);
        }
    }

    showNotification(message, type = 'info') {
        // Implement notification system
        console.log(`${type}: ${message}`);
    }

    showShortcuts() {
        // Show keyboard shortcuts modal
        console.log('Showing keyboard shortcuts');
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new BriefXAIDashboard();
});