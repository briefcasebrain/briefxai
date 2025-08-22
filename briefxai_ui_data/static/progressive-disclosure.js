// Progressive Disclosure Interface Manager

class ProgressiveDisclosureManager {
    constructor() {
        this.currentLevel = 1;
        this.complexityLevel = 'standard'; // simple, standard, expert
        this.userData = null;
        this.analysisResults = null;
        this.userExpertise = 'beginner'; // beginner, intermediate, expert
        this.interactionCount = 0;
        this.helpTips = [];
        this.currentHelpTip = 0;
        this.suggestionQueue = [];
        
        this.init();
    }
    
    init() {
        this.loadUserPreferences();
        this.setupEventListeners();
        this.initializeHelpSystem();
        this.setupDragAndDrop();
        this.startUserObservation();
        
        // Check if coming from onboarding
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.get('onboarding-complete')) {
            this.loadOnboardingResults();
        }
    }
    
    loadUserPreferences() {
        const saved = localStorage.getItem('briefxai_preferences');
        if (saved) {
            const prefs = JSON.parse(saved);
            this.complexityLevel = prefs.complexity || 'standard';
            this.userExpertise = prefs.expertise || 'beginner';
            this.applyComplexityLevel();
        }
    }
    
    saveUserPreferences() {
        const prefs = {
            complexity: this.complexityLevel,
            expertise: this.userExpertise,
            interactionCount: this.interactionCount,
            timestamp: Date.now()
        };
        localStorage.setItem('briefxai_preferences', JSON.stringify(prefs));
    }
    
    setupEventListeners() {
        // File upload
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', this.handleFileUpload.bind(this));
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcut(e);
        });
        
        // Window resize
        window.addEventListener('resize', this.handleResize.bind(this));
        
        // Track user interactions
        document.addEventListener('click', this.trackInteraction.bind(this));
        document.addEventListener('scroll', this.trackInteraction.bind(this));
    }
    
    initializeHelpSystem() {
        this.helpTips = [
            {
                title: "Getting Started",
                content: "Upload your conversation data to begin analysis. We support JSON format with up to 10,000 conversations.",
                target: ".simple-upload-area",
                level: 1
            },
            {
                title: "Understanding Results",
                content: "The key insights show the most important patterns in your data. Click on any card to explore deeper.",
                target: ".key-insights-grid",
                level: 2
            },
            {
                title: "Interactive Exploration",
                content: "Use the visualization controls to explore your data from different perspectives.",
                target: ".viz-controls",
                level: 3
            },
            {
                title: "Expert Features",
                content: "Enable Expert Mode to access advanced analysis tools and raw data views.",
                target: "#expert-toggle",
                level: 3
            }
        ];
    }
    
    setupDragAndDrop() {
        const uploadArea = document.querySelector('.simple-upload-area');
        if (uploadArea) {
            uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadArea.addEventListener('drop', this.handleFileDrop.bind(this));
        }
    }
    
    startUserObservation() {
        // Observe user behavior to adapt interface complexity
        setInterval(() => {
            this.assessUserExpertise();
            this.updateSuggestions();
        }, 30000); // Check every 30 seconds
    }
    
    // Level Management
    goToLevel(level) {
        if (level === this.currentLevel) return;
        
        const currentLevelEl = document.getElementById(`level-${this.currentLevel}`);
        const targetLevelEl = document.getElementById(`level-${level}`);
        
        if (!targetLevelEl) return;
        
        // Add transition classes
        currentLevelEl.classList.remove('active');
        if (level > this.currentLevel) {
            currentLevelEl.classList.add('prev');
        }
        
        setTimeout(() => {
            targetLevelEl.classList.add('active');
            this.currentLevel = level;
            this.onLevelChange(level);
        }, 200);
        
        this.trackInteraction('level_change', { from: this.currentLevel, to: level });
    }
    
    onLevelChange(level) {
        // Update help tips for new level
        this.updateHelpForLevel(level);
        
        // Show appropriate suggestions
        this.showLevelSuggestions(level);
        
        // Update URL without reload
        const url = new URL(window.location);
        url.searchParams.set('level', level);
        window.history.replaceState({}, '', url);
        
        // Level-specific initialization
        switch(level) {
            case 2:
                this.initializeBasicResults();
                break;
            case 3:
                this.initializeAdvancedInterface();
                break;
        }
    }
    
    // File Handling
    triggerFileUpload() {
        document.getElementById('file-input').click();
    }
    
    handleFileUpload(event) {
        const files = event.target.files;
        if (files && files.length > 0) {
            this.processFile(files[0]);
        }
    }
    
    handleDragOver(event) {
        event.preventDefault();
        event.currentTarget.classList.add('dragover');
    }
    
    handleDragLeave(event) {
        event.currentTarget.classList.remove('dragover');
    }
    
    handleFileDrop(event) {
        event.preventDefault();
        const uploadArea = event.currentTarget;
        uploadArea.classList.remove('dragover');
        
        const files = event.dataTransfer.files;
        if (files && files.length > 0) {
            this.processFile(files[0]);
        }
    }
    
    async processFile(file) {
        if (!file.name.endsWith('.json')) {
            this.showError('Please upload a JSON file');
            return;
        }
        
        try {
            this.showProgress();
            const text = await file.text();
            const data = JSON.parse(text);
            
            if (!Array.isArray(data)) {
                throw new Error('Data must be an array of conversations');
            }
            
            this.userData = {
                conversations: data,
                filename: file.name,
                size: file.size,
                uploadTime: Date.now()
            };
            
            await this.simulateAnalysis();
            this.goToLevel(2);
            
        } catch (error) {
            this.showError(`Error processing file: ${error.message}`);
        }
    }
    
    async tryExample() {
        this.showProgress();
        
        // Simulate loading example data
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        this.userData = {
            conversations: this.generateExampleData(),
            filename: 'example_conversations.json',
            size: 125000,
            uploadTime: Date.now(),
            isExample: true
        };
        
        await this.simulateAnalysis();
        this.goToLevel(2);
    }
    
    generateExampleData() {
        // Generate sample conversation data
        const topics = [
            'Technical Support', 'Billing Questions', 'Feature Requests',
            'Bug Reports', 'Account Issues', 'General Inquiries'
        ];
        
        const conversations = [];
        for (let i = 0; i < 1234; i++) {
            conversations.push({
                id: `conv_${i}`,
                topic: topics[Math.floor(Math.random() * topics.length)],
                messages: Math.floor(Math.random() * 10) + 3,
                sentiment: Math.random() * 2 - 1,
                timestamp: Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000
            });
        }
        
        return conversations;
    }
    
    showProgress() {
        const progressHint = document.getElementById('level-1-progress');
        if (progressHint) {
            progressHint.style.display = 'block';
            
            // Animate dots
            const dots = progressHint.querySelectorAll('.dot');
            let currentDot = 0;
            
            const animateDots = () => {
                dots.forEach(dot => dot.classList.remove('active'));
                dots[currentDot].classList.add('active');
                currentDot = (currentDot + 1) % dots.length;
            };
            
            this.progressInterval = setInterval(animateDots, 500);
        }
    }
    
    hideProgress() {
        const progressHint = document.getElementById('level-1-progress');
        if (progressHint) {
            progressHint.style.display = 'none';
        }
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
    }
    
    async simulateAnalysis() {
        // Simulate analysis processing
        const steps = [
            'Loading data...',
            'Analyzing conversations...',
            'Identifying patterns...',
            'Generating insights...',
            'Finalizing results...'
        ];
        
        for (let step of steps) {
            await new Promise(resolve => setTimeout(resolve, 800));
        }
        
        this.analysisResults = this.generateAnalysisResults();
        this.hideProgress();
    }
    
    generateAnalysisResults() {
        if (!this.userData) return null;
        
        const conversations = this.userData.conversations;
        const totalConvs = conversations.length;
        
        // Group by topic
        const topicCounts = {};
        conversations.forEach(conv => {
            topicCounts[conv.topic] = (topicCounts[conv.topic] || 0) + 1;
        });
        
        const sortedTopics = Object.entries(topicCounts)
            .sort((a, b) => b[1] - a[1]);
        
        const mainTopic = sortedTopics[0];
        
        // Calculate sentiment
        const avgSentiment = conversations.reduce((sum, conv) => sum + conv.sentiment, 0) / totalConvs;
        
        return {
            totalConversations: totalConvs,
            mainTopic: {
                name: mainTopic[0],
                count: mainTopic[1],
                percentage: Math.round((mainTopic[1] / totalConvs) * 100)
            },
            sentiment: {
                average: avgSentiment,
                label: avgSentiment > 0.2 ? 'Positive' : avgSentiment < -0.2 ? 'Negative' : 'Neutral',
                percentage: Math.round(Math.abs(avgSentiment) * 100)
            },
            patterns: Math.min(Math.floor(sortedTopics.length / 2), 8),
            topics: sortedTopics.slice(0, 10),
            timestamp: Date.now()
        };
    }
    
    // Results Display
    initializeBasicResults() {
        if (!this.analysisResults) return;
        
        const results = this.analysisResults;
        
        // Update data summary
        const dataSummary = document.getElementById('data-summary');
        if (dataSummary) {
            dataSummary.textContent = `Analyzed ${results.totalConversations.toLocaleString()} conversations`;
        }
        
        // Update main insight
        document.getElementById('main-insight-title').textContent = 'Top Discussion Topic';
        document.getElementById('main-insight-value').textContent = results.mainTopic.name;
        document.getElementById('main-insight-detail').textContent = `${results.mainTopic.percentage}% of conversations`;
        
        // Update sentiment
        document.getElementById('sentiment-value').textContent = results.sentiment.label;
        document.getElementById('sentiment-detail').textContent = `${results.sentiment.percentage}% ${results.sentiment.label.toLowerCase()} tone`;
        
        // Update patterns
        document.getElementById('patterns-value').textContent = `${results.patterns} patterns`;
        document.getElementById('patterns-detail').textContent = 'Including recurring themes';
        
        // Create simple visualization
        this.createSimpleVisualization();
        
        // Show contextual suggestions
        setTimeout(() => {
            this.suggestNextAction('explore_results');
        }, 3000);
    }
    
    createSimpleVisualization() {
        if (!this.analysisResults) return;
        
        const container = document.getElementById('simple-viz');
        if (!container) return;
        
        // Clear existing content
        container.innerHTML = '';
        
        // Create simple bubble chart
        const topics = this.analysisResults.topics.slice(0, 5);
        const maxCount = topics[0][1];
        
        const bubbles = topics.map((topic, i) => {
            const size = Math.max(20, (topic[1] / maxCount) * 60);
            const x = (i + 1) * (container.clientWidth / (topics.length + 1));
            const y = container.clientHeight / 2;
            
            return { name: topic[0], count: topic[1], size, x, y };
        });
        
        // Create SVG
        const svg = d3.select(container)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%');
        
        // Add bubbles
        const bubble = svg.selectAll('.bubble')
            .data(bubbles)
            .enter()
            .append('g')
            .attr('class', 'bubble')
            .attr('transform', d => `translate(${d.x}, ${d.y})`);
        
        bubble.append('circle')
            .attr('r', d => d.size)
            .attr('fill', (d, i) => d3.schemeCategory10[i])
            .attr('opacity', 0.7)
            .on('mouseover', function(event, d) {
                d3.select(this).attr('opacity', 1);
                // Show tooltip
            })
            .on('mouseout', function() {
                d3.select(this).attr('opacity', 0.7);
            });
        
        bubble.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', 5)
            .text(d => d.name)
            .style('font-size', '12px')
            .style('fill', 'white')
            .style('font-weight', 'bold');
    }
    
    // Advanced Interface
    initializeAdvancedInterface() {
        this.setupAdvancedNavigation();
        this.populateAdvancedViews();
        
        // Show expert mode suggestion after some time
        if (this.userExpertise !== 'expert') {
            setTimeout(() => {
                this.suggestExpertMode();
            }, 60000); // After 1 minute
        }
    }
    
    setupAdvancedNavigation() {
        // Set up tab switching
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const view = e.target.getAttribute('data-view');
                this.switchView(view);
            });
        });
    }
    
    switchView(viewName) {
        // Update active tab
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-view="${viewName}"]`).classList.add('active');
        
        // Switch content view
        document.querySelectorAll('.content-view').forEach(view => {
            view.classList.remove('active');
        });
        document.getElementById(`${viewName}-view`).classList.add('active');
        
        this.trackInteraction('view_switch', { view: viewName });
        
        // Load view-specific content
        this.loadViewContent(viewName);
    }
    
    loadViewContent(viewName) {
        switch(viewName) {
            case 'overview':
                this.updateOverviewStats();
                break;
            case 'map':
                this.renderTopicMap();
                break;
            case 'patterns':
                this.loadPatterns();
                break;
            case 'timeline':
                this.renderTimeline();
                break;
            case 'raw-data':
                this.loadRawData();
                break;
        }
    }
    
    populateAdvancedViews() {
        if (!this.analysisResults) return;
        
        const results = this.analysisResults;
        
        // Update overview stats
        document.getElementById('total-convs').textContent = results.totalConversations.toLocaleString();
        document.getElementById('total-topics').textContent = results.topics.length;
        document.getElementById('avg-length').textContent = '156'; // Placeholder
        
        // Create dynamic insights
        this.createDynamicInsights();
    }
    
    updateOverviewStats() {
        // Update stats with real-time calculations
        const insights = document.getElementById('dynamic-insights');
        if (insights && this.analysisResults) {
            insights.innerHTML = `
                <h4>Key Insights</h4>
                <div class="insight-item">
                    <strong>${this.analysisResults.mainTopic.name}</strong> is your most discussed topic
                </div>
                <div class="insight-item">
                    Overall sentiment is <strong>${this.analysisResults.sentiment.label}</strong>
                </div>
                <div class="insight-item">
                    Found <strong>${this.analysisResults.patterns}</strong> recurring patterns
                </div>
            `;
        }
    }
    
    createDynamicInsights() {
        const container = document.getElementById('dynamic-insights');
        if (!container || !this.analysisResults) return;
        
        const insights = [
            `<strong>${this.analysisResults.mainTopic.name}</strong> accounts for ${this.analysisResults.mainTopic.percentage}% of discussions`,
            `Sentiment analysis shows ${this.analysisResults.sentiment.percentage}% ${this.analysisResults.sentiment.label.toLowerCase()} tone`,
            `Discovered ${this.analysisResults.patterns} distinct conversation patterns`
        ];
        
        container.innerHTML = `
            <h4>Dynamic Insights</h4>
            ${insights.map(insight => `<div class="insight-item">${insight}</div>`).join('')}
        `;
    }
    
    // Complexity Management
    toggleExpertMode() {
        const isExpert = document.body.classList.contains('expert-mode');
        
        if (isExpert) {
            this.setComplexity('standard');
        } else {
            this.setComplexity('expert');
        }
    }
    
    setComplexity(level) {
        this.complexityLevel = level;
        this.applyComplexityLevel();
        this.saveUserPreferences();
        
        // Update UI to reflect complexity change
        const expertToggle = document.getElementById('expert-toggle');
        if (expertToggle) {
            expertToggle.textContent = level === 'expert' ? 'Standard Mode' : 'Expert Mode';
        }
        
        this.trackInteraction('complexity_change', { level });
    }
    
    applyComplexityLevel() {
        // Remove existing complexity classes
        document.body.classList.remove('simple-mode', 'standard-mode', 'expert-mode');
        
        // Apply new complexity class
        document.body.classList.add(`${this.complexityLevel}-mode`);
        
        // Show/hide elements based on complexity
        const expertElements = document.querySelectorAll('.expert-only');
        expertElements.forEach(el => {
            if (this.complexityLevel === 'expert') {
                el.style.display = el.dataset.originalDisplay || 'block';
            } else {
                if (!el.dataset.originalDisplay) {
                    el.dataset.originalDisplay = window.getComputedStyle(el).display;
                }
                el.style.display = 'none';
            }
        });
    }
    
    // User Expertise Assessment
    assessUserExpertise() {
        // Analyze user interaction patterns to determine expertise level
        const sessionTime = Date.now() - (this.sessionStart || Date.now());
        const interactionsPerMinute = this.interactionCount / (sessionTime / 60000);
        
        if (interactionsPerMinute > 5 && this.interactionCount > 50) {
            this.userExpertise = 'expert';
        } else if (interactionsPerMinute > 2 && this.interactionCount > 20) {
            this.userExpertise = 'intermediate';
        } else {
            this.userExpertise = 'beginner';
        }
        
        this.adaptToExpertise();
    }
    
    adaptToExpertise() {
        // Automatically adjust interface based on assessed expertise
        if (this.userExpertise === 'expert' && this.complexityLevel !== 'expert') {
            this.suggestExpertMode();
        }
        
        // Adjust help frequency
        const helpFrequency = {
            'beginner': 0.3,
            'intermediate': 0.15,
            'expert': 0.05
        };
        
        this.helpProbability = helpFrequency[this.userExpertise];
    }
    
    // Suggestion System
    suggestNextAction(context) {
        if (!this.shouldShowSuggestion()) return;
        
        const suggestions = {
            'explore_results': {
                icon: '💡',
                text: 'Try clicking on the topic bubbles to explore deeper insights!',
                target: '.simple-viz-container'
            },
            'advanced_features': {
                icon: '🚀',
                text: 'Ready for more? Click "More Options" to access advanced features.',
                target: '#more-options-btn'
            },
            'expert_mode': {
                icon: '🎯',
                text: 'You seem experienced! Try Expert Mode for full control.',
                target: '#expert-toggle'
            }
        };
        
        const suggestion = suggestions[context];
        if (suggestion) {
            this.showSuggestion(suggestion);
        }
    }
    
    shouldShowSuggestion() {
        const autoSuggestions = document.getElementById('auto-suggestions');
        return autoSuggestions && autoSuggestions.checked && 
               Math.random() < (this.helpProbability || 0.2);
    }
    
    showSuggestion(suggestion) {
        const overlay = document.getElementById('suggestion-overlay');
        const bubble = document.getElementById('suggestion-bubble');
        const content = document.getElementById('suggestion-content');
        
        if (!overlay || !bubble || !content) return;
        
        // Update content
        content.innerHTML = `
            <span class="suggestion-icon">${suggestion.icon}</span>
            <p>${suggestion.text}</p>
        `;
        
        // Position near target element
        const target = document.querySelector(suggestion.target);
        if (target) {
            const rect = target.getBoundingClientRect();
            overlay.style.top = (rect.top - 80) + 'px';
            overlay.style.left = rect.left + 'px';
        }
        
        // Show suggestion
        overlay.style.display = 'block';
        
        // Auto-hide after 8 seconds
        setTimeout(() => {
            this.dismissSuggestion();
        }, 8000);
    }
    
    dismissSuggestion() {
        const overlay = document.getElementById('suggestion-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
    
    suggestExpertMode() {
        this.suggestNextAction('expert_mode');
    }
    
    // Help System
    showHelp() {
        const helpSystem = document.getElementById('help-system');
        if (helpSystem) {
            helpSystem.style.display = 'block';
            this.showCurrentHelpTip();
        }
    }
    
    hideHelp() {
        const helpSystem = document.getElementById('help-system');
        if (helpSystem) {
            helpSystem.style.display = 'none';
        }
    }
    
    showCurrentHelpTip() {
        const tip = this.helpTips[this.currentHelpTip];
        if (!tip) return;
        
        document.getElementById('help-title').textContent = tip.title;
        document.getElementById('help-content').innerHTML = `<p>${tip.content}</p>`;
        
        // Update navigation buttons
        const prevBtn = document.getElementById('help-prev');
        const nextBtn = document.getElementById('help-next');
        
        prevBtn.disabled = this.currentHelpTip === 0;
        nextBtn.disabled = this.currentHelpTip === this.helpTips.length - 1;
        nextBtn.textContent = this.currentHelpTip === this.helpTips.length - 1 ? 'Done' : 'Next →';
    }
    
    nextHelpTip() {
        if (this.currentHelpTip < this.helpTips.length - 1) {
            this.currentHelpTip++;
            this.showCurrentHelpTip();
        } else {
            this.hideHelp();
        }
    }
    
    prevHelpTip() {
        if (this.currentHelpTip > 0) {
            this.currentHelpTip--;
            this.showCurrentHelpTip();
        }
    }
    
    updateHelpForLevel(level) {
        // Filter help tips for current level
        this.currentHelpTip = this.helpTips.findIndex(tip => tip.level === level);
        if (this.currentHelpTip === -1) this.currentHelpTip = 0;
    }
    
    // Settings Management
    showSettings() {
        const panel = document.getElementById('settings-panel');
        if (panel) {
            panel.classList.add('active');
        }
    }
    
    hideSettings() {
        const panel = document.getElementById('settings-panel');
        if (panel) {
            panel.classList.remove('active');
        }
    }
    
    updateSettings() {
        // Update settings based on form inputs
        const autoSuggestions = document.getElementById('auto-suggestions').checked;
        const smartDefaults = document.getElementById('smart-defaults').checked;
        const debugMode = document.getElementById('debug-mode')?.checked || false;
        const performanceMetrics = document.getElementById('performance-metrics')?.checked || false;
        
        const settings = {
            autoSuggestions,
            smartDefaults,
            debugMode,
            performanceMetrics
        };
        
        localStorage.setItem('briefxai_settings', JSON.stringify(settings));
        this.applySettings(settings);
    }
    
    applySettings(settings) {
        // Apply settings to interface
        if (settings.debugMode) {
            document.body.classList.add('debug-mode');
        } else {
            document.body.classList.remove('debug-mode');
        }
        
        if (settings.performanceMetrics) {
            this.enablePerformanceMetrics();
        } else {
            this.disablePerformanceMetrics();
        }
    }
    
    // Utility Functions
    trackInteraction(type, data = {}) {
        this.interactionCount++;
        
        const interaction = {
            type,
            data,
            timestamp: Date.now(),
            level: this.currentLevel,
            expertise: this.userExpertise
        };
        
        // Store interaction for analysis
        const interactions = JSON.parse(localStorage.getItem('briefxai_interactions') || '[]');
        interactions.push(interaction);
        
        // Keep only last 1000 interactions
        if (interactions.length > 1000) {
            interactions.splice(0, interactions.length - 1000);
        }
        
        localStorage.setItem('briefxai_interactions', JSON.stringify(interactions));
    }
    
    handleKeyboardShortcut(e) {
        // Keyboard shortcuts for power users
        if (e.ctrlKey || e.metaKey) {
            switch(e.key) {
                case '1':
                    e.preventDefault();
                    this.goToLevel(1);
                    break;
                case '2':
                    e.preventDefault();
                    this.goToLevel(2);
                    break;
                case '3':
                    e.preventDefault();
                    this.goToLevel(3);
                    break;
                case 'h':
                    e.preventDefault();
                    this.showHelp();
                    break;
                case ',':
                    e.preventDefault();
                    this.showSettings();
                    break;
            }
        }
        
        if (e.key === 'Escape') {
            this.hideHelp();
            this.hideSettings();
            this.dismissSuggestion();
        }
    }
    
    handleResize() {
        // Handle responsive behavior
        if (window.innerWidth < 768 && this.currentLevel === 3) {
            // Simplify interface on mobile
            this.setComplexity('simple');
        }
    }
    
    showError(message) {
        // Create error notification
        const notification = document.createElement('div');
        notification.className = 'error-notification';
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">⚠️</span>
                <span class="notification-message">${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }
    
    loadOnboardingResults() {
        // Load results from onboarding if available
        const results = localStorage.getItem('briefxai_analysis_results');
        const data = localStorage.getItem('briefxai_uploaded_data');
        
        if (results) {
            this.analysisResults = JSON.parse(results);
        }
        
        if (data) {
            this.userData = {
                conversations: JSON.parse(data),
                filename: 'onboarding_data.json',
                uploadTime: Date.now()
            };
        }
        
        if (this.userData && this.analysisResults) {
            this.goToLevel(2);
        }
    }
}

// Global Functions for HTML onclick handlers
let pdManager;

function triggerFileUpload() {
    pdManager.triggerFileUpload();
}

function tryExample() {
    pdManager.tryExample();
}

function showMoreOptions() {
    pdManager.goToLevel(3);
}

function exploreTopics() {
    pdManager.goToLevel(3);
    setTimeout(() => pdManager.switchView('map'), 300);
}

function showPatterns() {
    pdManager.goToLevel(3);
    setTimeout(() => pdManager.switchView('patterns'), 300);
}

function exportResults() {
    // Implement export functionality
    console.log('Exporting results...');
}

function expandVisualization() {
    pdManager.goToLevel(3);
}

function switchView(view) {
    pdManager.switchView(view);
}

function toggleExpertMode() {
    pdManager.toggleExpertMode();
}

function showSettings() {
    pdManager.showSettings();
}

function hideSettings() {
    pdManager.hideSettings();
}

function showHelp() {
    pdManager.showHelp();
}

function hideHelp() {
    pdManager.hideHelp();
}

function nextHelpTip() {
    pdManager.nextHelpTip();
}

function prevHelpTip() {
    pdManager.prevHelpTip();
}

function setComplexity(level) {
    pdManager.setComplexity(level);
}

function updateSettings() {
    pdManager.updateSettings();
}

function dismissSuggestion() {
    pdManager.dismissSuggestion();
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    pdManager = new ProgressiveDisclosureManager();
    window.pdManager = pdManager; // For debugging
});