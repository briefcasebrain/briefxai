// Dynamic Data Loading Manager

class DynamicDataLoader {
    constructor() {
        this.isRealTimeMode = false;
        this.websocket = null;
        this.eventSource = null;
        this.pollingInterval = null;
        this.uploadMode = 'file';
        
        // Data processing
        this.dataQueue = [];
        this.processingQueue = [];
        this.processedData = [];
        this.processingRate = 1; // items per second
        this.maxQueueSize = 1000;
        
        // Metrics
        this.metrics = {
            conversations: 0,
            topics: 0,
            sentiment: 0.0,
            processingRate: 0,
            messagesPerSec: 0,
            totalReceived: 0,
            bufferSize: 0
        };
        
        // Connection status
        this.connectionStatus = 'disconnected';
        this.connectionLatency = 0;
        this.lastPingTime = 0;
        
        // UI state
        this.animationsEnabled = true;
        this.autoScrollEnabled = true;
        this.topicUpdatesEnabled = true;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.updateConnectionStatus();
        this.startMetricsUpdater();
        this.startProcessingLoop();
        
        // Load saved preferences
        this.loadPreferences();
    }
    
    setupEventListeners() {
        // File input
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', this.handleFileUpload.bind(this));
        }
        
        // Upload zone drag and drop
        const uploadZone = document.getElementById('upload-zone');
        if (uploadZone) {
            uploadZone.addEventListener('click', () => fileInput?.click());
            uploadZone.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadZone.addEventListener('drop', this.handleFileDrop.bind(this));
        }
        
        // Stream configuration
        const streamType = document.getElementById('stream-type');
        if (streamType) {
            streamType.addEventListener('change', this.updateStreamConfig.bind(this));
        }
        
        // Visualization controls
        const vizType = document.getElementById('viz-type');
        if (vizType) {
            vizType.addEventListener('change', this.changeVisualizationType.bind(this));
        }
        
        // Window events
        window.addEventListener('beforeunload', this.cleanup.bind(this));
        window.addEventListener('online', () => this.handleConnectionChange(true));
        window.addEventListener('offline', () => this.handleConnectionChange(false));
    }
    
    // Upload Mode Management
    setUploadMode(mode) {
        this.uploadMode = mode;
        
        // Update UI
        document.querySelectorAll('.upload-mode').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
        
        // Show/hide content
        document.querySelectorAll('.upload-content').forEach(content => {
            content.style.display = 'none';
        });
        document.getElementById(`${mode}-upload`).style.display = 'block';
        
        this.showNotification('info', `Switched to ${mode} mode`, `Upload mode changed to ${mode}`);
    }
    
    // File Upload Handling
    handleFileUpload(event) {
        const files = event.target.files;
        if (files && files.length > 0) {
            this.processFiles(files);
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
        const uploadZone = event.currentTarget;
        uploadZone.classList.remove('dragover');
        
        const files = event.dataTransfer.files;
        if (files && files.length > 0) {
            this.processFiles(files);
        }
    }
    
    async processFiles(files) {
        for (let file of files) {
            await this.processFile(file);
        }
    }
    
    async processFile(file) {
        try {
            this.showUploadProgress();
            const data = await this.readFile(file);
            
            if (this.isRealTimeMode) {
                this.addToQueue(data);
            } else {
                await this.processBatchData(data);
            }
            
            this.hideUploadProgress();
            this.showAnalysisDashboard();
            
        } catch (error) {
            this.showNotification('error', 'Upload Failed', error.message);
            this.hideUploadProgress();
        }
    }
    
    async readFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onprogress = (event) => {
                if (event.lengthComputable) {
                    const progress = (event.loaded / event.total) * 100;
                    this.updateUploadProgress(progress);
                }
            };
            
            reader.onload = (event) => {
                try {
                    const result = event.target.result;
                    let data;
                    
                    if (file.name.endsWith('.json')) {
                        data = JSON.parse(result);
                    } else if (file.name.endsWith('.csv')) {
                        data = this.parseCSV(result);
                    } else {
                        throw new Error('Unsupported file format');
                    }
                    
                    resolve(data);
                } catch (error) {
                    reject(new Error(`Error parsing file: ${error.message}`));
                }
            };
            
            reader.onerror = () => {
                reject(new Error('Error reading file'));
            };
            
            reader.readAsText(file);
        });
    }
    
    parseCSV(csvText) {
        const lines = csvText.split('\n');
        const headers = lines[0].split(',');
        const data = [];
        
        for (let i = 1; i < lines.length; i++) {
            if (lines[i].trim()) {
                const values = lines[i].split(',');
                const row = {};
                headers.forEach((header, index) => {
                    row[header.trim()] = values[index]?.trim();
                });
                data.push(row);
            }
        }
        
        return data;
    }
    
    // Real-time Data Streaming
    toggleRealTimeMode() {
        this.isRealTimeMode = !this.isRealTimeMode;
        
        const toggle = document.getElementById('realtime-toggle');
        if (toggle) {
            toggle.textContent = `📡 Real-time: ${this.isRealTimeMode ? 'ON' : 'OFF'}`;
            toggle.classList.toggle('active', this.isRealTimeMode);
        }
        
        if (this.isRealTimeMode) {
            this.showAnalysisDashboard();
        }
        
        this.showNotification('info', 'Real-time Mode', 
            `Real-time processing ${this.isRealTimeMode ? 'enabled' : 'disabled'}`);
    }
    
    updateStreamConfig() {
        const streamType = document.getElementById('stream-type').value;
        const urlInput = document.getElementById('stream-url');
        
        switch (streamType) {
            case 'websocket':
                urlInput.placeholder = 'ws://localhost:8080/stream';
                break;
            case 'sse':
                urlInput.placeholder = 'http://localhost:8080/events';
                break;
            case 'polling':
                urlInput.placeholder = 'http://localhost:8080/data';
                break;
        }
    }
    
    async startStream() {
        const streamType = document.getElementById('stream-type').value;
        const streamUrl = document.getElementById('stream-url').value;
        const batchSize = parseInt(document.getElementById('batch-size').value);
        
        if (!streamUrl) {
            this.showNotification('error', 'Configuration Error', 'Please enter a valid stream URL');
            return;
        }
        
        try {
            this.showLoadingOverlay('Connecting to stream...', 'Establishing connection...');
            
            switch (streamType) {
                case 'websocket':
                    await this.startWebSocketStream(streamUrl);
                    break;
                case 'sse':
                    await this.startSSEStream(streamUrl);
                    break;
                case 'polling':
                    await this.startPollingStream(streamUrl, batchSize);
                    break;
            }
            
            this.hideLoadingOverlay();
            this.showStreamMonitor();
            this.showAnalysisDashboard();
            
            this.showNotification('success', 'Stream Connected', 
                `Successfully connected to ${streamType} stream`);
            
        } catch (error) {
            this.hideLoadingOverlay();
            this.showNotification('error', 'Connection Failed', error.message);
        }
    }
    
    async startWebSocketStream(url) {
        return new Promise((resolve, reject) => {
            try {
                this.websocket = new WebSocket(url);
                
                this.websocket.onopen = () => {
                    this.connectionStatus = 'connected';
                    this.updateConnectionStatus();
                    this.addStreamLog('info', 'WebSocket connection established');
                    resolve();
                };
                
                this.websocket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.handleStreamData(data);
                    } catch (error) {
                        this.addStreamLog('error', `Invalid JSON received: ${error.message}`);
                    }
                };
                
                this.websocket.onclose = (event) => {
                    this.connectionStatus = 'disconnected';
                    this.updateConnectionStatus();
                    this.addStreamLog('warning', `WebSocket closed: ${event.reason}`);
                    
                    // Auto-reconnect after 5 seconds
                    setTimeout(() => {
                        if (this.isRealTimeMode) {
                            this.startWebSocketStream(url);
                        }
                    }, 5000);
                };
                
                this.websocket.onerror = (error) => {
                    this.connectionStatus = 'error';
                    this.updateConnectionStatus();
                    this.addStreamLog('error', `WebSocket error: ${error.message}`);
                    reject(error);
                };
                
            } catch (error) {
                reject(error);
            }
        });
    }
    
    async startSSEStream(url) {
        try {
            this.eventSource = new EventSource(url);
            
            this.eventSource.onopen = () => {
                this.connectionStatus = 'connected';
                this.updateConnectionStatus();
                this.addStreamLog('info', 'SSE connection established');
            };
            
            this.eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleStreamData(data);
                } catch (error) {
                    this.addStreamLog('error', `Invalid JSON received: ${error.message}`);
                }
            };
            
            this.eventSource.onerror = (error) => {
                this.connectionStatus = 'error';
                this.updateConnectionStatus();
                this.addStreamLog('error', `SSE error: ${error.message}`);
            };
            
        } catch (error) {
            throw error;
        }
    }
    
    async startPollingStream(url, interval = 5000) {
        const pollData = async () => {
            try {
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                this.handleStreamData(data);
                this.connectionStatus = 'connected';
                
            } catch (error) {
                this.connectionStatus = 'error';
                this.addStreamLog('error', `Polling error: ${error.message}`);
            }
            
            this.updateConnectionStatus();
        };
        
        // Initial poll
        await pollData();
        
        // Set up interval
        this.pollingInterval = setInterval(pollData, interval);
        this.addStreamLog('info', `Polling started with ${interval}ms interval`);
    }
    
    stopStream() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
        
        this.connectionStatus = 'disconnected';
        this.updateConnectionStatus();
        this.hideStreamMonitor();
        
        this.addStreamLog('info', 'Stream stopped');
        this.showNotification('info', 'Stream Stopped', 'Data stream has been disconnected');
    }
    
    // Data Processing
    handleStreamData(data) {
        this.metrics.totalReceived++;
        this.metrics.messagesPerSec = this.calculateMessagesPerSecond();
        
        if (Array.isArray(data)) {
            data.forEach(item => this.addToQueue(item));
        } else {
            this.addToQueue(data);
        }
        
        this.addStreamLog('success', `Received ${Array.isArray(data) ? data.length : 1} items`);
    }
    
    addToQueue(data) {
        if (this.dataQueue.length >= this.maxQueueSize) {
            this.dataQueue.shift(); // Remove oldest item
            this.addStreamLog('warning', 'Queue full, dropping oldest item');
        }
        
        this.dataQueue.push({
            data,
            timestamp: Date.now(),
            id: Math.random().toString(36).substr(2, 9)
        });
        
        this.metrics.bufferSize = this.dataQueue.length;
    }
    
    startProcessingLoop() {
        const processItems = () => {
            const batchSize = Math.min(10, this.dataQueue.length);
            
            if (batchSize > 0) {
                const items = this.dataQueue.splice(0, batchSize);
                
                items.forEach(item => {
                    this.processingQueue.push(item);
                    this.processItem(item);
                });
            }
            
            setTimeout(processItems, 1000 / this.processingRate);
        };
        
        processItems();
    }
    
    async processItem(item) {
        try {
            // Simulate processing time
            await new Promise(resolve => setTimeout(resolve, Math.random() * 500));
            
            // Process the item
            const processed = this.analyzeConversation(item.data);
            this.processedData.push(processed);
            
            // Update metrics
            this.updateMetricsFromItem(processed);
            
            // Update UI
            this.updateLiveVisualization(processed);
            this.updateTopicsStream(processed);
            
            // Remove from processing queue
            const index = this.processingQueue.findIndex(p => p.id === item.id);
            if (index >= 0) {
                this.processingQueue.splice(index, 1);
            }
            
        } catch (error) {
            this.addStreamLog('error', `Processing failed: ${error.message}`);
        }
    }
    
    analyzeConversation(data) {
        // Simulate conversation analysis
        const topics = ['Support', 'Billing', 'Features', 'Bugs', 'Account'];
        const topic = topics[Math.floor(Math.random() * topics.length)];
        const sentiment = (Math.random() - 0.5) * 2; // -1 to 1
        
        return {
            ...data,
            topic,
            sentiment,
            analyzed: true,
            timestamp: Date.now(),
            processingTime: Math.random() * 500
        };
    }
    
    updateMetricsFromItem(item) {
        this.metrics.conversations++;
        this.metrics.sentiment = (this.metrics.sentiment + item.sentiment) / 2;
        
        // Update topics count
        const uniqueTopics = new Set(this.processedData.map(d => d.topic));
        this.metrics.topics = uniqueTopics.size;
    }
    
    // UI Updates
    updateConnectionStatus() {
        const statusIndicator = document.getElementById('status-indicator');
        const statusDot = statusIndicator?.querySelector('.status-dot');
        const statusText = statusIndicator?.querySelector('.status-text');
        
        if (!statusDot || !statusText) return;
        
        statusDot.className = 'status-dot';
        
        switch (this.connectionStatus) {
            case 'connected':
                statusDot.classList.add('connected');
                statusText.textContent = 'Connected';
                break;
            case 'error':
                statusDot.classList.add('error');
                statusText.textContent = 'Error';
                break;
            default:
                statusText.textContent = 'Disconnected';
        }
        
        // Update metrics
        const latencyEl = document.getElementById('latency');
        const queueSizeEl = document.getElementById('queue-size');
        
        if (latencyEl) latencyEl.textContent = `${this.connectionLatency}ms`;
        if (queueSizeEl) queueSizeEl.textContent = this.dataQueue.length;
    }
    
    showUploadProgress() {
        const uploadProgress = document.getElementById('upload-progress');
        const uploadZone = document.getElementById('upload-zone');
        
        if (uploadProgress) uploadProgress.style.display = 'block';
        if (uploadZone) uploadZone.classList.add('uploading');
    }
    
    hideUploadProgress() {
        const uploadProgress = document.getElementById('upload-progress');
        const uploadZone = document.getElementById('upload-zone');
        
        if (uploadProgress) uploadProgress.style.display = 'none';
        if (uploadZone) uploadZone.classList.remove('uploading');
    }
    
    updateUploadProgress(percentage) {
        const progressFill = document.getElementById('upload-progress-fill');
        const statusText = document.getElementById('upload-status');
        
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }
        
        if (statusText) {
            statusText.textContent = `Uploading... ${Math.round(percentage)}%`;
        }
    }
    
    showAnalysisDashboard() {
        const dashboard = document.getElementById('analysis-dashboard');
        if (dashboard) {
            dashboard.style.display = 'block';
            this.startMetricsUpdater();
        }
    }
    
    showStreamMonitor() {
        const monitor = document.getElementById('stream-monitor');
        const config = document.querySelector('.stream-config');
        
        if (monitor) monitor.style.display = 'block';
        if (config) config.style.display = 'none';
    }
    
    hideStreamMonitor() {
        const monitor = document.getElementById('stream-monitor');
        const config = document.querySelector('.stream-config');
        
        if (monitor) monitor.style.display = 'none';
        if (config) config.style.display = 'block';
    }
    
    startMetricsUpdater() {
        if (this.metricsInterval) return;
        
        this.metricsInterval = setInterval(() => {
            this.updateLiveMetrics();
            this.updateQueueVisualization();
        }, 1000);
    }
    
    updateLiveMetrics() {
        // Update metric displays
        const elements = {
            'live-conversations': this.metrics.conversations,
            'live-topics': this.metrics.topics,
            'live-sentiment': this.metrics.sentiment.toFixed(1),
            'live-processing-rate': Math.round(this.metrics.processingRate),
            'messages-per-sec': Math.round(this.metrics.messagesPerSec),
            'total-received': this.metrics.totalReceived,
            'buffer-size': this.metrics.bufferSize,
            'processing-rate': `${Math.round(this.processingRate)}/sec`
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        });
    }
    
    updateQueueVisualization() {
        const maxQueue = 100;
        const stages = [
            { id: 'incoming', count: this.dataQueue.length, max: maxQueue },
            { id: 'processing', count: this.processingQueue.length, max: maxQueue },
            { id: 'completed', count: this.processedData.length, max: maxQueue }
        ];
        
        stages.forEach(stage => {
            const countEl = document.getElementById(`${stage.id}-count`);
            const fillEl = document.getElementById(`${stage.id}-fill`);
            
            if (countEl) countEl.textContent = stage.count;
            if (fillEl) {
                const percentage = Math.min((stage.count / stage.max) * 100, 100);
                fillEl.style.width = `${percentage}%`;
            }
        });
    }
    
    updateLiveVisualization(item) {
        // Simple animation for new data points
        if (this.animationsEnabled) {
            this.animateNewDataPoint(item);
        }
    }
    
    animateNewDataPoint(item) {
        const viz = document.getElementById('live-visualization');
        if (!viz) return;
        
        const dot = document.createElement('div');
        dot.className = 'data-point';
        dot.style.cssText = `
            position: absolute;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: ${this.getSentimentColor(item.sentiment)};
            left: ${Math.random() * 90}%;
            top: ${Math.random() * 90}%;
            animation: dataPointFade 2s ease-out forwards;
        `;
        
        viz.appendChild(dot);
        
        setTimeout(() => {
            if (dot.parentNode) dot.remove();
        }, 2000);
    }
    
    updateTopicsStream(item) {
        if (!this.topicUpdatesEnabled) return;
        
        const stream = document.getElementById('topics-stream');
        if (!stream) return;
        
        // Add new topic item
        const topicItem = document.createElement('div');
        topicItem.className = 'topic-item';
        topicItem.innerHTML = `
            <span class="topic-name">${item.topic}</span>
            <span class="topic-count">1</span>
        `;
        
        // Check if topic already exists
        const existingTopic = Array.from(stream.children)
            .find(child => child.querySelector('.topic-name')?.textContent === item.topic);
        
        if (existingTopic) {
            const countEl = existingTopic.querySelector('.topic-count');
            const currentCount = parseInt(countEl.textContent);
            countEl.textContent = currentCount + 1;
        } else {
            stream.insertBefore(topicItem, stream.firstChild);
            
            // Limit to 10 topics
            while (stream.children.length > 10) {
                stream.removeChild(stream.lastChild);
            }
        }
    }
    
    // Utility Functions
    getSentimentColor(sentiment) {
        if (sentiment > 0.3) return '#10b981';
        if (sentiment < -0.3) return '#ef4444';
        return '#f59e0b';
    }
    
    calculateMessagesPerSecond() {
        // Simple calculation based on recent activity
        const now = Date.now();
        const oneSecondAgo = now - 1000;
        
        const recentMessages = this.processedData.filter(
            item => item.timestamp > oneSecondAgo
        ).length;
        
        return recentMessages;
    }
    
    showLoadingOverlay(title, message) {
        const overlay = document.getElementById('loading-overlay');
        const titleEl = document.getElementById('loading-title');
        const messageEl = document.getElementById('loading-message');
        
        if (overlay) overlay.style.display = 'flex';
        if (titleEl) titleEl.textContent = title;
        if (messageEl) messageEl.textContent = message;
    }
    
    hideLoadingOverlay() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) overlay.style.display = 'none';
    }
    
    showDataStream() {
        const panel = document.getElementById('stream-panel');
        if (panel) panel.style.display = 'flex';
    }
    
    hideDataStream() {
        const panel = document.getElementById('stream-panel');
        if (panel) panel.style.display = 'none';
    }
    
    addStreamLog(type, message) {
        const log = document.getElementById('stream-log');
        if (!log) return;
        
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        entry.innerHTML = `
            <span class="log-timestamp">[${timestamp}]</span>
            ${message}
        `;
        
        log.appendChild(entry);
        
        // Auto-scroll if enabled
        if (this.autoScrollEnabled) {
            log.scrollTop = log.scrollHeight;
        }
        
        // Limit log entries
        while (log.children.length > 100) {
            log.removeChild(log.firstChild);
        }
    }
    
    showNotification(type, title, message) {
        const container = document.getElementById('notifications-container');
        if (!container) return;
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-icon">${this.getNotificationIcon(type)}</div>
                <div>
                    <strong>${title}</strong><br>
                    <span class="notification-message">${message}</span>
                </div>
            </div>
            <button class="notification-close" onclick="this.parentElement.remove()">×</button>
        `;
        
        container.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }
    
    getNotificationIcon(type) {
        const icons = {
            info: 'ℹ️',
            success: '✅',
            warning: '⚠️',
            error: '❌'
        };
        return icons[type] || 'ℹ️';
    }
    
    // Control Functions
    pauseAnalysis() {
        // Toggle processing
        this.processingRate = this.processingRate > 0 ? 0 : 1;
        
        const pauseBtn = document.getElementById('pause-btn');
        if (pauseBtn) {
            pauseBtn.textContent = this.processingRate > 0 ? '⏸️ Pause' : '▶️ Resume';
        }
        
        this.showNotification('info', 'Analysis Control', 
            `Processing ${this.processingRate > 0 ? 'resumed' : 'paused'}`);
    }
    
    clearAnalysis() {
        this.processedData = [];
        this.dataQueue = [];
        this.processingQueue = [];
        this.metrics = {
            conversations: 0,
            topics: 0,
            sentiment: 0.0,
            processingRate: 0,
            messagesPerSec: 0,
            totalReceived: 0,
            bufferSize: 0
        };
        
        // Clear UI
        const topicsStream = document.getElementById('topics-stream');
        if (topicsStream) topicsStream.innerHTML = '';
        
        const viz = document.getElementById('live-visualization');
        if (viz) viz.innerHTML = '';
        
        this.showNotification('info', 'Analysis Cleared', 'All data and metrics have been reset');
    }
    
    adjustProcessingSpeed(multiplier) {
        this.processingRate = multiplier;
        this.showNotification('info', 'Processing Speed', 
            `Processing speed set to ${multiplier === 0.5 ? 'slow' : multiplier === 1 ? 'normal' : 'fast'}`);
    }
    
    changeVisualizationType() {
        const vizType = document.getElementById('viz-type').value;
        const viz = document.getElementById('live-visualization');
        
        if (viz) {
            viz.innerHTML = `<p>Visualization: ${vizType}</p>`;
        }
        
        this.showNotification('info', 'Visualization Changed', `Switched to ${vizType} visualization`);
    }
    
    // Cleanup
    cleanup() {
        this.stopStream();
        
        if (this.metricsInterval) {
            clearInterval(this.metricsInterval);
        }
    }
    
    // Preferences
    loadPreferences() {
        const saved = localStorage.getItem('dynamic_loader_preferences');
        if (saved) {
            const prefs = JSON.parse(saved);
            this.animationsEnabled = prefs.animations ?? true;
            this.autoScrollEnabled = prefs.autoScroll ?? true;
            this.topicUpdatesEnabled = prefs.topicUpdates ?? true;
        }
    }
    
    savePreferences() {
        const prefs = {
            animations: this.animationsEnabled,
            autoScroll: this.autoScrollEnabled,
            topicUpdates: this.topicUpdatesEnabled
        };
        localStorage.setItem('dynamic_loader_preferences', JSON.stringify(prefs));
    }
}

// Global Functions
let dynamicLoader;

function setUploadMode(mode) {
    dynamicLoader.setUploadMode(mode);
}

function toggleRealTimeMode() {
    dynamicLoader.toggleRealTimeMode();
}

function updateStreamConfig() {
    dynamicLoader.updateStreamConfig();
}

function startStream() {
    dynamicLoader.startStream();
}

function stopStream() {
    dynamicLoader.stopStream();
}

function showDataStream() {
    dynamicLoader.showDataStream();
}

function hideDataStream() {
    dynamicLoader.hideDataStream();
}

function pauseAnalysis() {
    dynamicLoader.pauseAnalysis();
}

function clearAnalysis() {
    dynamicLoader.clearAnalysis();
}

function adjustProcessingSpeed(multiplier) {
    dynamicLoader.adjustProcessingSpeed(multiplier);
}

function changeVisualizationType() {
    dynamicLoader.changeVisualizationType();
}

function toggleVizAnimation() {
    dynamicLoader.animationsEnabled = !dynamicLoader.animationsEnabled;
    dynamicLoader.savePreferences();
}

function toggleTopicUpdates() {
    dynamicLoader.topicUpdatesEnabled = !dynamicLoader.topicUpdatesEnabled;
    dynamicLoader.savePreferences();
}

function clearStreamLog() {
    const log = document.getElementById('stream-log');
    if (log) log.innerHTML = '';
}

function exportStreamLog() {
    const log = document.getElementById('stream-log');
    if (log) {
        const content = log.textContent;
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `stream-log-${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }
}

function exportLiveResults() {
    const data = {
        metrics: dynamicLoader.metrics,
        processedData: dynamicLoader.processedData,
        timestamp: Date.now()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `live-analysis-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// API Integration Functions
function configureSlackIntegration() {
    dynamicLoader.showNotification('info', 'Slack Integration', 'Slack integration coming soon!');
}

function configureEmailIntegration() {
    dynamicLoader.showNotification('info', 'Email Integration', 'Email integration coming soon!');
}

function configureCustomAPI() {
    dynamicLoader.showNotification('info', 'Custom API', 'Custom API integration available in settings');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    dynamicLoader = new DynamicDataLoader();
    
    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes dataPointFade {
            0% { opacity: 1; transform: scale(0.8); }
            50% { opacity: 0.8; transform: scale(1.2); }
            100% { opacity: 0; transform: scale(0.5); }
        }
    `;
    document.head.appendChild(style);
});