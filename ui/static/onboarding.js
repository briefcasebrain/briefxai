// Onboarding JavaScript
class OnboardingManager {
    constructor() {
        this.currentStep = 1;
        this.totalSteps = 5;
        this.selectedDataOption = null;
        this.uploadedData = null;
        this.analysisConfig = {
            sentiment: true,
            clustering: true,
            patterns: true,
            piiDetection: true,
            localProcessing: true
        };
        this.analysisProgress = 0;
        this.analysisComplete = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.showStep(1);
        
        // Check if user has been here before
        const hasSeenOnboarding = localStorage.getItem('briefxai_onboarding_complete');
        if (hasSeenOnboarding && window.location.search.includes('skip-onboarding')) {
            this.completeOnboarding();
            return;
        }
        
        // Show onboarding overlay
        document.getElementById('onboarding-overlay').classList.add('active');
    }
    
    setupEventListeners() {
        // File upload handling
        const fileInput = document.getElementById('file-input');
        const uploadZone = document.getElementById('upload-zone');
        
        if (fileInput) {
            fileInput.addEventListener('change', this.handleFileUpload.bind(this));
        }
        
        if (uploadZone) {
            uploadZone.addEventListener('click', () => fileInput?.click());
            uploadZone.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadZone.addEventListener('drop', this.handleFileDrop.bind(this));
        }
        
        // Data option selection
        document.querySelectorAll('.data-option').forEach(option => {
            option.addEventListener('click', this.selectDataOption.bind(this));
        });
        
        // Configuration checkboxes
        document.querySelectorAll('.config-option input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', this.updateAnalysisConfig.bind(this));
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.skipOnboarding();
            } else if (e.key === 'Enter' && !e.shiftKey) {
                this.handleEnterKey();
            }
        });
    }
    
    showStep(stepNumber) {
        // Hide all steps
        document.querySelectorAll('.onboarding-step').forEach(step => {
            step.classList.remove('active');
        });
        
        // Show current step
        const currentStep = document.querySelector(`[data-step="${stepNumber}"]`);
        if (currentStep) {
            setTimeout(() => {
                currentStep.classList.add('active');
            }, 150);
        }
        
        this.currentStep = stepNumber;
        
        // Update step-specific logic
        switch(stepNumber) {
            case 2:
                this.updateDataContinueButton();
                break;
            case 4:
                this.startAnalysisSimulation();
                break;
            case 5:
                this.showResults();
                break;
        }
    }
    
    nextStep() {
        if (this.currentStep < this.totalSteps) {
            this.showStep(this.currentStep + 1);
        }
    }
    
    prevStep() {
        if (this.currentStep > 1) {
            this.showStep(this.currentStep - 1);
        }
    }
    
    skipOnboarding() {
        document.getElementById('onboarding-overlay').classList.remove('active');
        localStorage.setItem('briefxai_onboarding_complete', 'true');
        // Redirect to main app
        window.location.href = '/';
    }
    
    selectDataOption(event) {
        const option = event.currentTarget;
        const optionType = option.getAttribute('data-option');
        
        // Remove previous selection
        document.querySelectorAll('.data-option').forEach(opt => {
            opt.classList.remove('selected');
        });
        
        // Select current option
        option.classList.add('selected');
        this.selectedDataOption = optionType;
        
        this.updateDataContinueButton();
    }
    
    updateDataContinueButton() {
        const continueBtn = document.getElementById('data-continue-btn');
        if (continueBtn) {
            continueBtn.disabled = !this.selectedDataOption;
        }
    }
    
    loadExampleData() {
        this.selectedDataOption = 'example';
        this.selectDataOptionVisually('example');
        this.updateDataContinueButton();
        
        // Simulate loading example data
        setTimeout(() => {
            this.nextStep();
        }, 500);
    }
    
    selectDataOptionVisually(optionType) {
        document.querySelectorAll('.data-option').forEach(opt => {
            opt.classList.remove('selected');
        });
        
        const option = document.querySelector(`[data-option="${optionType}"]`);
        if (option) {
            option.classList.add('selected');
        }
    }
    
    handleFileUpload(event) {
        const files = event.target.files;
        if (files && files.length > 0) {
            this.processUploadedFile(files[0]);
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
            this.processUploadedFile(files[0]);
        }
    }
    
    async processUploadedFile(file) {
        if (!file.name.endsWith('.json')) {
            this.showError('Please upload a JSON file containing conversation data.');
            return;
        }
        
        try {
            const text = await file.text();
            const data = JSON.parse(text);
            
            // Basic validation
            if (!Array.isArray(data)) {
                throw new Error('Data should be an array of conversations');
            }
            
            this.uploadedData = data;
            this.selectedDataOption = 'upload';
            this.selectDataOptionVisually('upload');
            
            // Update upload zone to show success
            const uploadZone = document.getElementById('upload-zone');
            if (uploadZone) {
                uploadZone.innerHTML = `
                    <div class="upload-content">
                        <span class="upload-icon">✅</span>
                        <p><strong>${file.name}</strong> uploaded successfully!</p>
                        <p>${data.length} conversations ready for analysis</p>
                        <button class="btn-secondary" onclick="document.getElementById('file-input').click()">Change File</button>
                    </div>
                `;
            }
            
            this.updateDataContinueButton();
            
        } catch (error) {
            this.showError(`Error reading file: ${error.message}`);
        }
    }
    
    updateAnalysisConfig(event) {
        const checkbox = event.target;
        const configKey = this.getConfigKeyFromCheckbox(checkbox);
        
        if (configKey) {
            this.analysisConfig[configKey] = checkbox.checked;
        }
    }
    
    getConfigKeyFromCheckbox(checkbox) {
        const label = checkbox.closest('.config-option').querySelector('strong').textContent;
        const keyMap = {
            'Sentiment Analysis': 'sentiment',
            'Topic Clustering': 'clustering',
            'Pattern Discovery': 'patterns',
            'PII Detection & Masking': 'piiDetection',
            'Local Processing': 'localProcessing'
        };
        return keyMap[label];
    }
    
    startAnalysis() {
        this.nextStep(); // Go to progress step
    }
    
    startAnalysisSimulation() {
        this.analysisProgress = 0;
        const progressSteps = [
            { progress: 10, step: 'Validating data format...', insight: '📊 Found structured conversation data' },
            { progress: 25, step: 'Preprocessing conversations...', insight: '🔍 Extracting key features from conversations' },
            { progress: 45, step: 'Running sentiment analysis...', insight: '😊 Detecting emotional patterns' },
            { progress: 65, step: 'Clustering conversations...', insight: '🏷️ Grouping similar conversations' },
            { progress: 80, step: 'Discovering patterns...', insight: '🔄 Identifying recurring themes' },
            { progress: 95, step: 'Generating insights...', insight: '💡 Creating actionable recommendations' },
            { progress: 100, step: 'Analysis complete!', insight: '✨ Ready to explore your data' }
        ];
        
        let currentStepIndex = 0;
        
        const updateProgress = () => {
            if (currentStepIndex >= progressSteps.length) {
                this.analysisComplete = true;
                setTimeout(() => this.nextStep(), 1000);
                return;
            }
            
            const stepData = progressSteps[currentStepIndex];
            this.updateProgressUI(stepData.progress, stepData.step);
            this.addProgressInsight(stepData.insight);
            
            // Simulate realistic timing
            const nextDelay = currentStepIndex === 0 ? 500 : Math.random() * 1500 + 1000;
            setTimeout(() => {
                currentStepIndex++;
                updateProgress();
            }, nextDelay);
        };
        
        updateProgress();
    }
    
    updateProgressUI(progress, stepText) {
        // Update progress circle
        const progressFill = document.getElementById('progress-fill');
        const progressPercentage = document.getElementById('progress-percentage');
        const progressBarFill = document.getElementById('progress-bar-fill');
        const currentStep = document.getElementById('current-step');
        
        if (progressFill) {
            const rotation = -90 + (progress / 100) * 360;
            progressFill.style.transform = `rotate(${rotation}deg)`;
        }
        
        if (progressPercentage) {
            progressPercentage.textContent = `${progress}%`;
        }
        
        if (progressBarFill) {
            progressBarFill.style.width = `${progress}%`;
        }
        
        if (currentStep) {
            currentStep.textContent = stepText;
        }
        
        // Update processed count (simulate)
        const totalCount = this.selectedDataOption === 'upload' 
            ? this.uploadedData?.length || 1234 
            : 1234;
        const processedCount = Math.floor((progress / 100) * totalCount);
        
        const processedElement = document.getElementById('processed-count');
        const totalElement = document.getElementById('total-count');
        
        if (processedElement) processedElement.textContent = processedCount;
        if (totalElement) totalElement.textContent = totalCount;
        
        this.analysisProgress = progress;
    }
    
    addProgressInsight(insight) {
        const insightsList = document.getElementById('progress-insights');
        if (insightsList) {
            const insightElement = document.createElement('div');
            insightElement.className = 'insight-item';
            insightElement.textContent = insight;
            insightsList.appendChild(insightElement);
            
            // Keep only the last 3 insights visible
            const insights = insightsList.children;
            if (insights.length > 3) {
                insightsList.removeChild(insights[0]);
            }
            
            // Scroll to latest insight
            insightElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
    
    showResults() {
        // Generate realistic results based on selected data
        const totalConversations = this.selectedDataOption === 'upload' 
            ? this.uploadedData?.length || 1234 
            : 1234;
        
        const totalClusters = Math.max(3, Math.floor(totalConversations / 50));
        const totalPatterns = Math.floor(totalClusters / 3);
        
        // Update result numbers
        document.getElementById('total-conversations').textContent = totalConversations.toLocaleString();
        document.getElementById('total-clusters').textContent = totalClusters;
        document.getElementById('total-patterns').textContent = totalPatterns;
        
        // Store results for the main app
        const results = {
            totalConversations,
            totalClusters,
            totalPatterns,
            dataSource: this.selectedDataOption,
            analysisConfig: this.analysisConfig,
            timestamp: new Date().toISOString()
        };
        
        localStorage.setItem('briefxai_analysis_results', JSON.stringify(results));
        
        if (this.uploadedData) {
            localStorage.setItem('briefxai_uploaded_data', JSON.stringify(this.uploadedData));
        }
    }
    
    completeOnboarding() {
        localStorage.setItem('briefxai_onboarding_complete', 'true');
        
        // Hide onboarding overlay
        document.getElementById('onboarding-overlay').classList.remove('active');
        
        // Redirect to main application with onboarding completion flag
        const url = new URL(window.location);
        url.search = 'onboarding-complete=true';
        window.location.href = url.toString();
    }
    
    goToView(view) {
        // Store the desired initial view
        localStorage.setItem('briefxai_initial_view', view);
        this.completeOnboarding();
    }
    
    showError(message) {
        // Create error toast notification
        const toast = document.createElement('div');
        toast.className = 'error-toast';
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">⚠️</span>
                <span class="toast-message">${message}</span>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        // Add error toast styles if not already added
        if (!document.querySelector('#error-toast-styles')) {
            const styles = document.createElement('style');
            styles.id = 'error-toast-styles';
            styles.textContent = `
                .error-toast {
                    position: fixed;
                    top: 2rem;
                    right: 2rem;
                    background: #fee2e2;
                    border: 1px solid #fecaca;
                    border-radius: 8px;
                    z-index: 10001;
                    animation: slideInRight 0.3s ease;
                }
                .toast-content {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    padding: 1rem 1.5rem;
                }
                .toast-icon {
                    font-size: 1.2rem;
                }
                .toast-message {
                    color: #dc2626;
                    font-weight: 500;
                }
                .toast-close {
                    background: none;
                    border: none;
                    color: #dc2626;
                    cursor: pointer;
                    font-size: 1.2rem;
                    padding: 0.25rem;
                    border-radius: 4px;
                }
                .toast-close:hover {
                    background: rgba(220, 38, 38, 0.1);
                }
                @keyframes slideInRight {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;
            document.head.appendChild(styles);
        }
        
        document.body.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
    }
    
    handleEnterKey() {
        // Handle Enter key in different steps
        switch(this.currentStep) {
            case 1:
                this.nextStep();
                break;
            case 2:
                if (!document.getElementById('data-continue-btn').disabled) {
                    this.nextStep();
                }
                break;
            case 3:
                this.startAnalysis();
                break;
            case 5:
                this.completeOnboarding();
                break;
        }
    }
}

// Global functions for HTML onclick handlers
function nextOnboardingStep() {
    if (window.onboardingManager) {
        window.onboardingManager.nextStep();
    }
}

function prevOnboardingStep() {
    if (window.onboardingManager) {
        window.onboardingManager.prevStep();
    }
}

function skipOnboarding() {
    if (window.onboardingManager) {
        window.onboardingManager.skipOnboarding();
    }
}

function loadExampleData() {
    if (window.onboardingManager) {
        window.onboardingManager.loadExampleData();
    }
}

function startAnalysis() {
    if (window.onboardingManager) {
        window.onboardingManager.startAnalysis();
    }
}

function completeOnboarding() {
    if (window.onboardingManager) {
        window.onboardingManager.completeOnboarding();
    }
}

function goToView(view) {
    if (window.onboardingManager) {
        window.onboardingManager.goToView(view);
    }
}

function closeQuickTour() {
    const quickTour = document.getElementById('quick-tour');
    if (quickTour) {
        quickTour.style.display = 'none';
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.onboardingManager = new OnboardingManager();
});

// Quick Tour for returning users
function showQuickTour() {
    const hasSeenOnboarding = localStorage.getItem('briefxai_onboarding_complete');
    if (hasSeenOnboarding && !sessionStorage.getItem('quick_tour_shown')) {
        const quickTour = document.getElementById('quick-tour');
        if (quickTour) {
            // Position near upload button or main action
            const uploadBtn = document.querySelector('.btn-primary');
            if (uploadBtn) {
                const rect = uploadBtn.getBoundingClientRect();
                quickTour.style.top = (rect.top - 100) + 'px';
                quickTour.style.left = rect.left + 'px';
                quickTour.style.display = 'block';
                
                sessionStorage.setItem('quick_tour_shown', 'true');
                
                // Auto-hide after 10 seconds
                setTimeout(() => {
                    quickTour.style.display = 'none';
                }, 10000);
            }
        }
    }
}