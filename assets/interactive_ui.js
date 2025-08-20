// Global state
let uploadedData = null;
let analysisResults = null;
let currentScreen = 'welcome';

// Initialize dropdowns on page load - moved to end of file or called after functions are defined

// Screen navigation
function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(screen => {
        screen.classList.remove('active');
    });
    document.getElementById(screenId + '-screen').classList.add('active');
    currentScreen = screenId;
}

function showWelcomeScreen() {
    showScreen('welcome');
}

function showUploadScreen() {
    showScreen('upload');
    setupDropZone();
}

function showConfigScreen() {
    showScreen('config');
    initializeConfigDefaults();
}

function showProgressScreen() {
    showScreen('progress');
}

function showResultsScreen() {
    showScreen('results');
}

// Upload functionality is defined later in the file with enhanced features

async function handleFiles(files) {
    const fileList = document.getElementById('file-list');
    fileList.innerHTML = '';
    
    // Create FormData for multipart upload
    const formData = new FormData();
    let fileCount = 0;
    
    Array.from(files).forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span>${file.name} (${formatFileSize(file.size)})</span>
            <span class="upload-status" id="status-${index}">⏳ Uploading...</span>
            <span class="remove" onclick="removeFile(this)">✕</span>
        `;
        fileList.appendChild(fileItem);
        
        // Add file to FormData
        formData.append(`file${fileCount}`, file);
        fileCount++;
    });
    
    if (fileCount === 0) {
        return;
    }
    
    // Upload files to server
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Update UI with upload results
            uploadedData = result.data.conversations;
            
            // Update file statuses
            result.data.files.forEach((fileInfo, index) => {
                const statusElement = document.getElementById(`status-${index}`);
                if (statusElement) {
                    statusElement.innerHTML = `✅ ${fileInfo.conversations} conversations`;
                    statusElement.style.color = '#4caf50';
                }
            });
            
            // Show warnings if any
            if (result.data.warnings && result.data.warnings.length > 0) {
                const warningMessage = result.data.warnings.join('\n');
                console.warn('Upload warnings:', warningMessage);
                addLogEntry(`Warnings: ${warningMessage}`, 'warning');
            }
            
            // Enable analyze button
            document.getElementById('analyze-btn').disabled = false;
            
            // Show success message
            const successMsg = `Successfully uploaded ${result.data.total_conversations} conversations from ${result.data.files.length} file(s)`;
            addLogEntry(successMsg, 'success');
            
        } else {
            throw new Error(result.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload failed:', error);
        alert(`Upload failed: ${error.message}`);
        
        // Update file statuses to show error
        Array.from(files).forEach((file, index) => {
            const statusElement = document.getElementById(`status-${index}`);
            if (statusElement) {
                statusElement.innerHTML = '❌ Failed';
                statusElement.style.color = '#f44336';
            }
        });
    }
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function removeFile(element) {
    element.parentElement.remove();
    if (document.querySelectorAll('.file-item').length === 0) {
        document.getElementById('analyze-btn').disabled = true;
        uploadedData = null;
    }
}

// Example data generation
async function useExampleData() {
    uploadedData = generateExampleData(100);
    showConfigScreen();
}

function generateExampleData(count) {
    const conversations = [];
    const topics = ['weather', 'coding', 'travel', 'cooking', 'science', 'history', 'sports'];
    const languages = ['English', 'Spanish', 'French', 'German', 'Chinese'];
    
    for (let i = 0; i < count; i++) {
        const topic = topics[Math.floor(Math.random() * topics.length)];
        const conversation = [
            {
                role: "user",
                content: `Can you help me with ${topic}? (Conversation ${i + 1})`
            },
            {
                role: "assistant",
                content: `Of course! I'd be happy to help you with ${topic}. What specific aspect would you like to know about?`
            },
            {
                role: "user",
                content: `I'm particularly interested in recent developments.`
            },
            {
                role: "assistant",
                content: `Here are some recent developments in ${topic}...`
            }
        ];
        conversations.push(conversation);
    }
    
    return conversations;
}

// Paste data functionality
function pasteData() {
    document.getElementById('paste-modal').style.display = 'block';
}

function closePasteModal() {
    document.getElementById('paste-modal').style.display = 'none';
}

async function processPastedData() {
    const pasteArea = document.getElementById('paste-area');
    const pastedText = pasteArea.value.trim();
    
    if (!pastedText) {
        alert('Please paste some data first');
        return;
    }
    
    try {
        // First try to parse as JSON locally
        let data;
        try {
            data = JSON.parse(pastedText);
        } catch (e) {
            // If not JSON, send to server for parsing
            data = pastedText;
        }
        
        // Create a file-like blob from the pasted data
        const blob = new Blob([pastedText], { type: 'application/json' });
        const file = new File([blob], 'pasted_data.json', { type: 'application/json' });
        
        // Use the existing upload functionality
        closePasteModal();
        await handleFiles([file]);
        
        if (uploadedData && uploadedData.length > 0) {
            showConfigScreen();
        }
    } catch (error) {
        console.error('Failed to process pasted data:', error);
        alert(`Failed to process pasted data: ${error.message}`);
    }
}

// Enhanced drag and drop with visual feedback
function setupDropZone() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);
    
    // Handle file input
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight(e) {
        dropZone.classList.add('active');
        dropZone.style.borderColor = '#667eea';
        dropZone.style.backgroundColor = '#f0f4ff';
    }
    
    function unhighlight(e) {
        dropZone.classList.remove('active');
        dropZone.style.borderColor = '';
        dropZone.style.backgroundColor = '';
    }
    
    async function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        await handleFiles(files);
    }
}

// Configuration
function updateLLMOptions() {
    const provider = document.getElementById('llm-provider').value;
    const apiKeySection = document.getElementById('api-key-section');
    const baseUrlSection = document.getElementById('base-url-section');
    const modelSelect = document.getElementById('llm-model');
    
    // Clear current options
    modelSelect.innerHTML = '';
    
    switch(provider) {
        case 'openai':
            apiKeySection.style.display = 'block';
            baseUrlSection.style.display = 'none';
            // Add OpenAI models
            addModelOptions(modelSelect, [
                { value: 'gpt-4o-mini', text: 'GPT-4o Mini (Recommended)' },
                { value: 'gpt-4o', text: 'GPT-4o' },
                { value: 'gpt-4-turbo', text: 'GPT-4 Turbo' },
                { value: 'gpt-3.5-turbo', text: 'GPT-3.5 Turbo' }
            ]);
            break;
            
        case 'ollama':
            apiKeySection.style.display = 'none';
            baseUrlSection.style.display = 'block';
            document.getElementById('base-url').value = 'http://localhost:11434';
            // Add Ollama models
            addModelOptions(modelSelect, [
                { value: 'llama3.2:latest', text: 'Llama 3.2' },
                { value: 'qwen2.5:7b', text: 'Qwen 2.5 7B' },
                { value: 'mistral:latest', text: 'Mistral' },
                { value: 'phi3:latest', text: 'Phi 3' },
                { value: 'gemma2:latest', text: 'Gemma 2' }
            ]);
            // Check if Ollama is running and show status
            checkOllamaStatus();
            break;
            
        case 'vllm':
            apiKeySection.style.display = 'none';
            baseUrlSection.style.display = 'block';
            document.getElementById('base-url').value = 'http://localhost:8000';
            // Add vLLM models
            addModelOptions(modelSelect, [
                { value: 'meta-llama/Llama-3.2-3B-Instruct', text: 'Llama 3.2 3B' },
                { value: 'microsoft/Phi-3.5-mini-instruct', text: 'Phi 3.5 Mini' },
                { value: 'custom', text: 'Custom Model (enter name)' }
            ]);
            // Check if vLLM is running
            checkLocalServer('vllm');
            break;
            
        case 'huggingface':
            apiKeySection.style.display = 'block';
            baseUrlSection.style.display = 'none';
            // Add HuggingFace models
            addModelOptions(modelSelect, [
                { value: 'meta-llama/Llama-3.2-3B-Instruct', text: 'Llama 3.2 3B' },
                { value: 'mistralai/Mistral-7B-Instruct-v0.3', text: 'Mistral 7B' },
                { value: 'google/flan-t5-base', text: 'Flan-T5 Base' }
            ]);
            break;
    }
}

function addModelOptions(select, options) {
    options.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt.value;
        option.textContent = opt.text;
        select.appendChild(option);
    });
}

function onModelChange() {
    const provider = document.getElementById('llm-provider').value;
    if (provider === 'ollama') {
        checkOllamaStatus();
    }
}

function updateEmbeddingOptions() {
    const provider = document.getElementById('embedding-provider').value;
    const modelSelect = document.getElementById('embedding-model');
    const apiKeySection = document.getElementById('embedding-api-key-section');
    
    // Clear current options
    modelSelect.innerHTML = '';
    
    switch(provider) {
        case 'openai':
            apiKeySection.style.display = 'block';
            addModelOptions(modelSelect, [
                { value: 'text-embedding-3-small', text: 'text-embedding-3-small (Recommended)' },
                { value: 'text-embedding-3-large', text: 'text-embedding-3-large' },
                { value: 'text-embedding-ada-002', text: 'text-embedding-ada-002' }
            ]);
            break;
            
        case 'sentence-transformers':
            apiKeySection.style.display = 'none';
            addModelOptions(modelSelect, [
                { value: 'all-mpnet-base-v2', text: 'all-mpnet-base-v2 (Recommended)' },
                { value: 'all-MiniLM-L6-v2', text: 'all-MiniLM-L6-v2 (Fast)' },
                { value: 'all-roberta-large-v1', text: 'all-roberta-large-v1 (High Quality)' }
            ]);
            break;
            
        case 'ollama':
            apiKeySection.style.display = 'none';
            addModelOptions(modelSelect, [
                { value: 'nomic-embed-text', text: 'Nomic Embed Text' },
                { value: 'mxbai-embed-large', text: 'MxBAI Embed Large' },
                { value: 'all-minilm', text: 'All MiniLM' }
            ]);
            break;
            
        case 'huggingface':
            apiKeySection.style.display = 'block';
            addModelOptions(modelSelect, [
                { value: 'sentence-transformers/all-mpnet-base-v2', text: 'All MPNet Base v2' },
                { value: 'BAAI/bge-large-en-v1.5', text: 'BGE Large EN v1.5' },
                { value: 'intfloat/e5-large-v2', text: 'E5 Large v2' }
            ]);
            break;
    }
}

async function checkOllamaStatus() {
    const statusDiv = document.createElement('div');
    statusDiv.id = 'ollama-status';
    statusDiv.style.marginTop = '15px';
    statusDiv.style.padding = '15px';
    statusDiv.style.background = '#f8f9fa';
    statusDiv.style.borderRadius = '8px';
    
    // Remove existing status if any
    const existing = document.getElementById('ollama-status');
    if (existing) existing.remove();
    
    const modelSelect = document.getElementById('llm-model');
    modelSelect.parentElement.appendChild(statusDiv);
    
    statusDiv.innerHTML = '<span style="color: blue;">🔄 Checking Ollama status...</span>';
    
    try {
        const response = await fetch('/api/ollama-status');
        const data = await response.json();
        
        if (data.success && data.data) {
            const status = data.data;
            
            if (status.installed === false) {
                statusDiv.innerHTML = `
                    <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 12px;">
                        <strong style="color: #856404;">⚠️ Ollama Not Installed</strong>
                        <div style="margin-top: 10px; font-size: 13px;">
                            <a href="https://ollama.ai" target="_blank" style="color: #007bff;">Download Ollama</a> to use local models
                        </div>
                    </div>
                `;
            } else if (!status.running) {
                statusDiv.innerHTML = `
                    <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 12px;">
                        <strong style="color: #721c24;">🔴 Ollama Server Not Running</strong>
                        <div style="margin-top: 10px;">
                            <button onclick="startOllamaServer()" class="btn-small" style="background: #28a745; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer;">
                                Start Ollama Server
                            </button>
                        </div>
                    </div>
                `;
            } else {
                // Ollama is running, show available models
                const availableModels = status.models || [];
                const selectedModel = modelSelect.value;
                const modelInstalled = availableModels.some(m => m.startsWith(selectedModel.replace(':latest', '')));
                
                if (modelInstalled) {
                    statusDiv.innerHTML = `
                        <div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 6px; padding: 12px;">
                            <strong style="color: #155724;">✅ Ollama Ready</strong>
                            <div style="margin-top: 8px; font-size: 13px; color: #155724;">
                                Server running • Model ${selectedModel} available
                            </div>
                        </div>
                    `;
                } else {
                    statusDiv.innerHTML = `
                        <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 12px;">
                            <strong style="color: #856404;">⚠️ Model Not Available</strong>
                            <div style="margin-top: 10px;">
                                <button onclick="pullOllamaModel('${selectedModel}')" class="btn-small" style="background: #007bff; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer;">
                                    Pull ${selectedModel}
                                </button>
                                <div style="margin-top: 8px; font-size: 12px; color: #666;">
                                    Available models: ${availableModels.join(', ') || 'none'}
                                </div>
                            </div>
                        </div>
                    `;
                }
            }
        }
    } catch (error) {
        statusDiv.innerHTML = `
            <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 12px;">
                <strong style="color: #721c24;">Error checking Ollama status</strong>
            </div>
        `;
    }
}

async function startOllamaServer() {
    const statusDiv = document.getElementById('ollama-status');
    statusDiv.innerHTML = '<span style="color: blue;">🔄 Starting Ollama server...</span>';
    
    try {
        const response = await fetch('/api/start-server', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ serverType: 'ollama' })
        });
        
        const data = await response.json();
        
        if (data.success) {
            setTimeout(() => checkOllamaStatus(), 2000);
        } else {
            statusDiv.innerHTML = `
                <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 12px;">
                    <strong style="color: #721c24;">Failed to start Ollama</strong>
                    <div style="margin-top: 8px; font-size: 12px;">${data.error}</div>
                </div>
            `;
        }
    } catch (error) {
        statusDiv.innerHTML = `
            <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 12px;">
                <strong style="color: #721c24;">Error starting Ollama</strong>
            </div>
        `;
    }
}

async function pullOllamaModel(model) {
    const statusDiv = document.getElementById('ollama-status');
    statusDiv.innerHTML = `
        <div style="background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 6px; padding: 12px;">
            <strong style="color: #0c5460;">📥 Pulling model ${model}...</strong>
            <div style="margin-top: 8px; font-size: 12px;">This may take a few minutes depending on model size</div>
        </div>
    `;
    
    try {
        const response = await fetch('/api/pull-model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model })
        });
        
        const data = await response.json();
        
        if (data.success) {
            statusDiv.innerHTML = `
                <div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 6px; padding: 12px;">
                    <strong style="color: #155724;">✅ Model pulled successfully!</strong>
                </div>
            `;
            setTimeout(() => checkOllamaStatus(), 1000);
        } else {
            statusDiv.innerHTML = `
                <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 12px;">
                    <strong style="color: #721c24;">Failed to pull model</strong>
                    <div style="margin-top: 8px; font-size: 12px;">${data.error}</div>
                </div>
            `;
        }
    } catch (error) {
        statusDiv.innerHTML = `
            <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 12px;">
                <strong style="color: #721c24;">Error pulling model</strong>
            </div>
        `;
    }
}

async function checkLocalServer(serverType) {
    const baseUrlInput = document.getElementById('base-url');
    const statusDiv = document.createElement('div');
    statusDiv.id = 'server-status';
    statusDiv.style.marginTop = '10px';
    
    // Remove existing status if any
    const existing = document.getElementById('server-status');
    if (existing) existing.remove();
    
    baseUrlInput.parentElement.appendChild(statusDiv);
    
    const baseUrl = baseUrlInput.value;
    
    try {
        // Try to connect to the server
        const response = await fetch(`${baseUrl}/health`, { 
            method: 'GET',
            mode: 'no-cors' // Allow checking without CORS
        }).catch(() => null);
        
        if (response || serverType === 'ollama') {
            // For Ollama, also try the specific endpoint
            if (serverType === 'ollama') {
                try {
                    await fetch(`${baseUrl}/api/tags`);
                    statusDiv.innerHTML = '<span style="color: green;">✓ Ollama server is running</span>';
                } catch {
                    statusDiv.innerHTML = `
                        <span style="color: orange;">⚠ Ollama server not detected</span>
                        <button onclick="startLocalServer('ollama')" class="btn-small">Start Ollama</button>
                    `;
                }
            } else {
                statusDiv.innerHTML = '<span style="color: green;">✓ Server is running</span>';
            }
        } else {
            throw new Error('Server not responding');
        }
    } catch (error) {
        statusDiv.innerHTML = `
            <span style="color: orange;">⚠ ${serverType} server not detected</span>
            <button onclick="startLocalServer('${serverType}')" class="btn-small">Start ${serverType}</button>
        `;
    }
}

async function startLocalServer(serverType) {
    const statusDiv = document.getElementById('server-status');
    statusDiv.innerHTML = '<span style="color: blue;">🔄 Checking server setup...</span>';
    
    try {
        const response = await fetch('/api/start-server', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ serverType })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            statusDiv.innerHTML = '<span style="color: green;">✓ Server started successfully</span>';
            // Re-check after a moment
            setTimeout(() => checkLocalServer(serverType), 2000);
        } else {
            const errorMsg = data.error || 'Failed to start server';
            
            // Format the error message based on the server type
            if (errorMsg.includes('vLLM is not installed')) {
                // Parse and format the vLLM message nicely
                const formattedMsg = errorMsg
                    .replace(/📝/g, '<span style="font-size: 1.2em;">📝</span>')
                    .replace(/🔧/g, '<span style="font-size: 1.2em;">🔧</span>')
                    .replace(/💡/g, '<span style="font-size: 1.2em;">💡</span>')
                    .replace(/✓/g, '<span style="color: green;">✓</span>')
                    .replace(/•/g, '&nbsp;&nbsp;•')
                    .replace(/\n/g, '<br>');
                
                statusDiv.innerHTML = `
                    <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 15px; margin-top: 10px;">
                        <div style="color: #856404; font-size: 13px; line-height: 1.6; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                            ${formattedMsg}
                        </div>
                    </div>
                `;
            } else if (errorMsg.includes('Ollama is not installed')) {
                statusDiv.innerHTML = `
                    <div style="background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 8px; padding: 15px; margin-top: 10px;">
                        <strong style="color: #0c5460; font-size: 14px;">ℹ️ Ollama Not Installed</strong>
                        <div style="color: #0c5460; margin-top: 10px; font-size: 13px; line-height: 1.6;">
                            Ollama is a user-friendly tool for running LLMs locally on your computer.<br><br>
                            <strong>Quick Setup:</strong><br>
                            1. Visit <a href="https://ollama.ai" target="_blank" style="color: #007bff; text-decoration: none;">ollama.ai</a> and download the installer<br>
                            2. Run the installer (takes ~2 minutes)<br>
                            3. Open terminal and run: <code style="background: #f0f0f0; padding: 2px 4px; border-radius: 3px;">ollama serve</code><br>
                            4. Pull a model: <code style="background: #f0f0f0; padding: 2px 4px; border-radius: 3px;">ollama pull llama3.2</code><br><br>
                            <em>💡 Tip: Or simply use OpenAI API (selected by default) - no setup needed!</em>
                        </div>
                    </div>
                `;
            } else {
                statusDiv.innerHTML = `
                    <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; padding: 15px; margin-top: 10px;">
                        <span style="color: #721c24;">✗ ${errorMsg}</span>
                        <div style="font-size: 12px; margin-top: 10px; color: #721c24;">
                            ${getServerInstructions(serverType)}
                        </div>
                    </div>
                `;
            }
        }
    } catch (error) {
        statusDiv.innerHTML = `
            <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; padding: 15px; margin-top: 10px;">
                <span style="color: #721c24;">✗ Network error: Could not reach server</span>
            </div>
        `;
    }
}

function getServerInstructions(serverType) {
    switch(serverType) {
        case 'ollama':
            return `
                <code style="display: block; margin: 5px 0;">
                    # Install Ollama<br>
                    curl -fsSL https://ollama.ai/install.sh | sh<br><br>
                    # Start server<br>
                    ollama serve<br><br>
                    # Pull a model (in another terminal)<br>
                    ollama pull llama3.2
                </code>
            `;
        case 'vllm':
            return `
                <code style="display: block; margin: 5px 0;">
                    # Install vLLM<br>
                    pip install vllm<br><br>
                    # Start with a small model<br>
                    python -m vllm.entrypoints.openai.api_server \\<br>
                    &nbsp;&nbsp;--model microsoft/Phi-3.5-mini-instruct \\<br>
                    &nbsp;&nbsp;--host 0.0.0.0 --port 8000 \\<br>
                    &nbsp;&nbsp;--max-model-len 2048<br><br>
                    # Or with a larger model<br>
                    python -m vllm.entrypoints.openai.api_server \\<br>
                    &nbsp;&nbsp;--model meta-llama/Llama-3.2-3B-Instruct \\<br>
                    &nbsp;&nbsp;--host 0.0.0.0 --port 8000
                </code>
            `;
        default:
            return 'Please refer to the server documentation for setup instructions.';
    }
}

function startAnalysis() {
    showConfigScreen();
}

// WebSocket connection for real-time progress
let progressWebSocket = null;
let currentSessionId = null;

// Analysis execution
async function checkOllamaReady(model) {
    try {
        const response = await fetch('/api/ollama-status');
        const data = await response.json();
        
        if (data.success && data.data) {
            const status = data.data;
            
            if (status.installed === false) {
                return { ready: false, message: 'Ollama is not installed' };
            } else if (!status.running) {
                return { ready: false, message: 'Ollama server is not running' };
            } else {
                const availableModels = status.models || [];
                const modelInstalled = availableModels.some(m => 
                    m.startsWith(model.replace(':latest', ''))
                );
                
                if (!modelInstalled) {
                    return { 
                        ready: false, 
                        message: `Model ${model} is not available. Available models: ${availableModels.join(', ') || 'none'}` 
                    };
                }
                
                return { ready: true };
            }
        }
        
        return { ready: false, message: 'Could not check Ollama status' };
    } catch (error) {
        return { ready: false, message: 'Error checking Ollama status' };
    }
}

async function runAnalysis() {
    const provider = document.getElementById('llm-provider').value;
    const model = document.getElementById('llm-model').value;
    
    // Pre-flight check for Ollama
    if (provider === 'ollama') {
        const ollamaCheck = await checkOllamaReady(model);
        if (!ollamaCheck.ready) {
            const useOpenAI = confirm(
                `❌ Ollama is not ready:\n\n${ollamaCheck.message}\n\n` +
                `Would you like to:\n` +
                `• Click OK to switch to OpenAI (recommended)\n` +
                `• Click Cancel to go back and fix Ollama setup`
            );
            
            if (useOpenAI) {
                // Switch to OpenAI
                document.getElementById('llm-provider').value = 'openai';
                updateLLMOptions();
                
                // Check if API key is provided
                const apiKey = document.getElementById('api-key').value;
                if (!apiKey) {
                    const key = prompt('Please enter your OpenAI API key:');
                    if (!key) {
                        alert('API key is required for OpenAI. Analysis cancelled.');
                        showConfigScreen();
                        return;
                    }
                    document.getElementById('api-key').value = key;
                }
                
                // Update provider for the analysis
                document.getElementById('llm-provider').value = 'openai';
                document.getElementById('llm-model').value = 'gpt-4o-mini';
            } else {
                // User wants to fix Ollama - stay on config screen
                return;
            }
        }
    }
    
    showProgressScreen();
    
    const config = {
        llm_provider: document.getElementById('llm-provider').value,
        llm_model: document.getElementById('llm-model').value,
        embedding_provider: document.getElementById('embedding-provider').value,
        embedding_model: document.getElementById('embedding-model').value,
        dedup: document.getElementById('dedup').checked,
        verbose: document.getElementById('verbose').checked,
        batch_size: parseInt(document.getElementById('batch-size').value),
        api_key: document.getElementById('api-key')?.value || '',
        embedding_api_key: document.getElementById('embedding-api-key')?.value || '',
        base_url: document.getElementById('base-url')?.value || ''
    };
    
    // Connect to WebSocket for progress updates
    await connectProgressWebSocket();
    
    try {
        // Start real analysis
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                data: uploadedData,
                config: config
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentSessionId = result.data.session_id;
            analysisResults = result.data.results;
            
            // Wait a moment for final progress updates
            await sleep(1000);
            
            showResultsScreen();
            renderResults();
        } else {
            throw new Error(result.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Analysis failed:', error);
        addLogEntry(`Error: ${error.message}`, 'error');
        
        // Re-enable analysis button on error
        const analyzeBtn = document.querySelector('button[onclick="runAnalysis()"]');
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Conversations';
        }
    } finally {
        // Close WebSocket connection
        if (progressWebSocket) {
            progressWebSocket.close();
            progressWebSocket = null;
        }
    }
}

async function connectProgressWebSocket() {
    return new Promise((resolve, reject) => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/progress`;
        
        progressWebSocket = new WebSocket(wsUrl);
        
        progressWebSocket.onopen = () => {
            console.log('Progress WebSocket connected');
            resolve();
        };
        
        progressWebSocket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                handleProgressMessage(message);
            } catch (error) {
                console.error('Failed to parse progress message:', error);
            }
        };
        
        progressWebSocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            reject(error);
        };
        
        progressWebSocket.onclose = () => {
            console.log('Progress WebSocket disconnected');
        };
        
        // Timeout after 5 seconds
        setTimeout(() => {
            if (progressWebSocket.readyState !== WebSocket.OPEN) {
                reject(new Error('WebSocket connection timeout'));
            }
        }, 5000);
    });
}

function handleProgressMessage(message) {
    console.log('Progress update:', message);
    
    switch (message.type || message.Update ? 'Update' : message.Started ? 'Started' : message.Completed ? 'Completed' : message.Error ? 'Error' : 'Unknown') {
        case 'Started':
            addLogEntry(`Analysis session ${message.session_id} started`, 'info');
            break;
            
        case 'Update':
            if (message.Update) {
                const update = message.Update;
                updateProgressDisplay(update.step, update.progress, update.message, update.details);
                addLogEntry(`${update.step}: ${update.message}`, 'info');
            } else {
                updateProgressDisplay(message.step, message.progress, message.message, message.details);
                addLogEntry(`${message.step}: ${message.message}`, 'info');
            }
            break;
            
        case 'Completed':
            addLogEntry(`Analysis completed: ${message.result}`, 'success');
            break;
            
        case 'Error':
            addLogEntry(`Analysis failed: ${message.error}`, 'error');
            break;
            
        default:
            console.log('Unknown message type:', message);
    }
}

// Updated progress display function for real-time updates
function updateProgressDisplay(step, percent, message, details) {
    // Enhanced mapping for new steps
    const stepMapping = {
        'validation': 'validation',
        'dedup': 'dedup',
        'facets': 'facets',
        'embeddings': 'embeddings',
        'clustering': 'clustering',
        'hierarchy': 'hierarchy',
        'umap': 'umap',
        'finalize': 'finalize',
        'complete': 'finalize',
        'connected': 'validation'
    };
    
    const uiStep = stepMapping[step] || step;
    
    // Update current stage display
    const stageTitle = document.getElementById('current-stage-title');
    const stageDetail = document.getElementById('current-stage-detail');
    if (stageTitle) stageTitle.textContent = message || 'Processing...';
    if (stageDetail) stageDetail.textContent = details || '';
    
    // Update percentage
    const percentageEl = document.getElementById('progress-percentage');
    if (percentageEl) {
        percentageEl.textContent = `${Math.round(percent)}%`;
    }
    
    // Update step status
    const stepElement = document.getElementById(`step-${uiStep}`);
    if (stepElement) {
        stepElement.classList.add('active');
        
        // Update step detail text
        const stepDetail = stepElement.querySelector('.step-detail');
        if (stepDetail && details) {
            stepDetail.textContent = details;
        }
        
        const allSteps = ['validation', 'dedup', 'facets', 'embeddings', 'clustering', 'hierarchy', 'umap', 'finalize'];
        const currentIndex = allSteps.indexOf(uiStep);
        
        // Mark previous steps as completed
        for (let i = 0; i < currentIndex; i++) {
            const prevStep = document.getElementById(`step-${allSteps[i]}`);
            if (prevStep) {
                prevStep.classList.add('completed');
                prevStep.classList.remove('active');
                const statusEl = prevStep.querySelector('.status');
                if (statusEl) statusEl.textContent = '✓';
                
                // Keep the detail text for completed steps
                const prevDetail = prevStep.querySelector('.step-detail');
                if (prevDetail && prevDetail.textContent.startsWith('✓')) {
                    // Keep the success message
                } else if (prevDetail && !prevDetail.textContent.startsWith('✓')) {
                    prevDetail.textContent = '✓ Completed';
                }
            }
        }
        
        // Update current step
        if (percent >= 100 && step === 'complete') {
            stepElement.classList.add('completed');
            stepElement.classList.remove('active');
            const statusEl = stepElement.querySelector('.status');
            if (statusEl) statusEl.textContent = '✓';
        }
    }
    
    // Update progress bar
    const progressFill = document.getElementById('progress-fill');
    if (progressFill) {
        progressFill.style.width = Math.min(percent, 100) + '%';
    }
    
    // Add detailed log entry
    addLogEntry(`${message}${details ? ` (${details})` : ''}`, 'info');
}

// Enhanced logging function
function addLogEntry(message, type = 'info') {
    const log = document.getElementById('progress-log');
    if (!log) return;
    
    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;
    
    const timestamp = new Date().toLocaleTimeString();
    entry.innerHTML = `
        <span class="log-time">[${timestamp}]</span>
        <span class="log-message">${message}</span>
    `;
    
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
    
    // Limit log entries to prevent memory issues
    while (log.children.length > 100) {
        log.removeChild(log.firstChild);
    }
}

// Legacy function for backward compatibility
async function updateProgress(step, percent) {
    updateProgressDisplay(step, percent, `Processing ${step}...`, null);
}

async function performAnalysis(data, config) {
    // This would be replaced with actual API call to backend
    // For now, return mock results
    return {
        conversations: data,
        clusters: generateMockClusters(),
        facets: generateMockFacets(),
        umap: generateMockUMAP(data.length),
        stats: {
            total_conversations: data.length,
            total_clusters: 12,
            languages: ['English', 'Spanish'],
            concern_levels: [60, 25, 10, 3, 2]
        }
    };
}

function generateMockClusters() {
    return [
        { id: 1, name: "Technical Questions", count: 45, children: [] },
        { id: 2, name: "General Chat", count: 30, children: [] },
        { id: 3, name: "Creative Writing", count: 25, children: [] }
    ];
}

function generateMockFacets() {
    return ["Request", "Language", "Task", "Concerning"];
}

function generateMockUMAP(count) {
    const points = [];
    for (let i = 0; i < count; i++) {
        points.push({
            x: Math.random() * 100 - 50,
            y: Math.random() * 100 - 50,
            cluster: Math.floor(Math.random() * 3)
        });
    }
    return points;
}

// Results rendering
function renderResults() {
    if (!analysisResults) return;
    
    // Update statistics
    document.getElementById('total-conversations').textContent = 
        analysisResults.stats.total_conversations;
    document.getElementById('total-clusters').textContent = 
        analysisResults.stats.total_clusters;
    document.getElementById('languages').textContent = 
        analysisResults.stats.languages.join(', ');
    
    // Render visualizations
    renderHierarchy();
    renderUMAP();
    renderConversations();
    renderConcernChart();
}

function renderHierarchy() {
    const container = document.getElementById('hierarchy-tree');
    
    // Create a sample hierarchy structure
    const hierarchyHTML = `
        <div style="padding: 20px;">
            <h4 style="margin-bottom: 20px; color: #667eea;">Conversation Clusters</h4>
            <div class="tree-node" style="margin-left: 0;">
                <details open>
                    <summary style="cursor: pointer; padding: 10px; background: #f0f0f0; border-radius: 5px; margin-bottom: 10px;">
                        📁 <strong>All Conversations (${analysisResults.stats.total_conversations})</strong>
                    </summary>
                    <div style="margin-left: 20px;">
                        <details open>
                            <summary style="cursor: pointer; padding: 8px; background: #e8f4fd; border-radius: 5px; margin: 5px 0;">
                                💡 <strong>Technical Support (${Math.floor(analysisResults.stats.total_conversations * 0.3)})</strong>
                            </summary>
                            <div style="margin-left: 20px; padding: 5px;">
                                <div style="padding: 5px; margin: 3px 0;">• WiFi & Network Issues</div>
                                <div style="padding: 5px; margin: 3px 0;">• Software Updates</div>
                                <div style="padding: 5px; margin: 3px 0;">• Hardware Problems</div>
                            </div>
                        </details>
                        <details>
                            <summary style="cursor: pointer; padding: 8px; background: #fce4ec; border-radius: 5px; margin: 5px 0;">
                                ✍️ <strong>Creative & Academic (${Math.floor(analysisResults.stats.total_conversations * 0.25)})</strong>
                            </summary>
                            <div style="margin-left: 20px; padding: 5px;">
                                <div style="padding: 5px; margin: 3px 0;">• Story Writing</div>
                                <div style="padding: 5px; margin: 3px 0;">• Research Papers</div>
                                <div style="padding: 5px; margin: 3px 0;">• Essay Help</div>
                            </div>
                        </details>
                        <details>
                            <summary style="cursor: pointer; padding: 8px; background: #e8f5e9; border-radius: 5px; margin: 5px 0;">
                                🏃 <strong>Lifestyle Advice (${Math.floor(analysisResults.stats.total_conversations * 0.25)})</strong>
                            </summary>
                            <div style="margin-left: 20px; padding: 5px;">
                                <div style="padding: 5px; margin: 3px 0;">• Health & Fitness</div>
                                <div style="padding: 5px; margin: 3px 0;">• Cooking & Recipes</div>
                                <div style="padding: 5px; margin: 3px 0;">• Travel Planning</div>
                            </div>
                        </details>
                        <details>
                            <summary style="cursor: pointer; padding: 8px; background: #fff3e0; border-radius: 5px; margin: 5px 0;">
                                💰 <strong>Professional & Finance (${Math.floor(analysisResults.stats.total_conversations * 0.2)})</strong>
                            </summary>
                            <div style="margin-left: 20px; padding: 5px;">
                                <div style="padding: 5px; margin: 3px 0;">• Career Advice</div>
                                <div style="padding: 5px; margin: 3px 0;">• Personal Finance</div>
                                <div style="padding: 5px; margin: 3px 0;">• Job Searching</div>
                            </div>
                        </details>
                    </div>
                </details>
            </div>
        </div>
    `;
    
    container.innerHTML = hierarchyHTML;
    
    // Add click handlers
    container.querySelectorAll('summary').forEach(summary => {
        summary.addEventListener('click', (e) => {
            // Update cluster details panel
            const clusterName = summary.querySelector('strong').textContent;
            updateClusterDetails(clusterName);
        });
    });
}

function updateClusterDetails(clusterName) {
    const detailsPanel = document.getElementById('cluster-details');
    
    const detailsHTML = `
        <div style="padding: 20px;">
            <h4 style="color: #667eea; margin-bottom: 15px;">${clusterName}</h4>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h5 style="margin-bottom: 10px;">Summary</h5>
                <p style="color: #666; line-height: 1.6;">
                    This cluster contains conversations related to ${clusterName.toLowerCase()}.
                    The topics are characterized by detailed, helpful responses addressing specific user needs.
                </p>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h5 style="margin-bottom: 10px;">Key Characteristics</h5>
                <ul style="color: #666; margin-left: 20px;">
                    <li>Average conversation length: 4-6 exchanges</li>
                    <li>High information density</li>
                    <li>Problem-solving focused</li>
                    <li>Step-by-step guidance provided</li>
                </ul>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                <h5 style="margin-bottom: 10px;">Sample Topics</h5>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    <span style="background: #667eea; color: white; padding: 4px 12px; border-radius: 15px; font-size: 0.9em;">Troubleshooting</span>
                    <span style="background: #667eea; color: white; padding: 4px 12px; border-radius: 15px; font-size: 0.9em;">How-to Guides</span>
                    <span style="background: #667eea; color: white; padding: 4px 12px; border-radius: 15px; font-size: 0.9em;">Best Practices</span>
                </div>
            </div>
        </div>
    `;
    
    detailsPanel.innerHTML = detailsHTML;
}

function renderUMAP() {
    const container = document.getElementById('umap-plot');
    
    // Create a simple canvas-based visualization
    const umapHTML = `
        <div style="padding: 20px;">
            <canvas id="umap-canvas" width="500" height="500" style="border: 1px solid #e0e0e0; border-radius: 8px; width: 100%; max-width: 500px;"></canvas>
        </div>
    `;
    
    container.innerHTML = umapHTML;
    
    // Draw points on canvas
    setTimeout(() => {
        const canvas = document.getElementById('umap-canvas');
        if (canvas && analysisResults) {
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;
            
            // Clear canvas
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, width, height);
            
            // Draw grid
            ctx.strokeStyle = '#e0e0e0';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 10; i++) {
                const x = (width / 10) * i;
                const y = (height / 10) * i;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }
            
            // Draw points
            const colors = ['#667eea', '#e91e63', '#4caf50', '#ff9800'];
            analysisResults.umap.forEach(point => {
                const x = (point.x + 50) * width / 100;
                const y = (point.y + 50) * height / 100;
                
                ctx.fillStyle = colors[point.cluster % colors.length];
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, 2 * Math.PI);
                ctx.fill();
                
                // Add glow effect
                ctx.shadowColor = colors[point.cluster % colors.length];
                ctx.shadowBlur = 10;
                ctx.fill();
                ctx.shadowBlur = 0;
            });
            
            // Add legend
            ctx.fillStyle = '#333';
            ctx.font = '12px sans-serif';
            const legendItems = ['Technical', 'Creative', 'Lifestyle', 'Professional'];
            legendItems.forEach((item, i) => {
                ctx.fillStyle = colors[i];
                ctx.fillRect(10, 10 + i * 20, 10, 10);
                ctx.fillStyle = '#333';
                ctx.fillText(item, 25, 18 + i * 20);
            });
        }
    }, 100);
}

function renderConversations() {
    const list = document.getElementById('conversation-list');
    list.innerHTML = '';
    
    analysisResults.conversations.slice(0, 10).forEach((conv, i) => {
        const item = document.createElement('div');
        item.className = 'conversation-item';
        item.innerHTML = `<p>Conversation ${i + 1}</p>`;
        item.onclick = () => showConversationDetail(i);
        list.appendChild(item);
    });
}

function showConversationDetail(index) {
    const detail = document.getElementById('conversation-detail');
    const conv = analysisResults.conversations[index];
    
    detail.innerHTML = '<h3>Conversation Detail</h3>';
    conv.forEach(message => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${message.role}`;
        msgDiv.innerHTML = `<strong>${message.role}:</strong> ${message.content}`;
        detail.appendChild(msgDiv);
    });
}

function renderConcernChart() {
    // Chart.js or D3.js chart would go here
    const canvas = document.getElementById('concern-chart');
    // Render chart...
}

// Tab navigation
function showTab(tabName) {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    event.target.classList.add('active');
    document.getElementById(tabName + '-tab').classList.add('active');
}

// Export functions
function exportJSON() {
    const dataStr = JSON.stringify(analysisResults, null, 2);
    downloadFile(dataStr, 'openclio-results.json', 'application/json');
}

function exportCSV() {
    // Convert to CSV format
    alert('CSV export coming soon!');
}

function exportReport() {
    // Generate PDF report
    alert('Report generation coming soon!');
}

function downloadFile(content, filename, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// Utility functions
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function startNewAnalysis() {
    uploadedData = null;
    analysisResults = null;
    showWelcomeScreen();
}

// Visualization controls
function resetZoom() {
    // Reset D3 zoom
}

function toggleLabels() {
    // Toggle labels in visualization
}

function updateColors() {
    // Update color scheme
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    showWelcomeScreen();
    
    // Close modal when clicking outside
    window.onclick = (event) => {
        const modal = document.getElementById('paste-modal');
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    };
});

// Initialize dropdowns when configuration screen is shown
function initializeConfigDefaults() {
    const llmProvider = document.getElementById('llm-provider');
    const embeddingProvider = document.getElementById('embedding-provider');
    
    if (llmProvider && typeof updateLLMOptions === 'function') {
        if (!llmProvider.value || llmProvider.value === '') {
            llmProvider.value = 'openai';
        }
        updateLLMOptions();
    }
    
    if (embeddingProvider && typeof updateEmbeddingOptions === 'function') {
        if (!embeddingProvider.value || embeddingProvider.value === '') {
            embeddingProvider.value = 'openai';
        }
        updateEmbeddingOptions();
    }
}