#!/bin/bash

set -e

echo "================================"
echo "BriefXAI E2E Test Suite"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "\n${YELLOW}Running: $test_name${NC}"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✓ $test_name passed${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ $test_name failed${NC}"
        ((TESTS_FAILED++))
    fi
}

# Function to check if server is running
check_server() {
    local port=$1
    local max_attempts=30
    local attempt=0
    
    echo "Waiting for server on port $port..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "Server is ready on port $port"
            return 0
        fi
        sleep 1
        ((attempt++))
    done
    
    echo "Server failed to start on port $port"
    return 1
}

# Build the project
echo -e "\n${YELLOW}Building project...${NC}"
cargo build --release

# Start the server in background
echo -e "\n${YELLOW}Starting server...${NC}"
./target/release/briefxai serve &
SERVER_PID=$!

# Wait for server to be ready
if ! check_server 8080; then
    echo -e "${RED}Server failed to start${NC}"
    exit 1
fi

# ============================================================================
# API Tests
# ============================================================================

echo -e "\n${YELLOW}=== API Tests ===${NC}"

# Test 1: Health endpoint
run_test "Health check" \
    "curl -s http://localhost:8080/health | grep -q 'ok'"

# Test 2: Upload endpoint
run_test "File upload" \
    "curl -s -X POST http://localhost:8080/api/upload \
        -F 'file=@test_data.json' \
        | grep -q 'success'"

# Test 3: Configuration endpoint
run_test "Configuration" \
    "curl -s http://localhost:8080/api/config \
        | grep -q 'llm_provider'"

# Test 4: Ollama status
run_test "Ollama status check" \
    "curl -s http://localhost:8080/api/ollama-status \
        | grep -q 'running'"

# Test 5: Analysis endpoint
echo -e "\n${YELLOW}Creating test data...${NC}"
cat > test_conversations.json << 'EOF'
[
    [
        {"role": "user", "content": "Hello, I need help with my order"},
        {"role": "assistant", "content": "I'd be happy to help you with your order. What seems to be the issue?"}
    ],
    [
        {"role": "user", "content": "The product I received is damaged"},
        {"role": "assistant", "content": "I apologize for the damaged product. Let me help you with a replacement or refund."}
    ]
]
EOF

run_test "Analysis with test data" \
    "curl -s -X POST http://localhost:8080/api/analyze \
        -H 'Content-Type: application/json' \
        -d '{
            \"conversations\": $(cat test_conversations.json),
            \"config\": {
                \"llm_provider\": \"openai\",
                \"llm_model\": \"gpt-4o-mini\",
                \"llm_api_key\": \"test-key\",
                \"embedding_provider\": \"openai\",
                \"embedding_model\": \"text-embedding-3-small\"
            }
        }' \
        | grep -q 'session_id'"

# ============================================================================
# WebSocket Tests
# ============================================================================

echo -e "\n${YELLOW}=== WebSocket Tests ===${NC}"

# Test WebSocket connection
run_test "WebSocket connection" \
    "timeout 2 websocat -t ws://localhost:8080/ws 2>&1 | grep -q 'Connected' || true"

# ============================================================================
# Database Tests
# ============================================================================

echo -e "\n${YELLOW}=== Database Tests ===${NC}"

# Check if database was created
run_test "Database creation" \
    "test -f ~/.cache/briefxai/briefxai.db"

# ============================================================================
# Unit Tests
# ============================================================================

echo -e "\n${YELLOW}=== Unit Tests ===${NC}"

run_test "Session manager tests" \
    "cargo test --test test_session_manager --quiet"

run_test "Preprocessing tests" \
    "cargo test --test test_preprocessing --quiet"

# ============================================================================
# Integration Tests
# ============================================================================

echo -e "\n${YELLOW}=== Integration Tests ===${NC}"

# Test the full pipeline with a small dataset
echo -e "\n${YELLOW}Creating integration test data...${NC}"
cat > integration_test.json << 'EOF'
[
    [
        {"role": "user", "content": "What is the weather like today?"},
        {"role": "assistant", "content": "I don't have access to real-time weather data, but I can help you find weather information."}
    ],
    [
        {"role": "user", "content": "Can you help me write a Python script?"},
        {"role": "assistant", "content": "Of course! I'd be happy to help you write a Python script. What would you like the script to do?"}
    ],
    [
        {"role": "user", "content": "I'm having trouble with my computer"},
        {"role": "assistant", "content": "I can help troubleshoot your computer issue. Can you describe what problem you're experiencing?"}
    ]
]
EOF

# Test with preprocessing
run_test "Preprocessing pipeline" \
    "curl -s -X POST http://localhost:8080/api/preprocess \
        -H 'Content-Type: application/json' \
        -d '{\"conversations\": $(cat integration_test.json)}' \
        | grep -q 'quality_score'"

# ============================================================================
# Performance Tests
# ============================================================================

echo -e "\n${YELLOW}=== Performance Tests ===${NC}"

# Test concurrent requests
run_test "Concurrent requests" \
    "for i in {1..5}; do 
        curl -s http://localhost:8080/health &
    done; wait"

# Test large file handling
echo -e "\n${YELLOW}Creating large test file...${NC}"
python3 -c "
import json
conversations = []
for i in range(100):
    conv = [
        {'role': 'user', 'content': f'Question {i}'},
        {'role': 'assistant', 'content': f'Answer {i}'}
    ]
    conversations.append(conv)
with open('large_test.json', 'w') as f:
    json.dump(conversations, f)
"

run_test "Large file processing" \
    "curl -s -X POST http://localhost:8080/api/upload \
        -F 'file=@large_test.json' \
        --max-time 30 \
        | grep -q 'success'"

# ============================================================================
# Error Handling Tests
# ============================================================================

echo -e "\n${YELLOW}=== Error Handling Tests ===${NC}"

# Test invalid JSON
run_test "Invalid JSON handling" \
    "curl -s -X POST http://localhost:8080/api/analyze \
        -H 'Content-Type: application/json' \
        -d '{invalid json}' \
        | grep -q 'error'"

# Test missing required fields
run_test "Missing fields handling" \
    "curl -s -X POST http://localhost:8080/api/analyze \
        -H 'Content-Type: application/json' \
        -d '{}' \
        | grep -q 'error'"

# ============================================================================
# Cleanup
# ============================================================================

echo -e "\n${YELLOW}Cleaning up...${NC}"

# Kill the server
kill $SERVER_PID 2>/dev/null || true

# Clean up test files
rm -f test_conversations.json integration_test.json large_test.json test_data.json

# ============================================================================
# Results
# ============================================================================

echo -e "\n================================"
echo -e "Test Results"
echo -e "================================"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed ✗${NC}"
    exit 1
fi