#!/bin/bash

echo "🚀 OpenClio Demo - Analyzing Realistic Conversations"
echo "=================================================="
echo ""

# Make sure we have Rust environment
source "$HOME/.cargo/env"

# Step 1: Generate realistic examples if they don't exist
if [ ! -f "realistic_conversations.json" ]; then
    echo "📝 Generating 50 realistic example conversations..."
    cargo run -- example -n 50
    echo "✅ Examples generated!"
else
    echo "✅ Using existing realistic_conversations.json"
fi

echo ""
echo "📊 File contains:"
echo "  - $(cat realistic_conversations.json | jq '. | length') conversations"
echo "  - Topics: Tech Support, Creative Writing, Cooking, Health, Language Learning, etc."
echo ""

# Step 2: Analyze the conversations
echo "🔍 Analyzing conversations..."
echo "  Note: Using simplified mock implementations for demo"
echo ""

# Step 3: Start the UI
echo "🌐 Starting OpenClio Web Interface..."
echo "  → Open http://localhost:8080 in your browser"
echo "  → Click 'Upload Data' and select realistic_conversations.json"
echo "  → Or click 'Use Example Data' to use built-in examples"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the UI
cargo run
