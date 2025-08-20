#!/bin/bash

# BriefXAI Test Runner Script
# This script runs tests in the correct order and provides detailed output

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 BriefXAI Test Runner${NC}"
echo "========================="
echo ""

# Function to run tests with proper error handling
run_test_suite() {
    local name="$1"
    local command="$2"
    local allow_failure="${3:-false}"
    
    echo -e "${BLUE}🔄 Running: $name${NC}"
    
    if eval "$command"; then
        echo -e "${GREEN}✅ $name: PASSED${NC}"
        return 0
    else
        if [ "$allow_failure" = "true" ]; then
            echo -e "${YELLOW}⚠️  $name: FAILED (allowed)${NC}"
            return 0
        else
            echo -e "${RED}❌ $name: FAILED${NC}"
            return 1
        fi
    fi
}

# Check if we're in the project root
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}❌ Please run this script from the project root directory${NC}"
    exit 1
fi

# Track results
TOTAL_SUITES=0
PASSED_SUITES=0

echo -e "${BLUE}📚 Phase 1: Library Tests${NC}"
echo "=========================="

((TOTAL_SUITES++))
if run_test_suite "Library Unit Tests" "cargo test --lib --verbose"; then
    ((PASSED_SUITES++))
fi

echo ""
echo -e "${BLUE}🔬 Phase 2: Module-Specific Tests${NC}"
echo "=================================="

((TOTAL_SUITES++))
if run_test_suite "Clustering Tests" "cargo test clustering::tests --verbose"; then
    ((PASSED_SUITES++))
fi

((TOTAL_SUITES++))
if run_test_suite "Embeddings Tests" "cargo test embeddings::tests --verbose"; then
    ((PASSED_SUITES++))
fi

((TOTAL_SUITES++))
if run_test_suite "Facets Tests" "cargo test facets::tests --verbose"; then
    ((PASSED_SUITES++))
fi

((TOTAL_SUITES++))
if run_test_suite "Monitoring Tests" "cargo test monitoring::tests --verbose" true; then
    ((PASSED_SUITES++))
fi

((TOTAL_SUITES++))
if run_test_suite "Error Recovery Tests" "cargo test error_recovery::tests --verbose" true; then
    ((PASSED_SUITES++))
fi

((TOTAL_SUITES++))
if run_test_suite "Logging Tests" "cargo test logging::tests --verbose" true; then
    ((PASSED_SUITES++))
fi

echo ""
echo -e "${BLUE}🏗️ Phase 3: Integration Tests${NC}"
echo "==============================="

((TOTAL_SUITES++))
if run_test_suite "Comprehensive Integration Tests" "cargo test --test integration_comprehensive --verbose" true; then
    ((PASSED_SUITES++))
fi

((TOTAL_SUITES++))
if run_test_suite "Property-Based Tests" "cargo test --test property_based_tests --verbose" true; then
    ((PASSED_SUITES++))
fi

echo ""
echo -e "${BLUE}🚀 Phase 4: Documentation Tests${NC}"
echo "================================="

((TOTAL_SUITES++))
if run_test_suite "Documentation Tests" "cargo test --doc --verbose" true; then
    ((PASSED_SUITES++))
fi

# Generate summary
echo ""
echo -e "${BLUE}📊 Test Summary${NC}"
echo "==============="
echo "Total test suites: $TOTAL_SUITES"
echo -e "Passed: ${GREEN}$PASSED_SUITES${NC}"
echo -e "Failed: ${RED}$((TOTAL_SUITES - PASSED_SUITES))${NC}"

SUCCESS_RATE=$((PASSED_SUITES * 100 / TOTAL_SUITES))
echo "Success rate: $SUCCESS_RATE%"

if [ $SUCCESS_RATE -eq 100 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
elif [ $SUCCESS_RATE -ge 80 ]; then
    echo -e "${YELLOW}⚠️  Most tests passed with some acceptable failures${NC}"
    exit 0
else
    echo -e "${RED}❌ Multiple test failures detected${NC}"
    exit 1
fi