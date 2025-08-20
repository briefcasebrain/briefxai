#!/bin/bash

# BriefXAI Comprehensive Quality Check Script
# This script runs all quality checks in the correct order

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🚀 BriefXAI Comprehensive Quality Check${NC}"
echo "=============================================="
echo ""

# Function to run a check with timing
run_check() {
    local name="$1"
    local script="$2"
    local start_time=$(date +%s)
    
    echo -e "${BLUE}🔄 Running: $name${NC}"
    
    if eval "$script"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${GREEN}✅ $name completed successfully in ${duration}s${NC}"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${RED}❌ $name failed after ${duration}s${NC}"
        return 1
    fi
    echo ""
}

# Check if we're in the project root
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}❌ Please run this script from the project root directory${NC}"
    exit 1
fi

# Track overall success
TOTAL_CHECKS=0
PASSED_CHECKS=0

echo -e "${BLUE}📋 Phase 1: Code Quality and Formatting${NC}"
echo "========================================="

((TOTAL_CHECKS++))
if run_check "Code Formatting Check" "cargo fmt --all -- --check"; then
    ((PASSED_CHECKS++))
fi

((TOTAL_CHECKS++))
if run_check "Linting and Code Analysis" "./scripts/lint.sh"; then
    ((PASSED_CHECKS++))
fi

echo -e "${BLUE}🧪 Phase 2: Testing${NC}"
echo "==================="

((TOTAL_CHECKS++))
if run_check "Unit Tests" "cargo test --lib"; then
    ((PASSED_CHECKS++))
fi

((TOTAL_CHECKS++))
if run_check "Integration Tests" "cargo test --test '*'"; then
    ((PASSED_CHECKS++))
fi

((TOTAL_CHECKS++))
if run_check "Property-Based Tests" "cargo test --test property_based_tests"; then
    ((PASSED_CHECKS++))
fi

echo -e "${BLUE}📊 Phase 3: Coverage and Performance${NC}"
echo "====================================="

((TOTAL_CHECKS++))
if run_check "Code Coverage Analysis" "./scripts/coverage.sh"; then
    ((PASSED_CHECKS++))
fi

((TOTAL_CHECKS++))
if run_check "Performance Benchmarks" "cargo bench --bench performance"; then
    ((PASSED_CHECKS++))
fi

echo -e "${BLUE}🔧 Phase 4: Build and Compilation${NC}"
echo "=================================="

((TOTAL_CHECKS++))
if run_check "Debug Build" "cargo build"; then
    ((PASSED_CHECKS++))
fi

((TOTAL_CHECKS++))
if run_check "Release Build" "cargo build --release"; then
    ((PASSED_CHECKS++))
fi

((TOTAL_CHECKS++))
if run_check "Documentation Build" "cargo doc --all-features --no-deps"; then
    ((PASSED_CHECKS++))
fi

echo -e "${BLUE}🔍 Phase 5: Additional Checks${NC}"
echo "=============================="

((TOTAL_CHECKS++))
if run_check "Dependency Audit" "cargo audit || echo 'cargo-audit not available'"; then
    ((PASSED_CHECKS++))
fi

((TOTAL_CHECKS++))
if run_check "Unused Dependencies" "cargo machete || echo 'cargo-machete not available'"; then
    ((PASSED_CHECKS++))
fi

# Generate final report
echo ""
echo -e "${PURPLE}📊 Final Quality Report${NC}"
echo "========================"
echo "Total checks: $TOTAL_CHECKS"
echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed: ${RED}$((TOTAL_CHECKS - PASSED_CHECKS))${NC}"

SUCCESS_RATE=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
echo "Success rate: $SUCCESS_RATE%"

# Quality grade based on success rate
if [ $SUCCESS_RATE -eq 100 ]; then
    echo -e "${GREEN}🏆 Quality Grade: A+ (Excellent)${NC}"
    echo -e "${GREEN}🎉 All quality checks passed! Your code is production-ready.${NC}"
elif [ $SUCCESS_RATE -ge 90 ]; then
    echo -e "${GREEN}🥇 Quality Grade: A (Very Good)${NC}"
    echo -e "${GREEN}✨ Most checks passed. Minor issues to address.${NC}"
elif [ $SUCCESS_RATE -ge 80 ]; then
    echo -e "${YELLOW}🥈 Quality Grade: B (Good)${NC}"
    echo -e "${YELLOW}⚠️  Good quality with some areas for improvement.${NC}"
elif [ $SUCCESS_RATE -ge 70 ]; then
    echo -e "${YELLOW}🥉 Quality Grade: C (Acceptable)${NC}"
    echo -e "${YELLOW}⚠️  Acceptable quality but needs attention.${NC}"
else
    echo -e "${RED}📉 Quality Grade: D (Needs Work)${NC}"
    echo -e "${RED}❌ Multiple issues found. Please address before deploying.${NC}"
fi

echo ""
echo -e "${BLUE}📁 Generated Reports:${NC}"
echo "   - Coverage: target/coverage/"
echo "   - Benchmarks: target/criterion/"
echo "   - Documentation: target/doc/"
echo ""

# Recommendations based on results
echo -e "${BLUE}💡 Recommendations:${NC}"
if [ $SUCCESS_RATE -lt 100 ]; then
    echo "   - Review and fix failed checks above"
    echo "   - Ensure all tests pass before deploying"
    echo "   - Consider improving code coverage if below 80%"
fi

if [ $SUCCESS_RATE -ge 80 ]; then
    echo "   - Great work! Consider running this script in CI/CD"
    echo "   - Review performance benchmarks for optimization opportunities"
    echo "   - Keep monitoring code quality metrics"
else
    echo "   - Focus on fixing critical issues first"
    echo "   - Run individual check scripts for detailed output"
    echo "   - Consider pair programming for complex fixes"
fi

echo ""

# Exit with appropriate code
if [ $SUCCESS_RATE -ge 80 ]; then
    echo -e "${GREEN}🎯 Quality check completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}🚨 Quality check completed with issues.${NC}"
    exit 1
fi