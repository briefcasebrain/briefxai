#!/bin/bash

# BriefXAI Code Coverage Generation Script
# This script generates comprehensive code coverage reports using cargo-tarpaulin

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 BriefXAI Code Coverage Generation${NC}"
echo "==============================================="

# Check if cargo-tarpaulin is installed
if ! command -v cargo-tarpaulin &> /dev/null; then
    echo -e "${YELLOW}⚠️  cargo-tarpaulin not found. Installing...${NC}"
    cargo install cargo-tarpaulin
fi

# Create coverage output directory
COVERAGE_DIR="target/coverage"
mkdir -p "$COVERAGE_DIR"

echo -e "${BLUE}📊 Running comprehensive coverage analysis...${NC}"

# Run tarpaulin with comprehensive options
cargo tarpaulin \
    --verbose \
    --all-features \
    --workspace \
    --timeout 300 \
    --exclude-files 'target/*' \
    --exclude-files 'tests/*' \
    --exclude-files 'benches/*' \
    --exclude-files '*/main.rs' \
    --exclude-files '*/bin/*' \
    --out xml \
    --out html \
    --out json \
    --out lcov \
    --output-dir "$COVERAGE_DIR" \
    --engine llvm \
    --release

echo -e "${GREEN}✅ Coverage analysis complete!${NC}"

# Generate summary report
echo -e "${BLUE}📋 Generating coverage summary...${NC}"

# Extract coverage percentage from JSON output
if [ -f "$COVERAGE_DIR/tarpaulin-report.json" ]; then
    COVERAGE_PERCENT=$(jq -r '.files[] | .coverage' "$COVERAGE_DIR/tarpaulin-report.json" | awk '{sum += $1; count++} END {if (count > 0) print sum/count; else print 0}')
    echo -e "${GREEN}Overall Coverage: ${COVERAGE_PERCENT}%${NC}"
fi

# Generate module-specific coverage report
echo -e "${BLUE}📈 Module Coverage Breakdown:${NC}"
echo "=================================="

# Parse HTML report for module breakdown (simplified)
if [ -f "$COVERAGE_DIR/tarpaulin-report.html" ]; then
    echo "📄 Detailed HTML report: $COVERAGE_DIR/tarpaulin-report.html"
fi

# Check coverage thresholds using awk instead of bc
MINIMUM_COVERAGE=70
GOOD_COVERAGE=85
EXCELLENT_COVERAGE=95

if awk -v a="$COVERAGE_PERCENT" -v b="$EXCELLENT_COVERAGE" 'BEGIN {exit !(a >= b)}'; then
    echo -e "${GREEN}🎉 Excellent coverage! (>= ${EXCELLENT_COVERAGE}%)${NC}"
elif awk -v a="$COVERAGE_PERCENT" -v b="$GOOD_COVERAGE" 'BEGIN {exit !(a >= b)}'; then
    echo -e "${GREEN}👍 Good coverage! (>= ${GOOD_COVERAGE}%)${NC}"
elif awk -v a="$COVERAGE_PERCENT" -v b="$MINIMUM_COVERAGE" 'BEGIN {exit !(a >= b)}'; then
    echo -e "${YELLOW}⚠️  Acceptable coverage (>= ${MINIMUM_COVERAGE}%), but could be improved${NC}"
else
    echo -e "${RED}❌ Coverage below minimum threshold (< ${MINIMUM_COVERAGE}%)${NC}"
    echo "Consider adding more tests to improve coverage."
fi

# Generate coverage badge data
echo -e "${BLUE}🏷️  Generating coverage badge data...${NC}"
BADGE_COLOR="red"
if awk -v a="$COVERAGE_PERCENT" -v b="$EXCELLENT_COVERAGE" 'BEGIN {exit !(a >= b)}'; then
    BADGE_COLOR="brightgreen"
elif awk -v a="$COVERAGE_PERCENT" -v b="$GOOD_COVERAGE" 'BEGIN {exit !(a >= b)}'; then
    BADGE_COLOR="green"
elif awk -v a="$COVERAGE_PERCENT" -v b="$MINIMUM_COVERAGE" 'BEGIN {exit !(a >= b)}'; then
    BADGE_COLOR="yellow"
fi

echo "{\"schemaVersion\": 1, \"label\": \"coverage\", \"message\": \"${COVERAGE_PERCENT}%\", \"color\": \"${BADGE_COLOR}\"}" > "$COVERAGE_DIR/coverage-badge.json"

# Open HTML report if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${BLUE}🔍 Opening coverage report in browser...${NC}"
    open "$COVERAGE_DIR/tarpaulin-report.html"
fi

echo ""
echo -e "${GREEN}✨ Coverage generation complete!${NC}"
echo "📁 Reports available in: $COVERAGE_DIR"
echo "   - HTML: tarpaulin-report.html"
echo "   - XML: cobertura.xml"
echo "   - LCOV: lcov.info"
echo "   - JSON: tarpaulin-report.json"
echo ""

# Generate additional analysis
echo -e "${BLUE}🔍 Additional Analysis:${NC}"
echo "======================="

# Count total lines of code
echo "📝 Lines of Code Analysis:"
find src -name "*.rs" -exec wc -l {} + | tail -1 | awk '{print "   Total LOC: " $1}'

# Count test functions
TEST_COUNT=$(grep -r "#\[test\]" src tests 2>/dev/null | wc -l || echo "0")
echo "🧪 Test Functions: $TEST_COUNT"

# Estimate lines covered
TOTAL_LOC=$(find src -name "*.rs" -exec wc -l {} + | tail -1 | awk '{print $1}')
COVERED_LOC=$(awk -v total="$TOTAL_LOC" -v percent="$COVERAGE_PERCENT" 'BEGIN {printf "%.0f", total * percent / 100}')
echo "✅ Estimated Lines Covered: $COVERED_LOC / $TOTAL_LOC"

# Generate recommendations
echo ""
echo -e "${YELLOW}💡 Recommendations:${NC}"
if ! awk -v a="$COVERAGE_PERCENT" -v b="$MINIMUM_COVERAGE" 'BEGIN {exit !(a < b)}'; then
    echo "   - Add more unit tests for core modules"
    echo "   - Focus on testing error conditions and edge cases"
    echo "   - Consider property-based testing for complex algorithms"
fi

if ! awk -v a="$COVERAGE_PERCENT" -v b="$GOOD_COVERAGE" 'BEGIN {exit !(a < b)}'; then
    echo "   - Add integration tests for complete workflows"
    echo "   - Test configuration validation and error handling"
    echo "   - Add tests for CLI argument parsing and validation"
fi

echo "   - Consider adding mutation testing for test quality"
echo "   - Review uncovered lines for critical code paths"
echo "   - Update tests when adding new features"

echo ""
echo -e "${GREEN}Happy testing! 🧪✨${NC}"