#!/bin/bash

# BriefXAI Comprehensive Linting Script
# This script runs all linting, formatting, and code quality checks

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters for summary
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Function to run a check and track results
run_check() {
    local name="$1"
    local cmd="$2"
    local allow_failure="${3:-false}"
    
    echo -e "${BLUE}🔍 Running: $name${NC}"
    ((TOTAL_CHECKS++))
    
    if eval "$cmd"; then
        echo -e "${GREEN}✅ $name: PASSED${NC}"
        ((PASSED_CHECKS++))
    else
        if [ "$allow_failure" = "true" ]; then
            echo -e "${YELLOW}⚠️  $name: FAILED (allowed)${NC}"
            ((PASSED_CHECKS++))
        else
            echo -e "${RED}❌ $name: FAILED${NC}"
            ((FAILED_CHECKS++))
        fi
    fi
    echo ""
}

echo -e "${BLUE}🧹 BriefXAI Code Quality Check${NC}"
echo "======================================"
echo ""

# 1. Format check
run_check "Code Formatting" "cargo fmt --all -- --check"

# 2. Core Clippy lints (strict)
run_check "Clippy - Core Lints" \
    "cargo clippy --all-targets --all-features -- \
    -D warnings \
    -D clippy::all \
    -D clippy::correctness \
    -D clippy::suspicious \
    -D clippy::complexity \
    -D clippy::perf \
    -D clippy::style"

# 3. Pedantic lints (with allowances for ML domain)
run_check "Clippy - Pedantic Lints" \
    "cargo clippy --all-targets --all-features -- \
    -W clippy::pedantic \
    -A clippy::cast_possible_truncation \
    -A clippy::cast_precision_loss \
    -A clippy::cast_sign_loss \
    -A clippy::similar_names \
    -A clippy::too_many_lines \
    -A clippy::module_name_repetitions \
    -A clippy::must_use_candidate \
    -A clippy::missing_errors_doc \
    -A clippy::missing_panics_doc" \
    true

# 4. Nursery lints (experimental, allowed to fail)
run_check "Clippy - Nursery Lints" \
    "cargo clippy --all-targets --all-features -- \
    -W clippy::nursery \
    -A clippy::significant_drop_in_scrutinee \
    -A clippy::significant_drop_tightening" \
    true

# 5. Cargo lints (dependency management)
run_check "Clippy - Cargo Lints" \
    "cargo clippy --all-targets --all-features -- \
    -W clippy::cargo \
    -A clippy::multiple_crate_versions" \
    true

# 6. Check for unused dependencies
run_check "Unused Dependencies" \
    "if command -v cargo-machete >/dev/null 2>&1; then cargo machete; else echo 'cargo-machete not installed, skipping'; fi" \
    true

# 7. Security audit
run_check "Security Audit" \
    "if command -v cargo-audit >/dev/null 2>&1; then cargo audit; else echo 'cargo-audit not installed, skipping'; fi" \
    true

# 8. Check for duplicate dependencies
run_check "Duplicate Dependencies" \
    "cargo tree --duplicates | head -20" \
    true

# 9. Spell check
run_check "Spell Check" \
    "if command -v typos >/dev/null 2>&1; then typos --config .typos.toml; else echo 'typos not installed, skipping'; fi" \
    true

# 10. Code complexity analysis
run_check "Code Complexity" \
    "if command -v tokei >/dev/null 2>&1; then tokei; else echo 'tokei not installed, skipping'; fi" \
    true

# 11. Documentation check
run_check "Documentation Check" \
    "cargo doc --all-features --no-deps --quiet" \
    true

# 12. Check for common issues
echo -e "${BLUE}🔍 Additional Code Quality Checks${NC}"
echo "=================================="

# Check for TODO/FIXME comments
echo "📝 Checking for TODO/FIXME comments:"
TODO_COUNT=$(grep -r "TODO\|FIXME\|XXX\|HACK" src/ --include="*.rs" | wc -l || echo "0")
if [ "$TODO_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $TODO_COUNT TODO/FIXME comments${NC}"
    grep -r "TODO\|FIXME\|XXX\|HACK" src/ --include="*.rs" | head -10
else
    echo -e "${GREEN}✅ No TODO/FIXME comments found${NC}"
fi
echo ""

# Check for println! in non-test code
echo "🐛 Checking for debug prints in source code:"
DEBUG_COUNT=$(grep -r "println!\|dbg!\|eprintln!" src/ --include="*.rs" | grep -v "#\[cfg(test)\]" | wc -l || echo "0")
if [ "$DEBUG_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $DEBUG_COUNT debug print statements${NC}"
    grep -r "println!\|dbg!\|eprintln!" src/ --include="*.rs" | grep -v "#\[cfg(test)\]" | head -5
else
    echo -e "${GREEN}✅ No debug print statements found${NC}"
fi
echo ""

# Check for unwrap() in non-test code
echo "💥 Checking for unwrap() calls in source code:"
UNWRAP_COUNT=$(grep -r "\.unwrap()" src/ --include="*.rs" | grep -v "#\[cfg(test)\]" | grep -v "tests" | wc -l || echo "0")
if [ "$UNWRAP_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $UNWRAP_COUNT unwrap() calls (consider using proper error handling)${NC}"
    grep -r "\.unwrap()" src/ --include="*.rs" | grep -v "#\[cfg(test)\]" | grep -v "tests" | head -5
else
    echo -e "${GREEN}✅ No unwrap() calls found in source code${NC}"
fi
echo ""

# Check for large files
echo "📏 Checking for large source files:"
LARGE_FILES=$(find src/ -name "*.rs" -exec wc -l {} + | awk '$1 > 500 {print $2 " (" $1 " lines)"}' | head -5)
if [ -n "$LARGE_FILES" ]; then
    echo -e "${YELLOW}⚠️  Large files found (>500 lines):${NC}"
    echo "$LARGE_FILES"
else
    echo -e "${GREEN}✅ No excessively large files found${NC}"
fi
echo ""

# Generate summary
echo -e "${BLUE}📊 Lint Summary${NC}"
echo "==============="
echo "Total checks: $TOTAL_CHECKS"
echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"

SUCCESS_RATE=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
echo "Success rate: $SUCCESS_RATE%"

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}🎉 All critical checks passed!${NC}"
    exit 0
elif [ $SUCCESS_RATE -ge 80 ]; then
    echo -e "${YELLOW}⚠️  Most checks passed, but some issues need attention${NC}"
    exit 1
else
    echo -e "${RED}❌ Multiple critical issues found${NC}"
    exit 1
fi