#!/bin/bash
# Comprehensive test of simplex-only mode fix

cd /root/sovereign-project/sovereign-backend

echo "=========================================="
echo "Testing Simplex-Only Mode Fix"
echo "=========================================="
echo ""

echo "1. AUTO mode (baseline):"
echo "----------------------------------------"
unset SOVEREIGN_LP_ALGORITHM
./build/solver/sovereign solve benchmarks/datasets/miplib/official/flugpl.mps 2>&1 | grep -E '(status|objective_value|optimality_proven|message)' | head -6
echo ""

echo "2. IPM mode:"
echo "----------------------------------------"
export SOVEREIGN_LP_ALGORITHM=ipm
./build/solver/sovereign solve benchmarks/datasets/miplib/official/flugpl.mps 2>&1 | grep -E '(status|objective_value|optimality_proven)' | head -3
echo ""

echo "3. SIMPLEX mode (the fix being tested):"
echo "----------------------------------------"
export SOVEREIGN_LP_ALGORITHM=simplex
./build/solver/sovereign solve benchmarks/datasets/miplib/official/flugpl.mps 2>&1 | grep -E '(status|objective_value|optimality_proven|warnings)' | head -10
echo ""

echo "4. Unit tests:"
echo "----------------------------------------"
cd build && ctest --output-on-failure 2>&1 | tail -5
cd ..
echo ""

echo "=========================================="
echo "Summary:"
echo "=========================================="
echo "Expected: All modes find objective 1201500"
echo "Known issue: Simplex-only mode may return NUMERICAL_ERROR"
echo "  due to presolved LP structure being difficult for simplex"
echo "Root cause: Presolved flugpl creates dual infeasible vertex"
echo "  (reduced cost = -708) that simplex cannot improve"
