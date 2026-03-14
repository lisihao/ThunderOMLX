#!/bin/bash
# Automated P0 Features Benchmark
#
# This script:
# 1. Starts oMLX server with P0 optimizations
# 2. Runs performance benchmark
# 3. Compares with baseline (119.3 tok/s)
# 4. Reports results

set -e

echo "========================================"
echo "P0 Features Performance Benchmark"
echo "========================================"
echo ""

# Environment setup
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PYTHONPATH=/Users/lisihao/ThunderOMLX/src:$PYTHONPATH

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
MODEL_ID="mlx-community/Qwen3.5-35B-A3B-6bit"
PORT=8000
BASELINE_TPS=119.3

echo "📋 Configuration:"
echo "  Model: $MODEL_ID"
echo "  Port: $PORT"
echo "  Baseline: $BASELINE_TPS tok/s"
echo ""

# Check if server is already running
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "⚠️  Port $PORT is already in use. Stopping existing server..."
    pkill -f "omlx.server" || true
    sleep 2
fi

# Check if model is downloaded
echo "🔍 Checking model availability..."
MODEL_PATH="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-6bit"
if [ ! -d "$MODEL_PATH" ]; then
    echo "${RED}❌ Model not found at $MODEL_PATH${NC}"
    echo "Please download the model first:"
    echo "  python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')\""
    exit 1
fi
echo "${GREEN}✅ Model found${NC}"
echo ""

# Start oMLX server in background
echo "🚀 Starting oMLX server..."

# Create model directory structure for omlx
MODEL_DIR="$HOME/.cache/omlx_models"
mkdir -p "$MODEL_DIR"

# Create symlink to HuggingFace model snapshot
MODEL_NAME="Qwen3.5-35B-A3B-6bit"
SNAPSHOT_DIR=$(find "$MODEL_PATH/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)

if [ -n "$SNAPSHOT_DIR" ]; then
    rm -f "$MODEL_DIR/$MODEL_NAME"  # Remove old link if exists
    ln -sf "$SNAPSHOT_DIR" "$MODEL_DIR/$MODEL_NAME"
    echo "  Created symlink: $MODEL_DIR/$MODEL_NAME -> $SNAPSHOT_DIR"
else
    echo "${RED}❌ No snapshot found in $MODEL_PATH/snapshots${NC}"
    exit 1
fi

python3 -m omlx.server \
    --model-dir "$MODEL_DIR" \
    --default-model "$MODEL_NAME" \
    --port $PORT \
    --max-model-memory 32GB \
    > omlx_server.log 2>&1 &

SERVER_PID=$!
echo "  Server PID: $SERVER_PID"
echo "  Log file: omlx_server.log"

# Wait for server to start
echo "⏳ Waiting for server to initialize (30s)..."
for i in {1..30}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "${GREEN}✅ Server ready!${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo "${RED}❌ Server failed to start. Check omlx_server.log${NC}"
        kill $SERVER_PID 2>/dev/null || true
        tail -20 omlx_server.log
        exit 1
    fi
done
echo ""

# Run benchmark
echo "📊 Running benchmark (this may take 2-3 minutes)..."
echo ""

python3 benchmark_omlx.py > benchmark_output.log 2>&1 || {
    echo "${RED}❌ Benchmark failed. Check benchmark_output.log${NC}"
    kill $SERVER_PID 2>/dev/null || true
    tail -50 benchmark_output.log
    exit 1
}

# Parse results
echo "${GREEN}✅ Benchmark completed!${NC}"
echo ""
echo "========================================"
echo "RESULTS"
echo "========================================"

# Extract key metrics from benchmark output
cat benchmark_output.log | tail -30

echo ""
echo "========================================"

# Compare with baseline
if grep -q "Generation TPS" benchmark_output.log; then
    ACTUAL_TPS=$(grep "Generation TPS" benchmark_output.log | tail -1 | awk '{print $3}')
    SPEEDUP=$(python3 -c "print(f'{$ACTUAL_TPS / $BASELINE_TPS:.1f}x')")

    echo ""
    echo "📈 Performance Comparison:"
    echo "  Baseline:  $BASELINE_TPS tok/s"
    echo "  Actual:    $ACTUAL_TPS tok/s"
    echo "  Speedup:   ${GREEN}${SPEEDUP}${NC}"

    # Check if target achieved (4.2x)
    TARGET_TPS=500
    if (( $(echo "$ACTUAL_TPS >= $TARGET_TPS" | bc -l) )); then
        echo ""
        echo "${GREEN}🎉 TARGET ACHIEVED!${NC} Performance improvement: ≥ 4.2x"
    else
        echo ""
        echo "${YELLOW}⚠️  Below target ($TARGET_TPS tok/s)${NC}"
    fi
fi

# Cleanup
echo ""
echo "🧹 Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "${GREEN}✅ Benchmark complete!${NC}"
echo ""
echo "Full logs:"
echo "  Server: omlx_server.log"
echo "  Benchmark: benchmark_output.log"
