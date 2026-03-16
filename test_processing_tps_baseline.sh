#!/bin/bash
# Processing TPS Baseline Test
#
# 匹配 PHASE1_2_ANALYSIS.md 中的原始基准场景：
# - 并发 Agent 请求
# - 大量 cache 操作
# - 测量 Processing TPS（包含 cleanup）
#
# 目标：验证从 692.7 tok/s → 730 tok/s（+5.4%）

set -e

echo "========================================================================"
echo "🎯 Processing TPS Baseline Test - Phase 1-4 Validation"
echo "========================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
MODEL_PATH="$HOME/models/qwen3.5-35b-mlx"
PORT=8000
BASELINE_TPS=692.7
TARGET_TPS=730.0

echo "📋 Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Port: $PORT"
echo "  Baseline: $BASELINE_TPS tok/s"
echo "  Target: $TARGET_TPS tok/s (+5.4%)"
echo ""

# Environment setup
export PYTHONPATH=/Users/lisihao/ThunderOMLX/src:$PYTHONPATH
export KMP_DUPLICATE_LIB_OK=TRUE

# Check if server is already running
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "⚠️  Port $PORT is in use. Stopping existing server..."
    pkill -f "omlx.server" || true
    sleep 2
fi

# Check model
if [ ! -d "$MODEL_PATH" ]; then
    echo "${RED}❌ Model not found: $MODEL_PATH${NC}"
    exit 1
fi
echo "${GREEN}✅ Model found${NC}"
echo ""

# Start oMLX server
echo "🚀 Starting oMLX server..."
echo "  Model: $MODEL_PATH"
echo "  Port: $PORT"

# Create model directory structure
MODEL_DIR="$HOME/.cache/omlx_models"
mkdir -p "$MODEL_DIR"

# Create symlink to model
MODEL_NAME="qwen3.5-35b-mlx"
rm -f "$MODEL_DIR/$MODEL_NAME"
ln -sf "$MODEL_PATH" "$MODEL_DIR/$MODEL_NAME"
echo "  Model dir: $MODEL_DIR"
echo "  Model link: $MODEL_NAME -> $MODEL_PATH"

python3 -m omlx.server \
    --model-dir "$MODEL_DIR" \
    --default-model "$MODEL_NAME" \
    --port $PORT \
    --host 127.0.0.1 \
    --max-model-memory 32GB \
    > /tmp/omlx_processing_tps_server.log 2>&1 &

SERVER_PID=$!
echo "  Server PID: $SERVER_PID"
echo "  Log: /tmp/omlx_processing_tps_server.log"

# Wait for server to start
echo ""
echo "⏳ Waiting for server to initialize..."
for i in {1..60}; do
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        echo "${GREEN}✅ Server ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
    if [ $i -eq 60 ]; then
        echo ""
        echo "${RED}❌ Server failed to start${NC}"
        kill $SERVER_PID 2>/dev/null || true
        tail -50 /tmp/omlx_processing_tps_server.log
        exit 1
    fi
done
echo ""

# Give server extra time to fully initialize
echo "⏳ Allowing server to fully initialize (10s)..."
sleep 10
echo ""

# Run benchmark
echo "📊 Running Processing TPS benchmark..."
echo "  Scenario: 4 concurrent requests (Agent scenario)"
echo "  Expected: Save_block operations triggered"
echo "  Measuring: Total walltime including cleanup"
echo ""

python3 benchmark_omlx.py > /tmp/processing_tps_benchmark.log 2>&1 || {
    echo "${RED}❌ Benchmark failed${NC}"
    kill $SERVER_PID 2>/dev/null || true
    tail -100 /tmp/processing_tps_benchmark.log
    exit 1
}

# Parse results
echo "${GREEN}✅ Benchmark completed!${NC}"
echo ""

# Extract metrics
echo "========================================================================"
echo "📈 RESULTS"
echo "========================================================================"
echo ""

# Show last 40 lines (contains all results)
tail -40 /tmp/processing_tps_benchmark.log

echo ""
echo "========================================================================"
echo "📊 Performance Analysis"
echo "========================================================================"
echo ""

# Extract Generation TPS (Agent scenario)
if grep -q "Agent Scenario" /tmp/processing_tps_benchmark.log; then
    ACTUAL_TPS=$(grep "Generation TPS:" /tmp/processing_tps_benchmark.log | tail -1 | awk '{print $3}')

    if [ -n "$ACTUAL_TPS" ]; then
        # Calculate improvement
        IMPROVEMENT=$(python3 -c "print(f'{($ACTUAL_TPS - $BASELINE_TPS) / $BASELINE_TPS * 100:.1f}')")
        SPEEDUP=$(python3 -c "print(f'{$ACTUAL_TPS / $BASELINE_TPS:.3f}')")

        echo "${BLUE}Processing TPS Comparison:${NC}"
        echo "  Baseline (Phase 0):  $BASELINE_TPS tok/s"
        echo "  Current (Phase 1-4): ${GREEN}$ACTUAL_TPS tok/s${NC}"
        echo "  Improvement:         ${GREEN}+$IMPROVEMENT%${NC} (${SPEEDUP}x)"
        echo "  Target:              $TARGET_TPS tok/s (+5.4%)"
        echo ""

        # Check if target achieved
        if (( $(echo "$ACTUAL_TPS >= $TARGET_TPS" | bc -l) )); then
            echo "${GREEN}🎉 TARGET ACHIEVED!${NC}"
            echo "   Phase 1-4 optimizations successfully improved Processing TPS"
        else
            SHORTFALL=$(python3 -c "print(f'{$TARGET_TPS - $ACTUAL_TPS:.1f}')")
            echo "${YELLOW}⚠️  Below target by ${SHORTFALL} tok/s${NC}"
            echo "   Target: ≥$TARGET_TPS tok/s"
            echo "   Actual: $ACTUAL_TPS tok/s"
        fi
    else
        echo "${YELLOW}⚠️  Could not parse TPS from benchmark output${NC}"
    fi
else
    echo "${YELLOW}⚠️  Agent scenario results not found in output${NC}"
fi

echo ""
echo "========================================================================"
echo "📋 Log Files"
echo "========================================================================"
echo "  Server log:    /tmp/omlx_processing_tps_server.log"
echo "  Benchmark log: /tmp/processing_tps_benchmark.log"
echo ""

# Cleanup
echo "🧹 Stopping server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "${GREEN}✅ Test complete!${NC}"
