#!/bin/bash

# OpenClaw 负载数据集演示脚本
# 一键生成负载、运行测试、查看报告

set -e

echo "======================================================================="
echo "OpenClaw 负载数据集演示"
echo "======================================================================="
echo ""

# Step 1: 生成负载数据
echo "Step 1: 生成 7 天模拟负载数据..."
echo "-----------------------------------------------------------------------"
python3 scripts/generate_openclaw_workload.py
echo ""

# Step 2: 运行 Phase 3 测试
echo "Step 2: 运行 Phase 3 高级策略测试..."
echo "-----------------------------------------------------------------------"
python3 scripts/test_phase3_with_openclaw_workload.py
echo ""

# Step 3: 显示测试报告摘要
echo "Step 3: 测试报告摘要"
echo "-----------------------------------------------------------------------"
echo ""
echo "📊 关键结果:"
echo ""
grep -A 15 "## 3.4 优化效果" docs/phase3-openclaw-workload-test-report.md || true
echo ""
echo "📈 ROI 估算:"
echo ""
grep -A 10 "### 6.2 成本节省" docs/phase3-openclaw-workload-test-report.md || true
echo ""

# Step 4: 提示下一步
echo "======================================================================="
echo "✅ 演示完成！"
echo "======================================================================="
echo ""
echo "📁 生成的文件:"
echo "   - openclaw-workload/openclaw-workload-7d.jsonl (2209 条记录)"
echo "   - openclaw-workload/metadata.json"
echo "   - docs/phase3-openclaw-workload-test-report.md"
echo ""
echo "📖 查看完整报告:"
echo "   cat docs/phase3-openclaw-workload-test-report.md"
echo ""
echo "🔧 自定义负载:"
echo "   编辑 scripts/generate_openclaw_workload.py"
echo "   修改 agent_profiles 配置"
echo ""
