#!/bin/bash
# 实时监控 Benchmark 性能数据

echo "🔍 监控 Admin Benchmark 性能数据..."
echo "按 Ctrl+C 停止监控"
echo ""

# 监控后台任务输出
tail -f /private/tmp/claude-501/-Users-lisihao/tasks/b1741cf.output 2>/dev/null | while IFS= read -r line
do
    # 高亮性能数据
    if echo "$line" | grep -q "⏱️"; then
        echo -e "\033[1;32m$line\033[0m"  # 绿色高亮
    elif echo "$line" | grep -q "✅"; then
        echo -e "\033[1;34m$line\033[0m"  # 蓝色高亮
    elif echo "$line" | grep -q "Test:"; then
        echo -e "\033[1;33m$line\033[0m"  # 黄色高亮
    else
        echo "$line"
    fi
done
