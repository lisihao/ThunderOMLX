#!/bin/bash
# ThunderOMLX 开发环境一键配置脚本

set -e

echo "🚀 ThunderOMLX 开发环境配置"
echo "================================"

# 1. 检查 Python 版本
echo ""
echo "Step 1: 检查 Python 版本..."
python3 --version | grep "3.1[1-9]" || {
    echo "❌ 需要 Python 3.11+，请升级"
    exit 1
}
echo "✅ Python 版本满足要求"

# 2. 创建虚拟环境
echo ""
echo "Step 2: 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ 虚拟环境已创建"
else
    echo "⏭️  虚拟环境已存在"
fi

# 3. 激活虚拟环境
echo ""
echo "Step 3: 激活虚拟环境..."
source venv/bin/activate
echo "✅ 虚拟环境已激活"

# 4. 升级 pip
echo ""
echo "Step 4: 升级 pip..."
pip install --upgrade pip > /dev/null
echo "✅ pip 已升级"

# 5. 安装依赖
echo ""
echo "Step 5: 安装依赖..."
pip install -r requirements.txt > /dev/null
echo "✅ 核心依赖已安装"

# 6. 安装 omlx（开发模式）
echo ""
echo "Step 6: 安装 omlx (editable)..."
cd src && pip install -e . > /dev/null && cd ..
echo "✅ omlx 已安装（开发模式）"

# 7. 检查 ThunderLLAMA
echo ""
echo "Step 7: 检查 ThunderLLAMA..."
if [ -f "$HOME/ThunderLLAMA/build/bin/llama-server" ]; then
    echo "✅ ThunderLLAMA llama-server 已编译"
else
    echo "⚠️  ThunderLLAMA llama-server 未找到"
    echo "   请先编译 ThunderLLAMA:"
    echo "   cd ~/ThunderLLAMA && mkdir build && cd build"
    echo "   cmake .. -DGGML_METAL=ON && cmake --build . --config Release"
fi

# 8. 创建模型目录
echo ""
echo "Step 8: 创建模型目录..."
mkdir -p ~/Models
echo "✅ 模型目录已创建: ~/Models"

# 9. 完成
echo ""
echo "================================"
echo "✅ 开发环境配置完成！"
echo ""
echo "下一步:"
echo "  1. 激活虚拟环境: source venv/bin/activate"
echo "  2. 下载测试模型到 ~/Models/"
echo "  3. 启动 llama-server"
echo "  4. 开始开发 🚀"
echo ""
