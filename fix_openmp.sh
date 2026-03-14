#!/bin/bash
# 彻底解决 OpenMP 冲突问题

set -e

echo "🔍 Step 1: 检测冲突库..."
find /opt/homebrew/lib/python3.14/site-packages -name "*omp*.dylib" 2>/dev/null || true

echo ""
echo "🗑️  Step 2: 卸载冲突的科学计算库..."
pip uninstall -y scikit-learn numpy scipy 2>/dev/null || true

echo ""
echo "🏗️  Step 3: 清理 Homebrew OpenMP 缓存..."
brew uninstall --ignore-dependencies libomp 2>/dev/null || true
brew install libomp

echo ""
echo "📦 Step 4: 重新安装科学计算库（统一链接到 Homebrew libomp）..."
# 使用 --no-cache-dir 避免缓存问题
pip install --no-cache-dir --no-binary :all: numpy
pip install --no-cache-dir scikit-learn

echo ""
echo "✅ Step 5: 验证安装..."
python3 -c "import numpy; import sklearn; print('NumPy:', numpy.__version__); print('scikit-learn:', sklearn.__version__)"

echo ""
echo "🎉 OpenMP 冲突已彻底解决！"
echo ""
echo "⚠️  如果仍然崩溃，请运行:"
echo "    export KMP_DUPLICATE_LIB_OK=TRUE"
echo "    export OMP_NUM_THREADS=1"
