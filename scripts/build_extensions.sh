#!/bin/bash
# 编译 C++ 扩展（tensor_loader）

set -e  # 遇到错误立即退出

echo "=== 编译 ThunderOMLX C++ 扩展 ==="

# 进入扩展目录
cd "$(dirname "$0")/../src/omlx/extensions"

# 清理旧构建
rm -rf build
rm -f _tensor_loader*.so

# 创建构建目录
mkdir -p build
cd build

# 配置 CMake
echo "[1/3] 配置 CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native"

# 编译
echo "[2/3] 编译..."
cmake --build . --config Release -j$(sysctl -n hw.ncpu)

# 拷贝到 extensions 目录
echo "[3/3] 安装..."
cp _tensor_loader*.so ..

echo "✅ 编译完成！"
echo "   输出: src/omlx/extensions/_tensor_loader.*.so"
