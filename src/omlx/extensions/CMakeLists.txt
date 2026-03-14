cmake_minimum_required(VERSION 3.15)
project(_tensor_loader)

# C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 Python 和 pybind11
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# 创建 Python 扩展模块
pybind11_add_module(_tensor_loader tensor_loader.cpp)

# 包含 numpy 头文件
target_include_directories(_tensor_loader PRIVATE
    ${Python_NumPy_INCLUDE_DIRS}
)

# 优化标志（Release 模式）
target_compile_options(_tensor_loader PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
)

# 设置输出目录（与 omlx/extensions 同级）
set_target_properties(_tensor_loader PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# macOS 特定设置
if(APPLE)
    set_target_properties(_tensor_loader PROPERTIES
        INSTALL_RPATH "@loader_path"
    )
endif()
