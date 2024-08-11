cmake_minimum_required(VERSION 3.14)
project(SimpleML)

# 设置目标架构为 x86_64（在 macOS 上）
set(CMAKE_OSX_ARCHITECTURES "x86_64")

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译器选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-undefined,dynamic_lookup")

# 找到 pybind11 包
find_package(pybind11 REQUIRED)

# 添加目标库
add_library(simple_ml_ext MODULE src/simple_ml_ext.cpp)

# 将库链接到 pybind11
target_link_libraries(simple_ml_ext PRIVATE pybind11::module)

# 设置生成文件的输出路径
set_target_properties(simple_ml_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src
)