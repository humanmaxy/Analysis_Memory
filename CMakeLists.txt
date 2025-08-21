cmake_minimum_required(VERSION 3.10)
project(XRayEnhancement)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 包含OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(xray_enhancement 
    main.cpp 
    xray_enhancement.cpp
)

# 链接OpenCV库
target_link_libraries(xray_enhancement ${OpenCV_LIBS})

# 编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(xray_enhancement PRIVATE -Wall -Wextra -O3)
endif()

# 如果需要GUI支持，添加以下定义
option(ENABLE_GUI "Enable GUI support" OFF)
if(ENABLE_GUI)
    target_compile_definitions(xray_enhancement PRIVATE OPENCV_GUI_ENABLED)
endif()

# 安装
install(TARGETS xray_enhancement DESTINATION bin)