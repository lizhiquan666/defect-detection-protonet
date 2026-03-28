cmake_minimum_required(VERSION 3.18)
project(DefectDetect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(/utf-8)

# Portable dependency roots (env first, workspace fallback)
set(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/..")

set(LIBTORCH_ROOT "$ENV{LIBTORCH_ROOT}")
if (NOT LIBTORCH_ROOT)
    set(LIBTORCH_ROOT "${PROJECT_ROOT}/extra/libtorch")
endif()
set(CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")

set(OPENCV_ROOT "$ENV{OPENCV_ROOT}")
if (NOT OPENCV_ROOT)
    set(OPENCV_ROOT "${PROJECT_ROOT}/extra/opencv/build")
endif()

# CUDA toolkit hints for VS Code CMake configure (without extra -D args)
if (WIN32)
    if (NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
        if (DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}")
            set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_PATH}" CACHE PATH "CUDA Toolkit Root" FORCE)
        endif()
    endif()

    if (DEFINED CUDA_TOOLKIT_ROOT_DIR AND EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")
        set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()

    if (NOT CMAKE_CUDA_FLAGS MATCHES "allow-unsupported-compiler")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler" CACHE STRING "CUDA flags" FORCE)
    endif()
endif()

find_package(Torch REQUIRED)

# OpenCV
include_directories("${OPENCV_ROOT}/include")
link_directories("${OPENCV_ROOT}/x64/vc16/lib")

# ==================== 生成两个独立可执行文件 ====================
add_executable(defect_detect 
    main.cpp
    prnet.cpp
    attention.cpp
    dataset.cpp
    losses.cpp
    modules.cpp
    utils.cpp
)

add_executable(prototype_generator 
    prototype.cpp
)

target_link_libraries(defect_detect 
    ${TORCH_LIBRARIES} 
    opencv_world4120d.lib
)

target_link_libraries(prototype_generator 
    ${TORCH_LIBRARIES} 
    opencv_world4120d.lib
)

target_compile_definitions(defect_detect PRIVATE 
    _CRT_SECURE_NO_WARNINGS 
    TORCH_USE_CXX11_ABI=0
)

target_compile_definitions(prototype_generator PRIVATE 
    _CRT_SECURE_NO_WARNINGS 
    TORCH_USE_CXX11_ABI=0
)

message(STATUS " 已生成两个目标：defect_detect.exe 和 prototype_generator.exe")