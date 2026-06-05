cmake_minimum_required(VERSION 3.21) # Note: 3.21+ is needed for TARGET_RUNTIME_DLLS
project(norm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)

# Find files
file(GLOB_RECURSE SOURCES
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/include/*.h"
)

# Find CUDA kernels
file(GLOB_RECURSE CUDA_KERNELS
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu")

# Create CUDA kernels target
add_library(cuda_kernels STATIC "${CUDA_KERNELS}")
set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Language Level
target_compile_features(cuda_kernels PUBLIC cxx_std_23)

# Newer features
target_compile_options(cuda_kernels PRIVATE -Xnvlink -use-host-info -gencode arch=compute_80,code=lto_80)

# Create exe target
add_executable(norm ${SOURCES} main.cpp)
set_target_properties(norm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link the cuda kernels
target_link_libraries(norm PRIVATE cuda_kernels)

# Configure include dirs
target_include_directories(norm PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
target_include_directories(cuda_kernels PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Add C++17 compile features
target_compile_features(norm PRIVATE cxx_std_23)

# CUDA Configuration
set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2")
list(APPEND CMAKE_PREFIX_PATH "${CUDA_ROOT}\\lib\\cmake")
target_include_directories(norm PRIVATE "${CUDA_ROOT}/include")
target_link_directories(norm PRIVATE "${CUDA_ROOT}/lib/x64")

# Link against CUDA libraries
target_link_libraries(norm PRIVATE npps nppim nppif nppitc nppisu nppicc nppist cuda cudart_static nppial)

# STB - FetchContent
include(FetchContent)
FetchContent_Declare(
        stb
        GIT_REPOSITORY
        "https://github.com/nothings/stb.git"
        GIT_TAG
        "master"
)

FetchContent_MakeAvailable(stb)
target_include_directories(norm PRIVATE ${stb_SOURCE_DIR})

# Copy necessary DLLs to the output directory
if (WIN32)
    add_custom_command(TARGET norm POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUDA_ROOT}/bin/x64/cudart64_13.dll"
#            "${CUDA_ROOT}/bin/x64/cublas64_13.dll"
#            "${CUDA_ROOT}/bin/x64/cusolver64_12.dll"
#            "${CUDA_ROOT}/bin/x64/cusparse64_12.dll"
            $<TARGET_FILE_DIR:norm>)
endif ()
