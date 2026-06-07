cmake_minimum_required(VERSION 4.0) # Note: 3.21+ is needed for TARGET_RUNTIME_DLLS
project(norm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)

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
target_compile_options(cuda_kernels PRIVATE -Xnvlink -use-host-info -gencode arch=compute_121,code=lto_121 --enable-tile --gen-opt-lto -t 6 --use_fast_math --prune)

# Create exe target
add_executable(perturb ${SOURCES} main.cpp)
set_target_properties(perturb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link the cuda kernels
target_link_libraries(perturb PRIVATE cuda_kernels)

# Configure include dirs
target_include_directories(perturb PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
target_include_directories(cuda_kernels PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Add C++23 compile features
target_compile_features(perturb PRIVATE cxx_std_23)

# CUDA Configuration
set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2")
list(APPEND CMAKE_PREFIX_PATH "${CUDA_ROOT}\\lib\\cmake")
target_include_directories(perturb PRIVATE "${CUDA_ROOT}/include")
target_link_directories(perturb PRIVATE "${CUDA_ROOT}/lib/x64")

# Link against CUDA libraries
target_link_libraries(perturb PRIVATE nppim nppif nppitc nppisu nppicc nppist cuda cudart_static nppial)

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
target_include_directories(perturb PRIVATE ${stb_SOURCE_DIR})

# TinyEXR - FetchContent
FetchContent_Declare(
        tinyexr
        GIT_REPOSITORY
        "https://github.com/syoyo/tinyexr.git"
        GIT_TAG
        "v1.0.13"
)

FetchContent_MakeAvailable(tinyexr)
target_include_directories(perturb PRIVATE ${tinyexr_SOURCE_DIR} "${tinyexr_SOURCE_DIR}/deps/miniz")

# Copy necessary DLLs to the output directory
if (WIN32)
    add_custom_command(TARGET perturb POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUDA_ROOT}/bin/x64/cudart64_13.dll"
            $<TARGET_FILE_DIR:perturb>)
endif ()
