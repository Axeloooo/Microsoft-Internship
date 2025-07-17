cmake_minimum_required(VERSION 3.23)
project(trt_allreduce LANGUAGES CXX CUDA)

# ---------------------------------------------------------------
#  Build options
# ---------------------------------------------------------------
option(TRT_ALLREDUCE_BUILD "Build benchmarks"                  ON)

# ---------------------------------------------------------------
#  Language / toolchain settings
# ---------------------------------------------------------------
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

# Default to Ada Lovelace (sm_90); override with -DCMAKE_CUDA_ARCHITECTURES=xx
set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "CUDA arch list")

# ---------------------------------------------------------------
#  Dependencies
# ---------------------------------------------------------------
find_package(MPI          REQUIRED)
find_package(CUDAToolkit  REQUIRED)
# Find CUDA driver API library for multicas and driver calls
find_library(CUDA_DRIVER_LIBRARY NAMES cuda REQUIRED)
find_library(NCCL_LIB NAMES nccl REQUIRED)

# ---------------------------------------------------------------
#  Include directories
# ---------------------------------------------------------------
include_directories(${PROJECT_SOURCE_DIR}/include)

# ---------------------------------------------------------------
#  Library target
# ---------------------------------------------------------------
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
     ${PROJECT_SOURCE_DIR}/src/*.cu
     ${PROJECT_SOURCE_DIR}/src/*.cpp
     ${PROJECT_SOURCE_DIR}/include/common/*.cpp
     ${PROJECT_SOURCE_DIR}/include/common/*.cu
     ${PROJECT_SOURCE_DIR}/include/runtime/*.cpp
     ${PROJECT_SOURCE_DIR}/include/runtime/utils/*.cpp
     ${PROJECT_SOURCE_DIR}/include/runtime/*.cu
     ${PROJECT_SOURCE_DIR}/include/runtime/utils/*.cu)

add_library(trt_allreduce STATIC ${SRC_FILES})

target_link_libraries(trt_allreduce PUBLIC
    CUDA::cudart
    CUDA::cublas
    MPI::MPI_CXX
    ${NCCL_LIB}
    ${CUDA_DRIVER_LIBRARY})

# Example compile definition often needed for half precision
target_compile_definitions(trt_allreduce PUBLIC __CUDA_NO_HALF_OPERATORS__ ENABLE_BF16 ENABLE_MULTI_DEVICE ENABLE_FP8)

# Add CUDA compiler flags for relaxed constexpr
target_compile_options(trt_allreduce PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Ensure all executables go to the build root for easy access
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# ---------------------------------------------------------------
#  Benchmarks (optional)
# ---------------------------------------------------------------
if(TRT_ALLREDUCE_BUILD)
    # Glob all .cu files in benchmarks recursively
    file(GLOB_RECURSE BENCH_SOURCES CONFIGURE_DEPENDS
         ${PROJECT_SOURCE_DIR}/benchmarks/*.cu)
    foreach(BFILE ${BENCH_SOURCES})
        get_filename_component(BNAME ${BFILE} NAME_WE)
        add_executable(${BNAME} ${BFILE})
        target_link_libraries(${BNAME} PRIVATE trt_allreduce)
        # Add include paths for MPI and custom kernels
        target_include_directories(${BNAME} PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src)
    endforeach()
endif()

# ---------------------------------------------------------------
#  Install (optional)
# ---------------------------------------------------------------
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
install(TARGETS trt_allreduce ARCHIVE DESTINATION lib)

# ---------------------------------------------------------------
#  Build summary
# ---------------------------------------------------------------
message(STATUS "------------------------------------------------------")
message(STATUS "  CUDA arch        : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build benchmarks : ${TRT_ALLREDUCE_BUILD}")
message(STATUS "------------------------------------------------------")
