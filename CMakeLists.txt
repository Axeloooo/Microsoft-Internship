cmake_minimum_required(VERSION 3.23)
project(mccl LANGUAGES CXX CUDA)

# ---------------------------------------------------------------
#  Build options
# ---------------------------------------------------------------
option(MCCL_BUILD_TESTS     "Build GoogleTest-based unit tests" ON)
option(MCCL_BUILD_BENCHMARK "Build benchmarks"                  ON)

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
     ${PROJECT_SOURCE_DIR}/include/common/*.cpp
     ${PROJECT_SOURCE_DIR}/include/common/*.cu)


add_library(mccl_core STATIC ${SRC_FILES})

target_link_libraries(mccl_core
    PUBLIC CUDA::cudart CUDA::cublas MPI::MPI_CXX ${NCCL_LIB})

# Example compile definition often needed for half precision
target_compile_definitions(mccl_core PUBLIC __CUDA_NO_HALF_OPERATORS__ ENABLE_BF16)

# ---------------------------------------------------------------
#  Benchmarks (optional)
# ---------------------------------------------------------------
if(MCCL_BUILD_BENCHMARK)
    file(GLOB_RECURSE BENCH_SOURCES CONFIGURE_DEPENDS
         ${PROJECT_SOURCE_DIR}/benchmarks/**/*.cu)
    foreach(BFILE ${BENCH_SOURCES})
        get_filename_component(BNAME ${BFILE} NAME_WE)
        add_executable(${BNAME} ${BFILE})
        target_link_libraries(${BNAME} PRIVATE mccl_core)
        
        # Add include paths for MPI and custom kernels
        target_include_directories(${BNAME} PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src)
    endforeach()
endif()

# ---------------------------------------------------------------
#  Install (optional)
# ---------------------------------------------------------------
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
install(TARGETS mccl_core ARCHIVE DESTINATION lib)

# ---------------------------------------------------------------
#  Build summary
# ---------------------------------------------------------------
message(STATUS "------------------------------------------------------")
message(STATUS "  CUDA arch        : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build benchmarks : ${MCCL_BUILD_BENCHMARK}")
message(STATUS "  Build tests      : ${MCCL_BUILD_TESTS}")
message(STATUS "------------------------------------------------------")
