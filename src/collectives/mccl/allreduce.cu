/**
 * @file allreduce.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief AllReduce example using CUDA IPC and MPI
 * @version 0.1
 * @date 2025-06-13
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "checker.cuh"
#include "data.cuh"

constexpr std::size_t kN = 4194304;
constexpr std::size_t kBytes = kN * sizeof(float);
constexpr std::size_t kRootGpu = 0;
constexpr std::size_t kBlockDim = 256;
constexpr std::size_t kGridDim = (kN + kBlockDim - 1) / kBlockDim;

/**
 * @brief Kernel to add two vectors in place
 *
 * @param dst Pointer to the destination vector
 * @param src Pointer to the source vector
 * @param n Number of elements in the vectors
 * @return __global__
 */
__global__ void VecAddInPlace(float* dst, const float* src, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] += src[i];
    }
}

/**
 * @brief Main function for the AllReduce example using CUDA IPC and MPI
 *
 * @param argc argument count
 * @param argv argument vector
 * @return int
 */
int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_size, world_rank;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    CUDA_CHECK(cudaSetDevice(world_rank));
    for (std::size_t i = 0; i < world_size; ++i) {
        if (i != world_rank) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(i, 0));
        }
    }
    float* d_buff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buff, kBytes));

    cudaIpcMemHandle_t handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, d_buff));

    std::vector<cudaIpcMemHandle_t> locals(world_size);
    MPI_CHECK(MPI_Allgather(&handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, locals.data(), sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD));

    std::vector<float*> remotes(world_size, nullptr);
    for (std::size_t rank = 0; rank < world_size; ++rank) {
        if (rank != world_rank) {
            CUDA_CHECK(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&remotes[rank]), locals[rank], cudaIpcMemLazyEnablePeerAccess));
        }
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<float> h_init(kN);
    InitVector<float>(h_init.data(), kN);
    CUDA_CHECK(cudaMemcpy(d_buff, h_init.data(), kBytes, cudaMemcpyHostToDevice));

    if (world_rank == kRootGpu) {
        float* d_tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_tmp, kBytes));

        for (std::size_t src = 0; src < world_size; ++src) {
            if (src != kRootGpu) {
                CUDA_CHECK(cudaMemcpyPeerAsync(d_tmp, kRootGpu, remotes[src], src, kBytes, stream));
            }
            VecAddInPlace<<<kGridDim, kBlockDim, 0, stream>>>(d_buff, d_tmp, kN);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_tmp));

        for (std::size_t dst = 0; dst < world_size; ++dst) {
            if (dst != kRootGpu) {
                CUDA_CHECK(cudaMemcpyPeerAsync(remotes[dst], dst, d_buff, kRootGpu, kBytes, stream));
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    std::vector<float> h_ref(kN);
    if (world_rank == kRootGpu) {
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_buff, kBytes, cudaMemcpyDeviceToHost));
    }
    MPI_CHECK(MPI_Bcast(h_ref.data(), kN, MPI_FLOAT, kRootGpu, MPI_COMM_WORLD));

    std::vector<float> h_check(kN);
    CUDA_CHECK(cudaMemcpy(h_check.data(), d_buff, kBytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (std::size_t i = 0; i < kN; ++i) {
        if (std::fabs(h_check[i] - h_ref[i]) > 1e-5f) {
            ok = false;
            break;
        }
    }
    std::cout << "Rank " << world_rank << " : " << (ok ? "PASS" : "FAIL") << std::endl;

    for (std::size_t rank = 0; rank < world_size; ++rank) {
        if (remotes[rank] != nullptr) {
            CUDA_CHECK(cudaIpcCloseMemHandle(remotes[rank]));
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_buff));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
