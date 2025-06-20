/**
 * @file reduce.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Reduce example using MPI and CUDA IPC
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
 * @brief Main function for the MPI and CUDA IPC reduce example
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
    for (std::size_t rank = 0; rank < world_size; ++rank) {
        if (rank != world_rank) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(rank, 0));
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

    std::vector<float> h_ref(kN, 0.f);
    MPI_CHECK(MPI_Reduce(h_init.data(), h_ref.data(), static_cast<int>(kN), MPI_FLOAT, MPI_SUM, kRootGpu, MPI_COMM_WORLD));

    if (world_rank == kRootGpu) {
        float* d_tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_tmp, kBytes));

        for (std::size_t src = 0; src < world_size; ++src) {
            if (src != kRootGpu) {
                CUDA_CHECK(cudaMemcpyPeerAsync(d_tmp, kRootGpu, remotes[src], src, kBytes, stream));

                VecAddInPlace<<<kGridDim, kBlockDim, 0, stream>>>(d_buff, d_tmp, kN);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_tmp));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    bool ok = true;
    if (world_rank == kRootGpu) {
        std::vector<float> h_result(kN);
        CUDA_CHECK(cudaMemcpy(h_result.data(), d_buff, kBytes, cudaMemcpyDeviceToHost));

        for (std::size_t i = 0; i < kN; ++i)
            if (std::fabs(h_result[i] - h_ref[i]) > 1e-5f) {
                ok = false;
                break;
            }

        std::cout << "Root rank " << world_rank << " : " << (ok ? "PASS" : "FAIL") << std::endl;
    }

    for (int rank = 0; rank < world_size; ++rank) {
        if (remotes[rank] != nullptr) {
            CUDA_CHECK(cudaIpcCloseMemHandle(remotes[rank]));
        };
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_buff));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
