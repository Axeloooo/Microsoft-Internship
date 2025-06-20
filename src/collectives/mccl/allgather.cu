/**
 * @file allgather.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief AllGather example using CUDA IPC and MPI
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
constexpr std::size_t kBytes = (kN * sizeof(float));

/**
 * @brief Main function for the AllGather example using CUDA IPC and MPI.
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
    std::size_t total_bytes = kBytes * world_size;

    CUDA_CHECK(cudaSetDevice(world_rank));
    for (std::size_t rank = 0; rank < world_size; ++rank) {
        if (rank != world_rank) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(rank, 0));
        }
    }

    float* d_send = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send, kBytes));

    std::vector<float> h_init(kN);
    InitVector<float>(h_init.data(), kN);
    CUDA_CHECK(cudaMemcpy(d_send, h_init.data(), kBytes, cudaMemcpyHostToDevice));

    float* d_buff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buff, total_bytes));

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

    std::size_t offset_bytes = static_cast<std::size_t>(world_rank) * kBytes;

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char*>(d_buff) + offset_bytes, d_send, kBytes, cudaMemcpyDeviceToDevice, stream));

    for (std::size_t dst = 0; dst < world_size; ++dst) {
        if (dst != world_rank) {
            CUDA_CHECK(cudaMemcpyPeerAsync(reinterpret_cast<char*>(remotes[dst]) + offset_bytes, dst, d_send, world_rank, kBytes, stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> h_ref(world_size * kN);
    MPI_CHECK(MPI_Allgather(h_init.data(), kN, MPI_FLOAT, h_ref.data(), kN, MPI_FLOAT, MPI_COMM_WORLD));

    std::vector<float> h_check(world_size * kN);
    CUDA_CHECK(cudaMemcpy(h_check.data(), d_buff, total_bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (std::size_t i = 0; i < h_ref.size(); ++i) {
        if (std::fabs(h_check[i] - h_ref[i]) > 1e-5f) {
            ok = false;
            break;
        }
    }

    std::cout << "Rank " << world_rank << " : " << (ok ? "PASS" : "FAIL") << '\n';

    for (std::size_t r = 0; r < world_size; ++r) {
        if (remotes[r] != nullptr) {
            CUDA_CHECK(cudaIpcCloseMemHandle(remotes[r]));
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_send));
    CUDA_CHECK(cudaFree(d_buff));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
