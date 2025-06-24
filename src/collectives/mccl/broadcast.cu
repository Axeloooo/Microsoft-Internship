/**
 * @file broadcast.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Broadcast example using CUDA IPC and MPI
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
#include "mccl.cuh"

constexpr std::size_t kN = 4194304;
constexpr std::size_t kBytes = (kN * sizeof(float));
constexpr std::size_t kRootGpu = 0;

/**
 * @brief Main function for the broadcast example using CUDA IPC and MPI
 *
 * @param argc argument count
 * @param argv argument vector
 * @return int
 */
int main(int argc, char **argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_size;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    int world_rank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    CUDA_CHECK(cudaSetDevice(world_rank));

    for (std::size_t rank = 0; rank < world_size; ++rank) {
        if (rank != world_rank) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(rank, 0));
        }
    }

    float *d_buff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buff, kBytes));

    if (world_rank == kRootGpu) {
        std::vector<float> h(kN);
        InitVector<float>(h.data(), kN);
        CUDA_CHECK(cudaMemcpy(d_buff, h.data(), kBytes, cudaMemcpyHostToDevice));
    }

    MCCL<float> mccl(world_rank, world_size);
    mccl.Broadcast(d_buff, kN, kRootGpu);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    std::vector<float> h_ref(kN);
    if (world_rank == kRootGpu) {
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_buff, kBytes, cudaMemcpyDeviceToHost));
    }
    MPI_CHECK(MPI_Bcast(h_ref.data(), static_cast<int>(kN), MPI_FLOAT, kRootGpu, MPI_COMM_WORLD));

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

    CUDA_CHECK(cudaFree(d_buff));
    MPI_CHECK(MPI_Finalize());
    return 0;
}