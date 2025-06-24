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
#include "mccl.cuh"

constexpr std::size_t kN = 4194304;
constexpr std::size_t kBytes = kN * sizeof(float);
constexpr std::size_t kRootGpu = 0;

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

    float *d_send = nullptr, *d_recv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send, kBytes));
    CUDA_CHECK(cudaMalloc(&d_recv, kBytes));

    std::vector<float> h_init(kN);
    InitVector<float>(h_init.data(), kN);
    CUDA_CHECK(cudaMemcpy(d_send, h_init.data(), kBytes, cudaMemcpyHostToDevice));

    MCCL<float> mccl(world_rank, world_size);
    mccl.Reduce(d_send, d_recv, kN, kRootGpu, Operation::SUM);

    std::vector<float> h_ref(kN);
    MPI_CHECK(MPI_Reduce(h_init.data(), h_ref.data(), static_cast<int>(kN), MPI_FLOAT, MPI_SUM, kRootGpu, MPI_COMM_WORLD));

    if (world_rank == kRootGpu) {
        std::vector<float> h_res(kN);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_recv, kBytes, cudaMemcpyDeviceToHost));
        bool ok = true;
        for (std::size_t i = 0; i < kN; ++i)
            if (std::fabs(h_res[i] - h_ref[i]) > 1e-5f) {
                ok = false;
                break;
            }
        std::cout << "Root rank " << world_rank << " : " << (ok ? "PASS" : "FAIL") << std::endl;
    }

    CUDA_CHECK(cudaFree(d_send));
    CUDA_CHECK(cudaFree(d_recv));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
