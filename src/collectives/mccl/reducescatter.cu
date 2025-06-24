/**
 * @file reducescatter.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief ReduceScatter example using CUDA IPC and MPI
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

/**
 * @brief Main function for the ReduceScatter example using CUDA IPC and MPI
 *
 * @param argc argument count
 * @param argv argument vector
 * @return int
 */
int main(int argc, char **argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    CUDA_CHECK(cudaSetDevice(world_rank));
    for (std::size_t rank = 0; rank < world_size; ++rank) {
        if (rank != world_rank) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(rank, 0));
        }
    }

    const int recvcount = kN / world_size;
    const std::size_t sendbytes = kN * sizeof(float);
    const std::size_t recvbytes = recvcount * sizeof(float);

    float *d_sendbuff = nullptr, *d_recvbuff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sendbuff, sendbytes));
    CUDA_CHECK(cudaMalloc(&d_recvbuff, recvbytes));

    std::vector<float> h_in(kN);
    InitVector<float>(h_in.data(), kN);
    CUDA_CHECK(cudaMemcpy(d_sendbuff, h_in.data(), sendbytes, cudaMemcpyHostToDevice));

    MCCL<float> mccl(world_rank, world_size);
    mccl.ReduceScatter(d_sendbuff, d_recvbuff, recvcount, Operation::SUM);

    std::vector<float> h_result(recvcount);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_recvbuff, recvbytes, cudaMemcpyDeviceToHost));

    std::vector<int> recvcounts(world_size, recvcount);
    std::vector<float> h_expected(recvcount);
    MPI_CHECK(MPI_Reduce_scatter(h_in.data(), h_expected.data(), recvcounts.data(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));

    bool success = true;
    for (std::size_t i = 0; i < recvcount; ++i) {
        if (std::fabs(h_result[i] - h_expected[i]) > 1e-4f) {
            success = false;
            break;
        }
    }
    std::cout << "Device " << world_rank << " : " << (success ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaFree(d_sendbuff));
    CUDA_CHECK(cudaFree(d_recvbuff));
    MPI_CHECK(MPI_Finalize());
    return 0;
}