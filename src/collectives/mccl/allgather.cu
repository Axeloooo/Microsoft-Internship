/**
 * @file allgather.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Example of AllGather operation using MCCL
 * @version 0.1
 * @date 2025-06-13
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>
#include <mpi.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "checker.cuh"
#include "data.cuh"
#include "mccl.cuh"

constexpr std::size_t kN = 4194304;
constexpr std::size_t kBytes = (kN * sizeof(float));

/**
 * @brief Main function to demonstrate AllGather operation using MCCL
 *
 * @param argc argument count
 * @param argv argument vector
 * @return int
 */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    CUDA_CHECK(cudaSetDevice(world_rank));

    for (int i = 0; i < world_size; ++i) {
        if (i != world_rank) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(i, 0));
        }
    }

    float *d_sendbuff = nullptr;
    float *d_recvbuff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sendbuff, kBytes));
    CUDA_CHECK(cudaMalloc(&d_recvbuff, kBytes * world_size));

    float *h_in = new float[kN];
    InitVector<float>(h_in, kN);
    CUDA_CHECK(cudaMemcpy(d_sendbuff, h_in, kBytes, cudaMemcpyHostToDevice));

    MCCL<float> mccl(world_rank, world_size);
    mccl.AllGather(d_sendbuff, d_recvbuff, kN);

    float *h_result = new float[kN * world_size];
    CUDA_CHECK(cudaMemcpy(h_result, d_recvbuff, kBytes * world_size, cudaMemcpyDeviceToHost));

    float *h_expected = new float[kN * world_size];
    MPI_Allgather(h_in, kN, MPI_FLOAT, h_expected, kN, MPI_FLOAT, MPI_COMM_WORLD);

    bool success = true;
    for (size_t i = 0; i < kN * world_size; i++) {
        if (std::fabs(h_result[i] - h_expected[i]) > 1e-5f) {
            success = false;
            break;
        }
    }
    std::cout << "Device " << world_rank << ": " << (success ? "PASS" : "FAIL") << std::endl;

    delete[] h_in;
    delete[] h_result;
    delete[] h_expected;
    CUDA_CHECK(cudaFree(d_sendbuff));
    CUDA_CHECK(cudaFree(d_recvbuff));

    MPI_Finalize();
    return 0;
}
