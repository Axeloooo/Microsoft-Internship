/**
 * @file reducescatter.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Example of ReduceScatter operation using MCCL library
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
 * @brief Main function to demonstrate ReduceScatter operation using MCCL library
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

    const std::size_t sendcount = kN;
    const int recvcount = kN / world_size;
    const std::size_t sendbytes = sendcount * sizeof(float);
    const std::size_t recvbytes = recvcount * sizeof(float);

    float *d_sendbuff = nullptr;
    float *d_recvbuff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sendbuff, sendbytes));
    CUDA_CHECK(cudaMalloc(&d_recvbuff, recvbytes));

    float *h_in = new float[sendcount];
    InitVector<float>(h_in, sendcount);
    CUDA_CHECK(cudaMemcpy(d_sendbuff, h_in, sendbytes, cudaMemcpyHostToDevice));

    MCCL<float> mccl(world_rank, world_size);
    const Operation op = SUM;
    mccl.ReduceScatter(d_sendbuff, d_recvbuff, recvcount, op);

    float *h_result = new float[recvcount];
    CUDA_CHECK(cudaMemcpy(h_result, d_recvbuff, recvbytes, cudaMemcpyDeviceToHost));

    float *h_expected = new float[recvcount];
    MPI_Reduce_scatter(h_in, h_expected, &recvcount, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    bool success = true;
    for (std::size_t i = 0; i < recvcount; i++) {
        if (std::fabs(h_result[i] - h_expected[i]) > 1e-4f) {
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