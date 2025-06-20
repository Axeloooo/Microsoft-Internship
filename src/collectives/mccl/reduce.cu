/**
 * @file reduce.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Example of using MCCL for a Reduce operation across multiple GPUs
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
constexpr std::size_t kRootGpu = 0;

/**
 * @brief Main function to demonstrate Reduce operation using MCCL
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
    CUDA_CHECK(cudaMalloc(&d_recvbuff, kBytes));

    float *h_in = new float[kN];
    InitVector<float>(h_in, kN);
    CUDA_CHECK(cudaMemcpy(d_sendbuff, h_in, kBytes, cudaMemcpyHostToDevice));

    MCCL<float> mccl(world_rank, world_size);
    const Operation op = SUM;
    mccl.Reduce(d_sendbuff, d_recvbuff, kN, kRootGpu, op);

    if (world_rank == kRootGpu) {
        float *h_result = new float[kN];
        CUDA_CHECK(cudaMemcpy(h_result, d_recvbuff, kBytes, cudaMemcpyDeviceToHost));

        float *h_expected = new float[kN];
        MPI_Reduce(h_in, h_expected, kN, MPI_FLOAT, MPI_SUM, kRootGpu, MPI_COMM_WORLD);

        // We need to gather all inputs to calculate the expected result for the root.
        float *h_all_in = new float[kN * world_size];
        MPI_Gather(h_in, kN, MPI_FLOAT, h_all_in, kN, MPI_FLOAT, kRootGpu, MPI_COMM_WORLD);

        for (size_t i = 0; i < kN; i++) {
            h_expected[i] = 0;
            for (int j = 0; j < world_size; j++) {
                h_expected[i] += h_all_in[j * kN + i];
            }
        }

        bool success = true;
        for (size_t i = 0; i < kN; i++) {
            if (std::fabs(h_result[i] - h_expected[i]) > 1e-5f * world_size) {
                success = false;
                break;
            }
        }
        std::cout << "Device " << world_rank << ": " << (success ? "PASS" : "FAIL") << std::endl;

        delete[] h_result;
        delete[] h_expected;
        delete[] h_all_in;
    }

    delete[] h_in;
    CUDA_CHECK(cudaFree(d_sendbuff));
    CUDA_CHECK(cudaFree(d_recvbuff));

    MPI_Finalize();
    return 0;
}
