/**
 * @file broadcast.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Example of using MCCL for broadcast operation across multiple GPUs
 * @version 0.1
 * @date 2025-06-13
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>
#include <mpi.h>

#include <iostream>
#include <vector>

#include "checker.cuh"
#include "data.cuh"
#include "mccl.cuh"
#include "timer.cuh"

constexpr std::size_t kN = 4194304;
constexpr std::size_t kBytes = (kN * sizeof(float));
constexpr std::size_t kRootGpu = 0;

/**
 * @brief Main function to demonstrate broadcast operation using MCCL
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

    // Set device to the current rank
    CUDA_CHECK(cudaSetDevice(world_rank));

    // Enable peer access between all GPUs
    for (int i = 0; i < world_size; ++i) {
        if (i != world_rank) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(i, 0));
        }
    }

    float *d_buff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buff, kBytes));

    // Initialize data on root device
    if (world_rank == kRootGpu) {
        float *h_buff = new float[kN];
        InitVector<float>(h_buff, kN);
        CUDA_CHECK(cudaMemcpy(d_buff, h_buff, kBytes, cudaMemcpyHostToDevice));
        delete[] h_buff;
    }

    // Initialize MCCL library
    MCCL<float> mccl(world_rank, world_size);

    // Perform broadcast operation using MCCL
    mccl.Broadcast(d_buff, kN, kRootGpu);

    // Verify results
    std::cout << "Broadcast completed on rank " << world_rank << ". Verifying results...";
    float *h_verify_data = new float[kN];
    CUDA_CHECK(cudaMemcpy(h_verify_data, d_buff, kBytes, cudaMemcpyDeviceToHost));

    float *h_root_data = new float[kN];
    if (world_rank == kRootGpu) {
        for (size_t i = 0; i < kN; ++i) {
            h_root_data[i] = h_verify_data[i];
        }
    }
    MPI_Bcast(h_root_data, kN, MPI_FLOAT, kRootGpu, MPI_COMM_WORLD);

    bool success = true;
    for (size_t i = 0; i < kN; i++) {
        if (std::fabs(h_verify_data[i] - h_root_data[i]) > 1e-5f) {
            success = false;
            break;
        }
    }
    std::cout << "Device " << world_rank << ": " << (success ? "PASS" : "FAIL") << std::endl;

    delete[] h_verify_data;
    delete[] h_root_data;

    // Cleanup
    CUDA_CHECK(cudaFree(d_buff));
    MPI_Finalize();

    return 0;
}