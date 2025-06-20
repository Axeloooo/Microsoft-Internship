/**
 * @file broadcast.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief MPI Broadcast collective operation
 * @version 0.1
 * @date 2025-06-15
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>
#include <mpi.h>

#include <cstddef>
#include <iostream>
#include <vector>

#include "data.cuh"

constexpr std::size_t kN = 1024;
constexpr std::size_t kRootGpu = 0;

/**
 * @brief Main function for the MPI Broadcast example.
 *
 * @param argc argument count
 * @param argv argument vector
 * @return int
 */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    cudaSetDevice(world_rank);

    float *d_buffer;
    cudaMalloc(&d_buffer, kN * sizeof(float));

    std::vector<float> h_buffer(kN);

    if (world_rank == kRootGpu) {
        InitVector<float>(h_buffer.data(), h_buffer.size());
        cudaMemcpy(d_buffer, h_buffer.data(), kN * sizeof(float), cudaMemcpyHostToDevice);
    }

    MPI_Bcast(d_buffer, kN, MPI_FLOAT, kRootGpu, MPI_COMM_WORLD);

    cudaMemcpy(h_buffer.data(), d_buffer, kN * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Process " << world_rank << " received broadcast data " << h_buffer[0] << std::endl;

    cudaFree(d_buffer);
    MPI_Finalize();
    return 0;
}