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

#include <mpi.h>

#include <iostream>

constexpr std::size_t kRootGpu = 0;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int broadcast_message;
    if (world_rank == kRootGpu) {
        broadcast_message = 42;
    }

    MPI_Bcast(&broadcast_message, 1, MPI_INT, kRootGpu, MPI_COMM_WORLD);

    std::cout << "Process " << world_rank << " received broadcast message " << broadcast_message << std::endl;

    MPI_Finalize();
    return 0;
}