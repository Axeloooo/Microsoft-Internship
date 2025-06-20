/**
 * @file reduce.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief MPI Reduce collective operation
 * @version 0.1
 * @date 2025-06-15
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <mpi.h>

#include <iostream>

constexpr std::size_t kRootProcess = 0;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int send_buffer = 1;
    int recv_buffer = 0;

    MPI_Reduce(&send_buffer, &recv_buffer, 1, MPI_INT, MPI_SUM, kRootProcess, MPI_COMM_WORLD);

    std::cout << "Process " << world_rank << " received global sum: " << recv_buffer << std::endl;

    MPI_Finalize();
    return 0;
}
