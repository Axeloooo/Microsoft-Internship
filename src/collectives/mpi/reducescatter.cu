/**
 * @file reducescatter.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief
 * @version 0.1
 * @date 2025-06-15
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <mpi.h>

#include <iostream>
#include <vector>

constexpr std::size_t kRootProcess = 0;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::vector<int> send_buffer;

    int recv_buffer;

    if (world_rank == 0) {
        send_buffer = {2, 3, 5, 7};
    }

    MPI_Scatter(send_buffer.data(), 1, MPI_INT, &recv_buffer, 1, MPI_INT, kRootProcess, MPI_COMM_WORLD);

    std::cout << "Process " << world_rank << " received " << recv_buffer << std::endl;

    MPI_Finalize();
    return 0;
}