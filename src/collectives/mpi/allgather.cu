/**
 * @file allgather.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief MPI Allgather collective operation
 * @version 0.1
 * @date 2025-06-15
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <mpi.h>

#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int send_buffer = world_rank + 1;

    std::vector<int> recv_buffer(world_size);

    MPI_Allgather(&send_buffer, 1, MPI_INT, recv_buffer.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::cout << "Process " << world_rank << " gathered data: ";
    for (int i = 0; i < world_size; i++) {
        std::cout << recv_buffer[i] << " ";
    }
    std::cout << std::endl;

    MPI_Finalize();
    return 0;
}
