/**
 * @file allgather.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief AllGather example using CUDA IPC and MPI
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
constexpr std::size_t kBytes = kN * sizeof(float);

/**
 * @brief Main function for the AllGather example using CUDA IPC and MPI.
 *
 * @param argc argument count
 * @param argv argument vector
 * @return int
 */
int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_size, world_rank;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    const std::size_t slice_elems = kN / world_size;
    const std::size_t slice_bytes = slice_elems * sizeof(float);

    CUDA_CHECK(cudaSetDevice(world_rank));
    for (std::size_t rank = 0; rank < world_size; ++rank) {
        if (rank != world_rank) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(rank, 0));
        }
    }

    float* d_send = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send, slice_bytes));

    std::vector<float> h_init(slice_elems);
    InitVector<float>(h_init.data(), slice_elems);
    CUDA_CHECK(cudaMemcpy(d_send, h_init.data(), slice_bytes, cudaMemcpyHostToDevice));

    float* d_recv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_recv, kBytes));

    MCCL<float> mccl(world_rank, world_size);
    mccl.AllGather(d_send, d_recv, slice_elems);

    std::vector<float> h_ref(kN);
    MPI_CHECK(MPI_Allgather(h_init.data(), static_cast<int>(slice_elems), MPI_FLOAT, h_ref.data(), static_cast<int>(slice_elems), MPI_FLOAT, MPI_COMM_WORLD));

    std::vector<float> h_chk(kN);
    CUDA_CHECK(cudaMemcpy(h_chk.data(), d_recv, kBytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (std::size_t i = 0; i < h_ref.size(); ++i)
        if (std::fabs(h_chk[i] - h_ref[i]) > 1e-5f) {
            ok = false;
            break;
        }
    std::cout << "Rank " << world_rank << " : " << (ok ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaFree(d_send));
    CUDA_CHECK(cudaFree(d_recv));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
