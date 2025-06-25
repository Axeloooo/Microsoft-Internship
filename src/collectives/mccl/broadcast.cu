/**
 * @file broadcast.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Broadcast example using CUDA IPC and MPI
 * @version 0.1
 * @date 2025-06-13
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "checker.cuh"
#include "data.cuh"
#include "mccl.cuh"
#include "timer.cuh"

constexpr std::size_t kDefaultN = 4194304;
constexpr std::size_t kRootGpu = 0;

/**
 * @brief Main function for the broadcast example using CUDA IPC and MPI
 *
 * @param argc argument count
 * @param argv argument vector
 * @return int
 */
int main(int argc, char **argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_size;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    int world_rank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    std::size_t n = kDefaultN;
    if (argc > 1) {
        n = std::stoull(argv[1]);
    }
    const std::size_t bytes = n * sizeof(float);

    CUDA_CHECK(cudaSetDevice(world_rank));

    ncclUniqueId nccl_id;
    if (world_rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t nccl_comm;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, world_rank));
    cudaStream_t nccl_stream;
    CUDA_CHECK(cudaStreamCreate(&nccl_stream));

    for (std::size_t rank = 0; rank < world_size; ++rank) {
        if (rank != world_rank) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(rank, 0));
        }
    }

    float *d_buff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buff, bytes));

    float *d_buff_nccl = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buff_nccl, bytes));

    if (world_rank == kRootGpu) {
        std::vector<float> h(n);
        InitVector<float>(h.data(), n);
        CUDA_CHECK(cudaMemcpy(d_buff, h.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_buff_nccl, h.data(), bytes, cudaMemcpyHostToDevice));
    }

    // ----------   M C C L   ---------- //
    MCCL<float> mccl(world_rank, world_size);

    // warm-up (not timed)
    mccl.Broadcast(d_buff, n, kRootGpu);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CudaTimer mccl_timer("MCCL Broadcast", bytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_timer.Start();
    mccl.Broadcast(d_buff, n, kRootGpu);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_timer.Stop();

    if (world_rank == kRootGpu) {
        mccl_timer.Print();
    }

    // correctness reference (root only)
    std::vector<float> h_ref(n);
    if (world_rank == kRootGpu) {
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_buff, bytes, cudaMemcpyDeviceToHost));
    }
    MPI_CHECK(MPI_Bcast(h_ref.data(), static_cast<int>(n), MPI_FLOAT, kRootGpu, MPI_COMM_WORLD));

    VerifyCollective<float>("MCCL", d_buff, h_ref, world_rank, n);

    // ----------   N C C L   ---------- //
    // warm-up (not timed)
    NCCL_CHECK(ncclBroadcast(d_buff_nccl, d_buff_nccl, n, ncclFloat32, kRootGpu, nccl_comm, nccl_stream));
    CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CudaTimer nccl_timer("NCCL Broadcast", bytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_timer.Start();
    NCCL_CHECK(ncclBroadcast(d_buff_nccl, d_buff_nccl, n, ncclFloat32, kRootGpu, nccl_comm, nccl_stream));
    CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_timer.Stop();

    if (world_rank == kRootGpu) {
        nccl_timer.Print();
    }

    VerifyCollective<float>("NCCL", d_buff_nccl, h_ref, world_rank, n);

    CUDA_CHECK(cudaFree(d_buff));
    CUDA_CHECK(cudaFree(d_buff_nccl));
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    CUDA_CHECK(cudaStreamDestroy(nccl_stream));

    MPI_CHECK(MPI_Finalize());
    return 0;
}