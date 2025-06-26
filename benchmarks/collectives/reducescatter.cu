/**
 * @file reducescatter.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief ReduceScatter example using CUDA IPC and MPI
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

#include "mccl/mccl.cuh"
#include "mccl/mccl_utils.cuh"
#include "utils/checker.cuh"
#include "utils/data.cuh"
#include "utils/timer.cuh"

constexpr unsigned int kN = 4194304;

/**
 * @brief Main function for the ReduceScatter example using CUDA IPC and MPI
 *
 * @param argc argument count
 * @param argv argument vector
 * @return int
 */
int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    CUDA_CHECK(cudaSetDevice(world_rank));
    for (unsigned int rank = 0; rank < world_size; ++rank) {
        if (rank != world_rank) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(rank, 0));
        }
    }

    const int recvcount = kN / world_size;
    const unsigned int sendbytes = kN * sizeof(float);
    const unsigned int recvbytes = recvcount * sizeof(float);

    float *d_sendbuff = nullptr, *d_recvbuff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sendbuff, sendbytes));
    CUDA_CHECK(cudaMalloc(&d_recvbuff, recvbytes));

    std::vector<float> h_in(kN);
    InitVector<float>(h_in.data(), kN);
    CUDA_CHECK(cudaMemcpy(d_sendbuff, h_in.data(), sendbytes, cudaMemcpyHostToDevice));

    // ---------- NCCL initialisation -----------------------------------------
    ncclUniqueId nccl_id;
    if (world_rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }
    MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t nccl_comm;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, world_rank));
    cudaStream_t nccl_stream;
    CUDA_CHECK(cudaStreamCreate(&nccl_stream));
    // ------------------------------------------------------------------------

    float* d_sendbuff_nccl = nullptr;
    float* d_recvbuff_nccl = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sendbuff_nccl, sendbytes));
    CUDA_CHECK(cudaMalloc(&d_recvbuff_nccl, recvbytes));
    if (world_rank == 0) {
        CUDA_CHECK(cudaMemcpy(d_sendbuff_nccl, h_in.data(), sendbytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_sendbuff_nccl, d_sendbuff, sendbytes, cudaMemcpyDeviceToDevice));
    }

    // -------------------  M C C L  ------------------------------------------
    MCCL<float> mccl(world_rank, world_size);
    mccl.ReduceScatter(d_sendbuff, d_recvbuff, recvcount, Operation::SUM);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CudaTimer mccl_timer("MCCL ReduceScatter", sendbytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_timer.Start();
    mccl.ReduceScatter(d_sendbuff, d_recvbuff, recvcount, Operation::SUM);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_timer.Stop();

    if (world_rank == 0) {
        mccl_timer.Print();
    }

    // ---------------- reference result --------------------------------------
    std::vector<int> recvcounts(world_size, recvcount);
    std::vector<float> h_expected(recvcount);
    MPI_CHECK(MPI_Reduce_scatter(h_in.data(), h_expected.data(), recvcounts.data(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));

    VerifyCollective<float>("MCCL", d_recvbuff, h_expected, world_rank, recvcount);

    // -------------------  N C C L  ------------------------------------------
    NCCL_CHECK(ncclReduceScatter(d_sendbuff_nccl, d_recvbuff_nccl, recvcount, ncclFloat32, ncclSum, nccl_comm, nccl_stream));
    CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CudaTimer nccl_timer("NCCL ReduceScatter", sendbytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_timer.Start();
    NCCL_CHECK(ncclReduceScatter(d_sendbuff_nccl, d_recvbuff_nccl, recvcount, ncclFloat32, ncclSum, nccl_comm, nccl_stream));
    CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_timer.Stop();

    if (world_rank == 0) {
        nccl_timer.Print();
    }

    VerifyCollective<float>("NCCL", d_recvbuff_nccl, h_expected, world_rank, recvcount);

    CUDA_CHECK(cudaFree(d_sendbuff_nccl));
    CUDA_CHECK(cudaFree(d_recvbuff_nccl));
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    CUDA_CHECK(cudaStreamDestroy(nccl_stream));

    CUDA_CHECK(cudaFree(d_sendbuff));
    CUDA_CHECK(cudaFree(d_recvbuff));
    MPI_CHECK(MPI_Finalize());
    return 0;
}