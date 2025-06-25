/**
 * @file allreduce.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief AllReduce example using CUDA IPC and MPI
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

constexpr std::size_t kN = 4194304;
constexpr std::size_t kBytes = kN * sizeof(float);

/**
 * @brief Main function for the AllReduce example using CUDA IPC and MPI
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

    CUDA_CHECK(cudaSetDevice(world_rank));
    for (std::size_t rank = 0; rank < world_size; ++rank) {
        if (rank != world_rank) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(rank, 0));
        }
    }

    float *d_send = nullptr, *d_recv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send, kBytes));
    CUDA_CHECK(cudaMalloc(&d_recv, kBytes));

    std::vector<float> h_init(kN);
    InitVector<float>(h_init.data(), kN);
    CUDA_CHECK(cudaMemcpy(d_send, h_init.data(), kBytes, cudaMemcpyHostToDevice));

    // NCCL init --------------------------------------------------------------
    ncclUniqueId nccl_id;
    if (world_rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }
    MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t nccl_comm;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, world_rank));
    cudaStream_t nccl_stream;
    CUDA_CHECK(cudaStreamCreate(&nccl_stream));
    // -----------------------------------------------------------------------

    float* d_send_nccl = nullptr;
    float* d_recv_nccl = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send_nccl, kBytes));
    CUDA_CHECK(cudaMalloc(&d_recv_nccl, kBytes));
    CUDA_CHECK(cudaMemcpy(d_send_nccl, h_init.data(), kBytes, cudaMemcpyHostToDevice));

    // ------------------ MCCL -----------------------------------------------
    MCCL<float> mccl(world_rank, world_size);

    mccl.AllReduce(d_send, d_recv, kN, Operation::SUM);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CudaTimer mccl_timer("MCCL AllReduce", kBytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_timer.Start();
    mccl.AllReduce(d_send, d_recv, kN, Operation::SUM);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_timer.Stop();
    if (world_rank == 0) {
        mccl_timer.Print();
    }

    std::vector<float> h_ref(kN);
    MPI_CHECK(MPI_Allreduce(h_init.data(), h_ref.data(), kN, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
    VerifyCollective("MCCL", d_recv, h_ref, world_rank, kN);

    // ------------------ NCCL -----------------------------------------------
    NCCL_CHECK(ncclAllReduce(d_send_nccl, d_recv_nccl, kN, ncclFloat32, ncclSum, nccl_comm, nccl_stream));
    CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CudaTimer nccl_timer("NCCL AllReduce", kBytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_timer.Start();
    NCCL_CHECK(ncclAllReduce(d_send_nccl, d_recv_nccl, kN, ncclFloat32, ncclSum, nccl_comm, nccl_stream));
    CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_timer.Stop();

    if (world_rank == 0) {
        nccl_timer.Print();
    }
    VerifyCollective("NCCL", d_recv_nccl, h_ref, world_rank, kN);

    CUDA_CHECK(cudaFree(d_send_nccl));
    CUDA_CHECK(cudaFree(d_recv_nccl));
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    CUDA_CHECK(cudaStreamDestroy(nccl_stream));

    CUDA_CHECK(cudaFree(d_send));
    CUDA_CHECK(cudaFree(d_recv));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
