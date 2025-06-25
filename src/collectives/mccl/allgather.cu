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
    // ------------------------------------------------------------------------

    float* d_send_nccl = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send_nccl, slice_bytes));
    CUDA_CHECK(cudaMemcpy(d_send_nccl, h_init.data(), slice_bytes, cudaMemcpyHostToDevice));
    float* d_recv_nccl = nullptr;
    CUDA_CHECK(cudaMalloc(&d_recv_nccl, kBytes));

    // ------------------ MCCL -----------------------------------------------
    MCCL<float> mccl(world_rank, world_size);

    mccl.AllGather(d_send, d_recv, slice_elems);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CudaTimer mccl_timer("MCCL AllGather", kBytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_timer.Start();
    mccl.AllGather(d_send, d_recv, slice_elems);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_timer.Stop();

    if (world_rank == 0) {
        mccl_timer.Print();
    }

    std::vector<float> h_ref(kN);
    MPI_CHECK(MPI_Allgather(h_init.data(), static_cast<int>(slice_elems), MPI_FLOAT, h_ref.data(), static_cast<int>(slice_elems), MPI_FLOAT, MPI_COMM_WORLD));

    VerifyCollective("MCCL", d_recv, h_ref, world_rank, kN);

    // ------------------ NCCL -----------------------------------------------
    NCCL_CHECK(ncclAllGather(d_send_nccl, d_recv_nccl, slice_elems, ncclFloat32, nccl_comm, nccl_stream));
    CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CudaTimer nccl_timer("NCCL AllGather", kBytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_timer.Start();
    NCCL_CHECK(ncclAllGather(d_send_nccl, d_recv_nccl, slice_elems, ncclFloat32, nccl_comm, nccl_stream));
    CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_timer.Stop();

    if (world_rank == 0) {
        nccl_timer.Print();
    }

    VerifyCollective("NCCL", d_recv_nccl, h_ref, world_rank, kN);

    // cleanup ----------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_send_nccl));
    CUDA_CHECK(cudaFree(d_recv_nccl));
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    CUDA_CHECK(cudaStreamDestroy(nccl_stream));

    CUDA_CHECK(cudaFree(d_send));
    CUDA_CHECK(cudaFree(d_recv));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
