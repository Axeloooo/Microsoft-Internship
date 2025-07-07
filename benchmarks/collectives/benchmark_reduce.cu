/**
 * @file benchmark_reduce.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief A benchmark for the Reduce operation using MCCL and NCCL.
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
constexpr unsigned int kBytes = kN * sizeof(float);
constexpr unsigned int kRootGpu = 0;

/**
 * @brief Main function for the MPI and CUDA IPC reduce example
 *
 * @param argc argument count
 * @param argv argument vector
 * @return int
 */
int main(int argc, char** argv) {
    // Set random seed for reproducibility
    srand(42);

    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_size, world_rank;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    CUDA_CHECK(cudaSetDevice(world_rank));
    for (unsigned int rank = 0; rank < world_size; ++rank) {
        if (rank != world_rank) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(rank, 0));
        }
    }

    // ------- NCCL init ------------------------------------------------------
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

    float *d_send = nullptr, *d_recv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send, kBytes));
    CUDA_CHECK(cudaMalloc(&d_recv, kBytes));

    std::vector<float> h_init(kN);
    InitVector<float>(h_init.data(), kN);
    CUDA_CHECK(cudaMemcpy(d_send, h_init.data(), kBytes, cudaMemcpyHostToDevice));

    // ---------- MCCL timing --------------------------------------------------
    MCCL<float> mccl(world_rank, world_size);
    mccl.Reduce(d_send, d_recv, kN, kRootGpu, Operation::SUM);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CudaTimer mccl_t("MCCL Reduce", kBytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_t.Start();
    mccl.Reduce(d_send, d_recv, kN, kRootGpu, Operation::SUM);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_t.Stop();
    if (world_rank == kRootGpu) {
        mccl_t.Print();
    }

    std::vector<float> h_ref(kN);
    MPI_CHECK(MPI_Reduce(h_init.data(), h_ref.data(), kN, MPI_FLOAT, MPI_SUM, kRootGpu, MPI_COMM_WORLD));

    if (world_rank == kRootGpu) {
        VerifyCollective<float>("MCCL", d_recv, h_ref, world_rank, kN);
    }

    // ---------- NCCL timing --------------------------------------------------
    float* d_send_nccl = nullptr;
    float* d_recv_nccl = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send_nccl, kBytes));
    CUDA_CHECK(cudaMalloc(&d_recv_nccl, kBytes));
    CUDA_CHECK(cudaMemcpy(d_send_nccl, h_init.data(), kBytes, cudaMemcpyHostToDevice));

    NCCL_CHECK(ncclReduce(d_send_nccl, d_recv_nccl, kN, ncclFloat32, ncclSum, kRootGpu, nccl_comm, nccl_stream));
    CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CudaTimer nccl_t("NCCL Reduce", kBytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_t.Start();
    NCCL_CHECK(ncclReduce(d_send_nccl, d_recv_nccl, kN, ncclFloat32, ncclSum, kRootGpu, nccl_comm, nccl_stream));
    CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_t.Stop();
    if (world_rank == kRootGpu) {
        nccl_t.Print();
    }

    if (world_rank == kRootGpu) {
        VerifyCollective<float>("NCCL", d_recv_nccl, h_ref, world_rank, kN);
    }
    CUDA_CHECK(cudaFree(d_send));
    CUDA_CHECK(cudaFree(d_recv));
    CUDA_CHECK(cudaFree(d_send_nccl));
    CUDA_CHECK(cudaFree(d_recv_nccl));
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    CUDA_CHECK(cudaStreamDestroy(nccl_stream));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
