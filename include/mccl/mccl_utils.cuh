/**
 * @file mccl_utils.cuh
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Utility functions for initializing and finalizing NCCL and CUDA in a collective communication context.
 * @version 0.1
 * @date 2025-07-02
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <iostream>

#include "utils/checker.cuh"

struct CollectiveComm {
    ncclComm_t comm;
    cudaStream_t stream;
    int world_size;
    int world_rank;
};

inline void Init(int argc, char** argv, CollectiveComm* mccl_comm) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mccl_comm->world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mccl_comm->world_rank));

    CUDA_CHECK(cudaSetDevice(mccl_comm->world_rank));

    ncclUniqueId nccl_id;
    if (mccl_comm->world_rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }
    MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

    NCCL_CHECK(ncclCommInitRank(&mccl_comm->comm, mccl_comm->world_size, nccl_id, mccl_comm->world_rank));
    CUDA_CHECK(cudaStreamCreate(&mccl_comm->stream));
}

inline void Finalize(CollectiveComm* mccl_comm) {
    NCCL_CHECK(ncclCommDestroy(mccl_comm->comm));
    CUDA_CHECK(cudaStreamDestroy(mccl_comm->stream));
    MPI_CHECK(MPI_Finalize());
}
