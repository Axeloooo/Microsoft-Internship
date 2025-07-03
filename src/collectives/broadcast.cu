/**
 * @file broadcast.cu
 * @author Axel Sanchez (t-axels@microsoft.com)
 * @brief Implements a ring-based broadcast algorithm using CUDA IPC for peer-to-peer communication.
 * @version 0.1
 * @date 2025-06-30
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "mccl/mccl.cuh"
#include "utils/checker.cuh"

constexpr std::size_t kBlockDim = 256;

template <typename T>
__global__ void ringBroadcastKernel(const T* __restrict__ sendbuff, T* __restrict__ recvbuff, std::size_t n, int root, PeerLink next, int rank, int size) {
    std::size_t base = blockIdx.x * blockDim.x;
    if (base < n) {
        // 1. Pick source
        const T* src = (rank == root) ? sendbuff + base : recvbuff + base;
        // 2. Copy to local recvbuff
        T* dst = recvbuff + base;
#pragma unroll
        for (std::size_t i = threadIdx.x; i < blockDim.x && base + i < n; i += blockDim.x) {
            dst[i] = src[i];
        }
        // 3. Forward to next peer
        if (rank != ((root + size - 1) % size)) {
            __syncthreads();
            T* nextDst = static_cast<T*>(next.peer_recvbuff) + base;
#pragma unroll
            for (std::size_t i = threadIdx.x; i < blockDim.x && base + i < n; i += blockDim.x) {
                nextDst[i] = src[i];
            }
        }
    }
}

void mcclBroadcast(const void* sendbuff, void* recvbuff, std::size_t count, mcclDataType_t datatype, int root, mcclComm_t* comm, cudaStream_t user_stream) {
    const std::size_t grid_dim = (count + kBlockDim - 1) / kBlockDim;
    dim3 grid(grid_dim), block(kBlockDim);

    // 0. exchange IPC handles of recvbuff
    cudaIpcMemHandle_t handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, recvbuff));
    std::vector<cudaIpcMemHandle_t> all_handles(comm->size);
    MPI_CHECK(MPI_Allgather(&handle, sizeof(handle), MPI_BYTE, all_handles.data(), sizeof(handle), MPI_BYTE, MPI_COMM_WORLD));

    // 1. open peer handle for next rank
    const int next = comm->channels[0].next.peer_rank;
    CUDA_CHECK(cudaIpcOpenMemHandle(&comm->channels[0].next.peer_recvbuff, all_handles[next], cudaIpcMemLazyEnablePeerAccess));

    // 2. serialized ring broadcast
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    for (int step = 0; step < comm->size; ++step) {
        if (comm->rank == (root + step) % comm->size) {
            switch (datatype) {
                case mcclFloat32:
                    ringBroadcastKernel<float><<<grid, block, 0, comm->stream>>>(static_cast<const float*>(sendbuff), static_cast<float*>(recvbuff), count, root,
                                                                                 comm->channels[0].next, comm->rank, comm->size);
                    break;
                default:
                    assert(false);
            }
            CUDA_CHECK(cudaStreamSynchronize(comm->stream));
        }
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    // 3. signal user stream
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreate(&event));
    CUDA_CHECK(cudaEventRecord(event, comm->stream));
    CUDA_CHECK(cudaStreamWaitEvent(user_stream, event, 0));
    CUDA_CHECK(cudaEventDestroy(event));
}
