/**
 * @file collectives.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Implementation of collective operations for multi-GPU systems
 * @version 0.1
 * @date 2025-06-13
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>
#include <mpi.h>

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "mccl/mccl.cuh"
#include "utils/checker.cuh"

constexpr std::size_t kBlockSize = 256;

void mcclGetUniqueId(mcclUniqueId* id) {
    std::random_device rd;
    id->token[0] = (static_cast<uint64_t>(rd()) << 32) | rd();
    id->token[1] = (static_cast<uint64_t>(rd()) << 32) | rd();
}

void mcclCommInitRank(mcclComm_t* comm, int size, mcclUniqueId id, int rank) {
    comm->rank = rank;
    comm->size = size;
    comm->cuda_dev = rank % size;
    CUDA_CHECK(cudaSetDevice(comm->cuda_dev));

    // 1. discover local ranks (not strictly needed for 1 ring)
    // 2. enable P2P
    for (std::size_t d = 0; d < size; ++d) {
        if (d != rank) {
            unsigned int devPeer = d % size;
            int can;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, comm->cuda_dev, devPeer));
            if (can) {
                cudaDeviceEnablePeerAccess(devPeer, 0);
            }
        }
    }

    // 3. allocate channel table and fill prev/next ranks
    comm->channels = new mcclChannel[1];
    comm->channels[0].prev.peer_rank = (rank + size - 1) % size;
    comm->channels[0].next.peer_rank = (rank + 1) % size;

    CUDA_CHECK(cudaStreamCreateWithFlags(&comm->stream, cudaStreamNonBlocking));
}

void mcclCommDestroy(mcclComm_t* comm) {
    cudaIpcCloseMemHandle(comm->channels[0].prev.peer_recvbuff);
    cudaIpcCloseMemHandle(comm->channels[0].next.peer_recvbuff);
    CUDA_CHECK(cudaStreamDestroy(comm->stream));
    delete[] comm->channels;
}

/**
 * @brief CUDA kernel for element-wise sum operation
 */
template <typename T>
__global__ void VectorSum(const T* a, const T* b, T* result, const unsigned int n) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise product operation
 */
template <typename T>
__global__ void VectorProd(const T* a, const T* b, T* result, const unsigned int n) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise minimum operation
 */
template <typename T>
__global__ void VectorMin(const T* a, const T* b, T* result, const unsigned int n) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise maximum operation
 */
template <typename T>
__global__ void VectorMax(const T* a, const T* b, T* result, const unsigned int n) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
    }
}

/**
 * @brief CUDA kernel for scalar division (used for averaging)
 */
template <typename T>
__global__ void VectorDivideScalar(const T* a, const T divisor, T* result, const unsigned int n) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] / divisor;
    }
}

/**
 * @brief Helper function to finalize average operation by dividing by GPU count
 *
 * @param data Array to divide
 * @param n Number of elements
 * @param gpu_count Number of GPUs (divisor)
 * @param stream CUDA stream for asynchronous execution
 */
template <typename T>
void FinalizeAverage(T* data, const unsigned int& n, const unsigned int& gpu_count, const cudaStream_t& stream) {
    const unsigned int grid_size = (n + kBlockSize - 1) / kBlockSize;
    VectorDivideScalar<<<grid_size, kBlockSize, 0, stream>>>(data, static_cast<T>(gpu_count), data, n);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Perform reduction operation on two vectors
 *
 * @tparam T Type of data to be used in collective operations
 * @param a First input vector
 * @param b Second input vector
 * @param result Output vector to store the result of the reduction
 * @param n Number of elements in the vectors
 * @param op Reduction operation to perform (e.g., SUM, PROD, MIN, MAX, AVG)
 * @param stream CUDA stream for asynchronous execution
 */
template <typename T>
void MCCL<T>::PerformReduction(const T* a, const T* b, T* result, const unsigned int& n, const Operation& op, const cudaStream_t& stream) {
    const unsigned int grid_size = (n + kBlockSize - 1) / kBlockSize;

    switch (op) {
        case SUM:
            VectorSum<<<grid_size, kBlockSize, 0, stream>>>(a, b, result, n);
            break;
        case PROD:
            VectorProd<<<grid_size, kBlockSize, 0, stream>>>(a, b, result, n);
            break;
        case MIN:
            VectorMin<<<grid_size, kBlockSize, 0, stream>>>(a, b, result, n);
            break;
        case MAX:
            VectorMax<<<grid_size, kBlockSize, 0, stream>>>(a, b, result, n);
            break;
        case AVG:
            VectorSum<<<grid_size, kBlockSize, 0, stream>>>(a, b, result, n);
            break;
        default:
            assert(false);
    }

    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Close remote handles for IPC memory
 *
 * @tparam T Type of data in the remote handles
 * @param remote_handles Vector of pointers to remote buffers
 */
template <typename T>
void MCCL<T>::CloseRemoteHandles(std::vector<T*>& remote_handles) {
    for (unsigned int rank = 0; rank < size_; ++rank) {
        if (remote_handles[rank] != nullptr) {
            cudaIpcCloseMemHandle(remote_handles[rank]);
        }
    }
}

/**
 * @brief Exchange IPC memory handles between GPUs
 *
 * @tparam T Type of data to exchange
 * @param buffer Pointer to the local buffer to exchange
 * @param bytes Size of the buffer in bytes
 * @param remote_ptrs Vector of pointers to remote buffers
 * @param handles Vector of IPC memory handles for local buffers
 */
template <typename T>
void MCCL<T>::ExchangeIpc(const T* buffer, unsigned int bytes, std::vector<T*>& remote_handles, std::vector<cudaIpcMemHandle_t>& local_handles) {
    local_handles.resize(size_);
    remote_handles.assign(size_, nullptr);

    cudaIpcMemHandle_t handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, const_cast<T*>(buffer)));

    MPI_CHECK(MPI_Allgather(&handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, local_handles.data(), sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD));

    for (unsigned int rank = 0; rank < size_; ++rank) {
        if (rank != rank_) {
            CUDA_CHECK(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&remote_handles[rank]), local_handles[rank], cudaIpcMemLazyEnablePeerAccess));
        }
    }
}

/**
 * @brief Construct a new MCCL<T>::MCCL object
 *
 * @tparam T Type of data to be used in collective operations
 * @param rank The rank of the current process (0-indexed)
 * @param size The total number of processes in the collective operation
 */
template <typename T>
MCCL<T>::MCCL(int rank, int size) : rank_(rank), size_(size) {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    streams_.push_back(stream_);
}

/**
 * @brief Destroy the MCCL<T>::MCCL object
 *
 * @tparam T Type of data to be used in collective operations
 */
template <typename T>
MCCL<T>::~MCCL() {
    CUDA_CHECK(cudaStreamDestroy(stream_));
}

/**
 * @brief Broadcast operation: distribute data from root to all GPUs
 *
 * @tparam T Type of data to be used in collective operations
 * @param buffer Buffer for the operation. This buffer will be filled with data from the root GPU and distributed to all other GPUs.
 * @param count Number of elements in the buffer to be broadcasted
 * @param root Source GPU ID (0-indexed) from which the data will be broadcasted
 */
template <typename T>
void MCCL<T>::Broadcast(T* buffer, const unsigned int& count, const unsigned int& root) {
    const unsigned int bytes = count * sizeof(T);
    std::vector<T*> remote_handles;
    std::vector<cudaIpcMemHandle_t> local_handles;
    ExchangeIpc(buffer, bytes, remote_handles, local_handles);

    if (rank_ == static_cast<int>(root)) {
        for (unsigned int dst = 0; dst < size_; ++dst) {
            if (dst != root) {
                CUDA_CHECK(cudaMemcpyPeerAsync(remote_handles[dst], dst, buffer, root, bytes, stream_));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CloseRemoteHandles(remote_handles);
}

/**
 * @brief Reduce operation: reduce data from all GPUs to root
 *
 * @tparam T Type of data to be used in collective operations
 * @param sendbuff Input buffers on each GPU (each GPU has its own data to send)
 * @param recvbuff Output buffer (only root's buffer receives the result of the reduction)
 * @param count Number of elements in each input buffer (each GPU has the same number of elements)
 * @param root Destination GPU ID (0-indexed) where the reduced result will be stored
 * @param op Reduction operation to perform (e.g., SUM, PROD, MIN, MAX, AVG)
 */
template <typename T>
void MCCL<T>::Reduce(const T* sendbuff, T* recvbuff, const unsigned int& count, const unsigned int& root, const Operation& op) {
    const unsigned int bytes = count * sizeof(T);
    std::vector<T*> remote_handles;
    std::vector<cudaIpcMemHandle_t> local_handles;
    ExchangeIpc(sendbuff, bytes, remote_handles, local_handles);

    if (rank_ == static_cast<int>(root)) {
        CUDA_CHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream_));

        T* tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&tmp, bytes));

        for (int src = 0; src < size_; ++src) {
            if (src != root) {
                CUDA_CHECK(cudaMemcpyPeerAsync(tmp, root, remote_handles[src], src, bytes, stream_));
                PerformReduction(recvbuff, tmp, recvbuff, count, op, stream_);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        if (op == AVG) {
            FinalizeAverage(recvbuff, count, size_, stream_);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }

        CUDA_CHECK(cudaFree(tmp));
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CloseRemoteHandles(remote_handles);
}

/**
 * @brief Reduce-Scatter operation: reduce and scatter chunks to GPUs
 *
 * @tparam T Type of data to be used in collective operations
 * @param sendbuff Input buffers on each GPU (each GPU has its own data to send, total size is count * size_)
 * @param recvbuff Output buffers on each GPU (each GPU receives a chunk of the reduced data, size is count)
 * @param count Number of elements in each output buffer (each GPU receives the same number of elements)
 * @param op Reduction operation to perform (e.g., SUM, PROD, MIN, MAX, AVG)
 */
template <typename T>
void MCCL<T>::ReduceScatter(const T* sendbuff, T* recvbuff, const unsigned int& count, const Operation& op) {
    const unsigned int bytes_total = count * size_ * sizeof(T);
    const unsigned int bytes_slice = count * sizeof(T);
    const unsigned int slice_off = static_cast<unsigned int>(rank_) * count;

    std::vector<T*> remote_handles;
    std::vector<cudaIpcMemHandle_t> local_handles;
    ExchangeIpc(sendbuff, bytes_total, remote_handles, local_handles);

    CUDA_CHECK(cudaMemcpyAsync(recvbuff, sendbuff + slice_off, bytes_slice, cudaMemcpyDeviceToDevice, stream_));

    T* tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&tmp, bytes_slice));

    for (unsigned int src = 0; src < size_; ++src) {
        if (src != rank_) {
            const T* remote_slice = remote_handles[src] + slice_off;
            CUDA_CHECK(cudaMemcpyPeerAsync(tmp, rank_, remote_slice, src, bytes_slice, stream_));
            PerformReduction(recvbuff, tmp, recvbuff, count, op, stream_);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaFree(tmp));

    if (op == AVG) {
        FinalizeAverage(recvbuff, count, size_, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CloseRemoteHandles(remote_handles);
}

/**
 * @brief All-gather operation: gather data from all GPUs to all GPUs
 *
 * @tparam T Type of data to be used in collective operations
 * @param sendbuff Input buffers on each GPU (each GPU has its own data to send, size is count)
 * @param recvbuff Output buffers on each GPU (each GPU receives data from all other GPUs, size is count * size_)
 * @param count Number of elements in each input buffer (each GPU has the same number of elements)
 */
template <typename T>
void MCCL<T>::AllGather(const T* sendbuff, T* recvbuff, const unsigned int& count) {
    const unsigned int bytes = count * sizeof(T);
    const unsigned int off_bytes = static_cast<unsigned int>(rank_) * bytes;

    std::vector<T*> remote_handles;
    std::vector<cudaIpcMemHandle_t> local_handles;
    ExchangeIpc(recvbuff, bytes * size_, remote_handles, local_handles);

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char*>(recvbuff) + off_bytes, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream_));

    for (unsigned int dst = 0; dst < size_; ++dst) {
        if (dst != rank_) {
            CUDA_CHECK(cudaMemcpyPeerAsync(reinterpret_cast<char*>(remote_handles[dst]) + off_bytes, dst, sendbuff, rank_, bytes, stream_));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CloseRemoteHandles(remote_handles);
}

/**
 * @brief AllReduce operation: reduce and broadcast result to all GPUs
 *
 * @tparam T Type of data to be used in collective operations
 * @param sendbuff Input buffers on each GPU (each GPU has its own data to send, size is count)
 * @param recvbuff Output buffers on each GPU (each GPU receives the reduced result, size is count)
 * @param count Number of elements in each input buffer (each GPU has the same number of elements)
 * @param op Reduction operation to perform (e.g., SUM, PROD, MIN, MAX, AVG)
 */
template <typename T>
void MCCL<T>::AllReduce(const T* sendbuff, T* recvbuff, const unsigned int& count, const Operation& op) {
    const unsigned int root = 0;
    Reduce(sendbuff, recvbuff, count, root, op);
    Broadcast(recvbuff, count, root);
}

/**
 * @brief Get the rank of the current process in the collective operation
 *
 * @tparam T Type of data to be used in collective operations
 * @return const unsigned int&
 */
template <typename T>
const unsigned int& MCCL<T>::GetRank() const {
    return reinterpret_cast<const unsigned int&>(rank_);
}

/**
 * @brief Get the size of the collective operation
 *
 * @tparam T Type of data to be used in collective operations
 * @return const unsigned int&
 */
template <typename T>
const unsigned int& MCCL<T>::GetSize() const {
    return reinterpret_cast<const unsigned int&>(size_);
}

/**
 * @brief Get the CUDA stream used for operations
 *
 * @tparam T Type of data to be used in collective operations
 * @return const cudaStream_t&
 */
template <typename T>
const cudaStream_t& MCCL<T>::GetStream() const {
    return stream_;
}

/**
 * @brief Get the CUDA stream used for operations
 *
 * @tparam T Type of data to be used in collective operations
 * @return const std::vector<cudaStream_t>&
 */
template <typename T>
const std::vector<cudaStream_t>& MCCL<T>::GetStreams() const {
    return streams_;
}

template class MCCL<float>;