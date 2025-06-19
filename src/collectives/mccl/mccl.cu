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

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "checker.cuh"
#include "mccl.cuh"

constexpr std::size_t kBlockSize = 256;

/**
 * @brief CUDA kernel for element-wise sum operation
 */
template <typename T>
__global__ void VectorSum(const T *a, const T *b, T *result, const std::size_t n) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise product operation
 */
template <typename T>
__global__ void VectorProd(const T *a, const T *b, T *result, const std::size_t n) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise minimum operation
 */
template <typename T>
__global__ void VectorMin(const T *a, const T *b, T *result, const std::size_t n) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise maximum operation
 */
template <typename T>
__global__ void VectorMax(const T *a, const T *b, T *result, const std::size_t n) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
    }
}

/**
 * @brief CUDA kernel for scalar division (used for averaging)
 */
template <typename T>
__global__ void VectorDivideScalar(const T *a, const T divisor, T *result, const std::size_t n) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] / divisor;
    }
}

/**
 * @brief Helper function to perform reduction operations on GPU
 *
 * @param a First input array
 * @param b Second input array
 * @param result Output array
 * @param n Number of elements
 * @param op Operation to perform
 * @param stream CUDA stream for asynchronous execution
 */
template <typename T>
void PerformReduction(const T *a, const T *b, T *result, const std::size_t &n, const Operation &op, const cudaStream_t &stream) {
    const std::size_t grid_size = (n + kBlockSize - 1) / kBlockSize;

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
 * @brief Helper function to finalize average operation by dividing by GPU count
 *
 * @param data Array to divide
 * @param n Number of elements
 * @param gpu_count Number of GPUs (divisor)
 * @param stream CUDA stream for asynchronous execution
 */
template <typename T>
void FinalizeAverage(T *data, const std::size_t &n, const std::size_t &gpu_count, const cudaStream_t &stream) {
    const std::size_t grid_size = (n + kBlockSize - 1) / kBlockSize;
    VectorDivideScalar<<<grid_size, kBlockSize, 0, stream>>>(data, static_cast<T>(gpu_count), data, n);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Construct a new MCCL<T>::MCCL object
 *
 * @param gpus Number of GPUs available for collective operations
 */
template <typename T>
MCCL<T>::MCCL(int rank, int size) : rank_(rank), size_(size) {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

/**
 * @brief Destroy the MCCL<T>::MCCL object
 *
 */
template <typename T>
MCCL<T>::~MCCL() {
    CUDA_CHECK(cudaStreamDestroy(stream_));
}

/**
 * @brief All-reduce operation using a ring-based algorithm
 *
 * @param sendbuff Input buffer
 * @param recvbuff Output buffer
 * @param count Number of elements
 * @param op Reduction operation
 */
template <typename T>
void MCCL<T>::AllReduce(const T *sendbuff, T *recvbuff, const std::size_t &count, const Operation &op) {
    const int root = 0;
    T *reduce_buff = nullptr;
    if (rank_ == root) {
        CUDA_CHECK(cudaMalloc(&reduce_buff, count * sizeof(T)));
    } else {
        reduce_buff = recvbuff;
    }

    Reduce(sendbuff, reduce_buff, count, root, op);

    Broadcast(reduce_buff, count, root);

    if (reduce_buff != recvbuff) {
        CUDA_CHECK(cudaMemcpyAsync(recvbuff, reduce_buff, count * sizeof(T), cudaMemcpyDeviceToDevice, stream_));
    }

    if (rank_ == root) {
        CUDA_CHECK(cudaFree(reduce_buff));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

/**
 * @brief Broadcast operation using a ring-based algorithm
 *
 * @param buffer Buffer for the operation
 * @param count Number of elements
 * @param root Source GPU ID
 */
template <typename T>
void MCCL<T>::Broadcast(T *buffer, const std::size_t &count, const std::size_t &root) {
    if (rank_ == root) {
        for (std::size_t i = 0; i < size_; ++i) {
            if (i != rank_) {
                CUDA_CHECK(cudaMemcpyPeerAsync(buffer, i, buffer, rank_, count * sizeof(T), stream_));
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

/**
 * @brief Reduce operation using a ring-based algorithm
 *
 * @param sendbuff Input buffer
 * @param recvbuff Output buffer
 * @param count Number of elements
 * @param root Destination GPU ID
 * @param op Reduction operation
 */
template <typename T>
void MCCL<T>::Reduce(const T *sendbuff, T *recvbuff, const std::size_t &count, const std::size_t &root, const Operation &op) {
    if (rank_ == root) {
        CUDA_CHECK(cudaMemcpyAsync(recvbuff, sendbuff, count * sizeof(T), cudaMemcpyDeviceToDevice, stream_));
        T *temp_buff;
        CUDA_CHECK(cudaMalloc(&temp_buff, count * sizeof(T)));
        for (std::size_t i = 0; i < size_; i++) {
            if (i != root) {
                CUDA_CHECK(cudaMemcpyPeerAsync(temp_buff, rank_, sendbuff, i, count * sizeof(T), stream_));
                PerformReduction(recvbuff, temp_buff, recvbuff, count, op, stream_);
            }
        }
        CUDA_CHECK(cudaFree(temp_buff));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

/**
 * @brief All-gather operation using a ring-based algorithm
 *
 * @param sendbuff Input buffer
 * @param recvbuff Output buffer
 * @param count Number of elements
 */
template <typename T>
void MCCL<T>::AllGather(const T *sendbuff, T *recvbuff, const std::size_t &count) {
    for (std::size_t i = 0; i < size_; i++) {
        if (i == rank_) {
            CUDA_CHECK(cudaMemcpyAsync(recvbuff + i * count, sendbuff, count * sizeof(T), cudaMemcpyDeviceToDevice, stream_));
        } else {
            CUDA_CHECK(cudaMemcpyPeerAsync(recvbuff + i * count, rank_, sendbuff, i, count * sizeof(T), stream_));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

/**
 * @brief Reduce-scatter operation using a ring-based algorithm
 *
 * @param sendbuff Input buffer
 * @param recvbuff Output buffer
 * @param count Number of elements
 * @param op Reduction operation
 */
template <typename T>
void MCCL<T>::ReduceScatter(const T *sendbuff, T *recvbuff, const std::size_t &recvcount, const Operation &op) {
    const T *my_chunk_in_sendbuff = sendbuff + rank_ * recvcount;
    CUDA_CHECK(cudaMemcpyAsync(recvbuff, my_chunk_in_sendbuff, recvcount * sizeof(T), cudaMemcpyDeviceToDevice, stream_));

    T *temp_buff;
    CUDA_CHECK(cudaMalloc(&temp_buff, recvcount * sizeof(T)));

    for (std::size_t i = 0; i < size_; i++) {
        if (i != rank_) {
            const T *remote_chunk = sendbuff + rank_ * recvcount;
            CUDA_CHECK(cudaMemcpyPeerAsync(temp_buff, rank_, remote_chunk, i, recvcount * sizeof(T), stream_));
            PerformReduction(recvbuff, temp_buff, recvbuff, recvcount, op, stream_);
        }
    }

    CUDA_CHECK(cudaFree(temp_buff));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

template class MCCL<float>;