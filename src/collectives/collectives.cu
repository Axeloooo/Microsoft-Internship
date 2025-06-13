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

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "checker.cuh"
#include "collectives.cuh"

// CUDA kernel definitions for reduction operations
constexpr std::size_t kBlockSize = 256;

/**
 * @brief CUDA kernel for element-wise sum operation
 */
template <typename T>
__global__ void VectorSum(T *a, T *b, T *result, std::size_t n) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise product operation
 */
template <typename T>
__global__ void VectorProd(T *a, T *b, T *result, std::size_t n) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise minimum operation
 */
template <typename T>
__global__ void VectorMin(T *a, T *b, T *result, std::size_t n) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise maximum operation
 */
template <typename T>
__global__ void VectorMax(T *a, T *b, T *result, std::size_t n) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
    }
}

/**
 * @brief CUDA kernel for scalar division (used for averaging)
 */
template <typename T>
__global__ void VectorDivideScalar(T *a, T divisor, T *result, std::size_t n) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
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
void PerformReduction(const T *a, const T *b, const T *result, const std::size_t &n, const Operation &op, const cudaStream_t &stream) {
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
            std::assert(false)
    }

    CHECK_CUDA(cudaGetLastError());
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
void FinalizeAverage(const T *data, const std::size_t &n, const std::size_t &gpu_count, const cudaStream_t &stream) {
    const std::size_t grid_size = (n + kBlockSize - 1) / kBlockSize;
    VectorDivideScalar<<<grid_size, kBlockSize, 0, stream>>>(data, static_cast<T>(gpu_count), data, n);
    CHECK_CUDA(cudaGetLastError());
}

/**
 * @brief Construct a new MCCL object
 *
 * @param gpus Number of GPUs available for collective operations
 */
template <typename T>
MCCL<T>::MCCL(const std::size_t &gpus) : num_gpus_(gpus), streams_(gpus), tmpbuff_(gpus) {
    for (std::size_t i = 0; i < gpus; i++) {
        for (std::size_t j = 0; j < gpus; j++) {
            if (i != j) {
                CHECK_CUDA(cudaSetDevice(static_cast<int>(i)));
                int can_access;
                CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, static_cast<int>(i), static_cast<int>(j)));
                if (can_access) {
                    CHECK_CUDA(cudaDeviceEnablePeerAccess(static_cast<int>(j), 0));
                }
            }
        }
    }

    for (std::size_t gpu = 0; gpu < num_gpus_; gpu++) {
        CHECK_CUDA(cudaSetDevice(static_cast<int>(gpu)));
        CHECK_CUDA(cudaStreamCreate(&streams_[gpu]));
        tmpbuff_[gpu] = nullptr;
    }
}

/**
 * @brief Destroy the MCCL object
 *
 */
template <typename T>
MCCL<T>::~MCCL() {
    for (const std::size_t gpu = 0; gpu < num_gpus_; gpu++) {
        CHECK_CUDA(cudaSetDevice(static_cast<int>(gpu)));
        if (tmpbuff_[gpu] != nullptr) {
            CHECK_CUDA(cudaFree(tmpbuff_[gpu]));
        }
        CHECK_CUDA(cudaStreamDestroy(streams_[gpu]));
    }
}

/**
 * @brief Helper function to ensure temporary buffers are allocated
 *
 * @param count Number of elements needed
 */
template <typename T>
void MCCL<T>::EnsureTmpBuffers(const std::size_t &count) {
    const std::size_t bytes = count * sizeof(T);
    for (std::size_t gpu = 0; gpu < num_gpus_; gpu++) {
        if (tmpbuff_[gpu] == nullptr) {
            CHECK_CUDA(cudaSetDevice(static_cast<int>(gpu)));
            CHECK_CUDA(cudaMalloc(&tmpbuff_[gpu], bytes));
        }
    }
}

/**
 * @brief AllReduce operation: reduce and broadcast result to all GPUs
 *
 * @param sendbuff Input buffers on each GPU
 * @param recvbuff Output buffers on each GPU (results stored here)
 * @param count Number of elements per buffer
 * @param op Reduction operation to perform
 */
template <typename T>
void MCCL<T>::AllReduce(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, const std::size_t &count, const Operation &op) {
    EnsureTmpBuffers(count);
    const std::size_t root = 0;
    const std::size_t bytes = count * sizeof(T);

    CHECK_CUDA(cudaSetDevice(static_cast<int>(root)));
    CHECK_CUDA(cudaMemcpyAsync(recvbuff[root], sendbuff[root], bytes, cudaMemcpyDeviceToDevice, streams_[root]));

    for (std::size_t gpu = 0; gpu < num_gpus_; gpu++) {
        if (gpu != root) {
            CHECK_CUDA(cudaSetDevice(static_cast<int>(root)));
            CHECK_CUDA(cudaMemcpyPeerAsync(tmpbuff_[root], static_cast<int>(root), sendbuff[gpu], static_cast<int>(gpu), bytes, streams_[root]));
            CHECK_CUDA(cudaStreamSynchronize(streams_[root]));
            PerformReduction(recvbuff[root], tmpbuff_[root], recvbuff[root], count, op, streams_[root], num_gpus_);
        }
    }

    if (op == AVG) {
        CHECK_CUDA(cudaSetDevice(static_cast<int>(root)));
        FinalizeAverage(recvbuff[root], count, num_gpus_, streams_[root]);
        CHECK_CUDA(cudaStreamSynchronize(streams_[root]));
    }

    for (std::size_t gpu = 0; gpu < num_gpus_; gpu++) {
        if (gpu != root) {
            CHECK_CUDA(cudaSetDevice(static_cast<int>(gpu)));
            CHECK_CUDA(cudaMemcpyPeerAsync(recvbuff[gpu], static_cast<int>(gpu), recvbuff[root], static_cast<int>(root), bytes, streams_[gpu]));
        }
    }

    for (std::size_t gpu = 0; gpu < num_gpus_; gpu++) {
        CHECK_CUDA(cudaSetDevice(static_cast<int>(gpu)));
        CHECK_CUDA(cudaStreamSynchronize(streams_[gpu]));
    }
}

/**
 * @brief Broadcast operation: distribute data from root to all GPUs
 *
 * @param sendbuff Input buffer (only root's buffer is used)
 * @param recvbuff Output buffers on each GPU (results stored here)
 * @param count Number of elements per buffer
 * @param root Source GPU ID (0-indexed)
 */
template <typename T>
void MCCL<T>::Broadcast(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, const std::size_t &count, const std::size_t &root) {
    if (root >= num_gpus_) {
        std::cerr << "Error: Root GPU " << root << " is invalid. Must be < " << num_gpus_ << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const std::size_t bytes = count * sizeof(T);

    CHECK_CUDA(cudaSetDevice(static_cast<int>(root)));
    CHECK_CUDA(cudaMemcpyAsync(recvbuff[root], sendbuff[root], bytes, cudaMemcpyDeviceToDevice, streams_[root]));

    for (std::size_t gpu = 0; gpu < num_gpus_; gpu++) {
        if (gpu != root) {
            CHECK_CUDA(cudaSetDevice(static_cast<int>(gpu)));
            CHECK_CUDA(cudaMemcpyPeerAsync(recvbuff[gpu], static_cast<int>(gpu), recvbuff[root], static_cast<int>(root), bytes, streams_[gpu]));
        }
    }

    for (std::size_t gpu = 0; gpu < num_gpus_; gpu++) {
        CHECK_CUDA(cudaSetDevice(static_cast<int>(gpu)));
        CHECK_CUDA(cudaStreamSynchronize(streams_[gpu]));
    }
}

/**
 * @brief Reduce operation: reduce data from all GPUs to root
 *
 * @param sendbuff Input buffers on each GPU
 * @param recvbuff Output buffer (only root's buffer receives result)
 * @param count Number of elements per buffer
 * @param root Destination GPU ID (0-indexed)
 * @param op Reduction operation to perform
 */
template <typename T>
void MCCL<T>::Reduce(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, const std::size_t &count, const std::size_t &root, const Operation &op) {
    if (root >= num_gpus_) {
        std::cerr << "Error: Root GPU " << root << " is invalid. Must be < " << num_gpus_ << std::endl;
        std::exit(EXIT_FAILURE);
    }

    EnsureTmpBuffers(count);
    const std::size_t bytes = count * sizeof(T);

    CHECK_CUDA(cudaSetDevice(static_cast<int>(root)));
    CHECK_CUDA(cudaMemcpyAsync(recvbuff[root], sendbuff[root], bytes, cudaMemcpyDeviceToDevice, streams_[root]));

    for (std::size_t gpu = 0; gpu < num_gpus_; gpu++) {
        if (gpu != root) {
            CHECK_CUDA(cudaSetDevice(static_cast<int>(root)));
            CHECK_CUDA(cudaMemcpyPeerAsync(tmpbuff_[root], static_cast<int>(root), sendbuff[gpu], static_cast<int>(gpu), bytes, streams_[root]));
            CHECK_CUDA(cudaStreamSynchronize(streams_[root]));
            PerformReduction(recvbuff[root], tmpbuff_[root], recvbuff[root], count, op, streams_[root], num_gpus_);
        }
    }

    if (op == AVG) {
        CHECK_CUDA(cudaSetDevice(static_cast<int>(root)));
        FinalizeAverage(recvbuff[root], count, num_gpus_, streams_[root]);
        CHECK_CUDA(cudaStreamSynchronize(streams_[root]));
    }

    CHECK_CUDA(cudaSetDevice(static_cast<int>(root)));
    CHECK_CUDA(cudaStreamSynchronize(streams_[root]));
}

/**
 * @brief All-gather operation: gather data from all GPUs to all GPUs
 *
 * @param sendbuff Input buffers on each GPU (count elements each)
 * @param recvbuff Output buffers on each GPU (count*gpus elements each)
 * @param count Number of elements per input buffer
 */
template <typename T>
void MCCL<T>::AllGather(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, const std::size_t &count) {
    const std::size_t bytesPerChunk = count * sizeof(T);

    for (std::size_t src = 0; src < num_gpus_; src++) {
        for (std::size_t dst = 0; dst < num_gpus_; dst++) {
            const std::size_t offset = src * count;

            if (src == dst) {
                CHECK_CUDA(cudaSetDevice(static_cast<int>(dst)));
                CHECK_CUDA(cudaMemcpyAsync(recvbuff[dst] + offset, sendbuff[src], bytesPerChunk, cudaMemcpyDeviceToDevice, streams_[dst]));
            } else {
                CHECK_CUDA(cudaSetDevice(static_cast<int>(dst)));
                CHECK_CUDA(cudaMemcpyPeerAsync(recvbuff[dst] + offset, static_cast<int>(dst), sendbuff[src], static_cast<int>(src), bytesPerChunk, streams_[dst]));
            }
        }
    }

    for (std::size_t gpu = 0; gpu < num_gpus_; gpu++) {
        CHECK_CUDA(cudaSetDevice(static_cast<int>(gpu)));
        CHECK_CUDA(cudaStreamSynchronize(streams_[gpu]));
    }
}

/**
 * @brief Reduce-scatter operation: reduce and scatter chunks to GPUs
 *
 * @param sendbuff Input buffers on each GPU (count*gpus elements each)
 * @param recvbuff Output buffers on each GPU (count elements each)
 * @param count Number of elements per output buffer (input is count*gpus)
 * @param op Reduction operation to perform
 */
template <typename T>
void MCCL<T>::ReduceScatter(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, const std::size_t &count, const Operation &op) {
    EnsureTmpBuffers(count);
    const std::size_t bytesPerChunk = count * sizeof(T);

    for (std::size_t gpu = 0; gpu < num_gpus_; gpu++) {
        const std::size_t offset = gpu * count;

        CHECK_CUDA(cudaSetDevice(static_cast<int>(gpu)));
        CHECK_CUDA(cudaMemcpyAsync(recvbuff[gpu], sendbuff[gpu] + offset, bytesPerChunk, cudaMemcpyDeviceToDevice, streams_[gpu]));

        for (std::size_t src = 0; src < num_gpus_; src++) {
            if (src != gpu) {
                CHECK_CUDA(cudaSetDevice(static_cast<int>(gpu)));
                CHECK_CUDA(cudaMemcpyPeerAsync(tmpbuff_[gpu], static_cast<int>(gpu), sendbuff[src] + offset, static_cast<int>(src), bytesPerChunk, streams_[gpu]));
                CHECK_CUDA(cudaStreamSynchronize(streams_[gpu]));
                PerformReduction(recvbuff[gpu], tmpbuff_[gpu], recvbuff[gpu], count, op, streams_[gpu], num_gpus_);
            }
        }

        if (op == AVG) {
            CHECK_CUDA(cudaSetDevice(static_cast<int>(gpu)));
            FinalizeAverage(recvbuff[gpu], count, num_gpus_, streams_[gpu]);
        }
    }

    for (std::size_t gpu = 0; gpu < num_gpus_; gpu++) {
        CHECK_CUDA(cudaSetDevice(static_cast<int>(gpu)));
        CHECK_CUDA(cudaStreamSynchronize(streams_[gpu]));
    }
}

/**
 * @brief Get number of GPUs
 *
 * @return Number of available GPUs
 */
template <typename T>
const std::size_t &MCCL<T>::GetGpus() const {
    return num_gpus_;
}

/**
 * @brief Get reference to CUDA streams vector
 *
 * @return Reference to streams vector
 */
template <typename T>
const std::vector<cudaStream_t> &MCCL<T>::GetStreams() const {
    return streams_;
}

template class MCCL<float>;

template void PerformReduction<float>(const float *a, const float *b, const float *result, const std::size_t &n, const Operation &op, const cudaStream_t &stream);

template void FinalizeAverage<float>(const float *data, const std::size_t &n, const std::size_t &gpu_count, const cudaStream_t &stream);