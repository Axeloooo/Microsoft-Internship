#include <cuda_runtime.h>

#include <vector>

#include "checker.cuh"
#include "collectives.cuh"

// CUDA kernel definitions for reduction operations
#define BLOCK_SIZE 256

/**
 * @brief CUDA kernel for element-wise sum operation
 */
template <typename T>
__global__ void vectorSum(T *a, T *b, T *result, uint n) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise product operation
 */
template <typename T>
__global__ void vectorProd(T *a, T *b, T *result, uint n) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise minimum operation
 */
template <typename T>
__global__ void vectorMin(T *a, T *b, T *result, uint n) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise maximum operation
 */
template <typename T>
__global__ void vectorMax(T *a, T *b, T *result, uint n) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
    }
}

/**
 * @brief CUDA kernel for scalar division (used for averaging)
 */
template <typename T>
__global__ void vectorDivideScalar(T *a, T divisor, T *result, uint n) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
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
 * @param gpu_count Number of GPUs (used for averaging)
 */
template <typename T>
void performReduction(T *a, T *b, T *result, uint n, Operation op,
                      cudaStream_t stream, uint gpu_count = 1) {
    const uint grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (op) {
        case SUM:
            vectorSum<<<grid_size, BLOCK_SIZE, 0, stream>>>(a, b, result, n);
            break;
        case PROD:
            vectorProd<<<grid_size, BLOCK_SIZE, 0, stream>>>(a, b, result, n);
            break;
        case MIN:
            vectorMin<<<grid_size, BLOCK_SIZE, 0, stream>>>(a, b, result, n);
            break;
        case MAX:
            vectorMax<<<grid_size, BLOCK_SIZE, 0, stream>>>(a, b, result, n);
            break;
        case AVG:
            // For average, we sum first and divide by count later
            vectorSum<<<grid_size, BLOCK_SIZE, 0, stream>>>(a, b, result, n);
            break;
        default:
            std::cerr << "Error: Unsupported operation" << std::endl;
            return;
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
void finalizeAverage(T *data, uint n, uint gpu_count, cudaStream_t stream) {
    const uint grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorDivideScalar<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        data, static_cast<T>(gpu_count), data, n);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
MCCL<T>::MCCL(const uint &gpus)
    : num_gpus_(gpus), streams_(gpus), tmpbuff_(gpus) {
    for (uint i = 0; i < gpus; i++) {
        for (uint j = 0; j < gpus; j++) {
            if (i != j) {
                CHECK_CUDA(cudaSetDevice(i));
                int can_access;
                CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    CHECK_CUDA(cudaDeviceEnablePeerAccess(j, 0));
                }
            }
        }
    }

    for (uint gpu = 0; gpu < this->num_gpus_; gpu++) {
        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaStreamCreate(&this->streams_[gpu]));
        this->tmpbuff_[gpu] = nullptr;
    }
}

template <typename T>
MCCL<T>::~MCCL() {
    for (uint gpu = 0; gpu < this->num_gpus_; gpu++) {
        CHECK_CUDA(cudaSetDevice(gpu));
        if (this->tmpbuff_[gpu] != nullptr) {
            CHECK_CUDA(cudaFree(this->tmpbuff_[gpu]));
        }
        CHECK_CUDA(cudaStreamDestroy(this->streams_[gpu]));
    }
}

/**
 * @brief Helper function to ensure temporary buffers are allocated
 * @param count Number of elements needed
 */
template <typename T>
void MCCL<T>::ensureTmpBuffers(uint count) {
    uint bytes = count * sizeof(T);
    for (uint gpu = 0; gpu < this->gpus; gpu++) {
        if (this->tmpbuff[gpu] == nullptr) {
            CHECK_CUDA(cudaSetDevice(gpu));
            CHECK_CUDA(cudaMalloc(&this->tmpbuff[gpu], bytes));
        }
    }
}

/**
 * @brief Performs an allreduce operation across all GPUs
 *
 * Reduces data from all input buffers using the specified operation and stores
 * the result in all output buffers. This is a collective operation where all
 * GPUs participate and receive the same reduced result.
 *
 * @param sendbuff Input buffers on each GPU
 * @param recvbuff Output buffers on each GPU (results stored here)
 * @param count Number of elements per buffer
 * @param op The reduction operation to perform (SUM, PROD, MIN, MAX, AVG)
 *
 * Algorithm:
 * 1. Reduce phase: Accumulate all data to GPU 0 (root)
 * 2. Broadcast phase: Copy reduced result from root to all other GPUs
 */
template <typename T>
void MCCL<T>::allReduce(const std::vector<T *> &sendbuff,
                        std::vector<T *> &recvbuff, uint count, Operation op) {
    ensureTmpBuffers(count);
    const uint root = 0;
    const uint bytes = count * sizeof(T);

    // Step 1: Reduce to root GPU
    CHECK_CUDA(cudaSetDevice(root));
    CHECK_CUDA(cudaMemcpyAsync(recvbuff[root], sendbuff[root], bytes,
                               cudaMemcpyDeviceToDevice, this->streams[root]));

    for (uint gpu = 0; gpu < this->gpus; gpu++) {
        if (gpu != root) {
            CHECK_CUDA(cudaSetDevice(root));
            CHECK_CUDA(cudaMemcpyPeerAsync(this->tmpbuff[root], root,
                                           sendbuff[gpu], gpu, bytes,
                                           this->streams[root]));
            CHECK_CUDA(cudaStreamSynchronize(this->streams[root]));
            performReduction(recvbuff[root], this->tmpbuff[root],
                             recvbuff[root], count, op, this->streams[root],
                             this->gpus);
        }
    }

    if (op == AVG) {
        CHECK_CUDA(cudaSetDevice(root));
        finalizeAverage(recvbuff[root], count, this->gpus, this->streams[root]);
        CHECK_CUDA(cudaStreamSynchronize(this->streams[root]));
    }

    // Step 2: Broadcast result from root to all other GPUs
    for (uint gpu = 0; gpu < this->gpus; gpu++) {
        if (gpu != root) {
            CHECK_CUDA(cudaSetDevice(gpu));
            CHECK_CUDA(cudaMemcpyPeerAsync(recvbuff[gpu], gpu, recvbuff[root],
                                           root, bytes, this->streams[gpu]));
        }
    }

    for (uint gpu = 0; gpu < this->gpus; gpu++) {
        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaStreamSynchronize(this->streams[gpu]));
    }
}

/**
 * @brief Broadcasts data from root GPU to all other GPUs
 *
 * Copies data from the root GPU's input buffer to all GPUs' output buffers.
 * This is a one-to-many communication pattern where only the root GPU's data
 * is distributed to all participating GPUs.
 *
 * @param sendbuff Input buffers (only root's buffer is used)
 * @param recvbuff Output buffers on each GPU (results stored here)
 * @param count Number of elements per buffer
 * @param root The GPU ID that contains the data to broadcast (0-indexed)
 */
template <typename T>
void MCCL<T>::broadcast(const std::vector<T *> &sendbuff,
                        std::vector<T *> &recvbuff, uint count, uint root) {
    if (root >= this->gpus) {
        std::cerr << "Error: Root GPU " << root << " is invalid. Must be < "
                  << this->gpus << std::endl;
        exit(EXIT_FAILURE);
    }

    const uint bytes = count * sizeof(T);

    CHECK_CUDA(cudaSetDevice(root));
    CHECK_CUDA(cudaMemcpyAsync(recvbuff[root], sendbuff[root], bytes,
                               cudaMemcpyDeviceToDevice, this->streams[root]));

    for (uint gpu = 0; gpu < this->gpus; gpu++) {
        if (gpu != root) {
            CHECK_CUDA(cudaSetDevice(gpu));
            CHECK_CUDA(cudaMemcpyPeerAsync(recvbuff[gpu], gpu, recvbuff[root],
                                           root, bytes, this->streams[gpu]));
        }
    }

    for (uint gpu = 0; gpu < this->gpus; gpu++) {
        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaStreamSynchronize(this->streams[gpu]));
    }
}

/**
 * @brief Reduces data from all GPUs to a single root GPU
 *
 * Performs a reduction operation on data from all input buffers and stores
 * the result only in the root GPU's output buffer. Other GPUs' output buffers
 * remain unchanged.
 *
 * @param sendbuff Input buffers on each GPU
 * @param recvbuff Output buffers (only root's buffer receives result)
 * @param count Number of elements per buffer
 * @param root The GPU ID where the reduced result will be stored (0-indexed)
 * @param op The reduction operation to perform (SUM, PROD, MIN, MAX, AVG)
 */
template <typename T>
void MCCL<T>::reduce(const std::vector<T *> &sendbuff,
                     std::vector<T *> &recvbuff, uint count, uint root,
                     Operation op) {
    if (root >= this->gpus) {
        std::cerr << "Error: Root GPU " << root << " is invalid. Must be < "
                  << this->gpus << std::endl;
        exit(EXIT_FAILURE);
    }

    ensureTmpBuffers(count);
    const uint bytes = count * sizeof(T);

    CHECK_CUDA(cudaSetDevice(root));
    CHECK_CUDA(cudaMemcpyAsync(recvbuff[root], sendbuff[root], bytes,
                               cudaMemcpyDeviceToDevice, this->streams[root]));

    for (uint gpu = 0; gpu < this->gpus; gpu++) {
        if (gpu != root) {
            CHECK_CUDA(cudaSetDevice(root));
            CHECK_CUDA(cudaMemcpyPeerAsync(this->tmpbuff[root], root,
                                           sendbuff[gpu], gpu, bytes,
                                           this->streams[root]));
            CHECK_CUDA(cudaStreamSynchronize(this->streams[root]));
            performReduction(recvbuff[root], this->tmpbuff[root],
                             recvbuff[root], count, op, this->streams[root],
                             this->gpus);
        }
    }

    if (op == AVG) {
        CHECK_CUDA(cudaSetDevice(root));
        finalizeAverage(recvbuff[root], count, this->gpus, this->streams[root]);
        CHECK_CUDA(cudaStreamSynchronize(this->streams[root]));
    }

    CHECK_CUDA(cudaSetDevice(root));
    CHECK_CUDA(cudaStreamSynchronize(this->streams[root]));
}

/**
 * @brief Gathers data from all GPUs and concatenates on all GPUs
 *
 * Each GPU contributes its input data, and the result is the concatenation
 * of all inputs distributed to all GPUs. The output buffer size should be
 * gpus * input_size to accommodate all data.
 *
 * @param sendbuff Input buffers on each GPU (count elements each)
 * @param recvbuff Output buffers on each GPU (count*gpus elements each)
 * @param count Number of elements per input buffer
 *
 * Note: This implementation assumes the output buffers are large enough
 * to hold data from all GPUs (size = count * gpus).
 */
template <typename T>
void MCCL<T>::allGather(const std::vector<T *> &sendbuff,
                        std::vector<T *> &recvbuff, uint count) {
    const uint bytesPerChunk = count * sizeof(T);

    for (uint src = 0; src < this->gpus; src++) {
        for (uint dst = 0; dst < this->gpus; dst++) {
            const uint offset = src * bytesPerChunk;

            if (src == dst) {
                // Local copy within the same GPU
                CHECK_CUDA(cudaSetDevice(dst));
                CHECK_CUDA(cudaMemcpyAsync(
                    recvbuff[dst] + offset, sendbuff[src], bytesPerChunk,
                    cudaMemcpyDeviceToDevice, this->streams[dst]));
            } else {
                // Peer-to-peer copy between different GPUs
                CHECK_CUDA(cudaSetDevice(dst));
                CHECK_CUDA(cudaMemcpyPeerAsync(
                    recvbuff[dst] + offset, dst, sendbuff[src], src,
                    bytesPerChunk, this->streams[dst]));
            }
        }
    }

    for (uint gpu = 0; gpu < this->gpus; gpu++) {
        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaStreamSynchronize(this->streams[gpu]));
    }
}

/**
 * @brief Performs a reduce-scatter operation across all GPUs
 *
 * Reduces data from all input buffers and scatters different portions of the
 * result to different GPUs. Each GPU receives a unique chunk of the reduced
 * data. This is the inverse of allGather - it reduces and then distributes
 * chunks.
 *
 * @param sendbuff Input buffers on each GPU (count*gpus elements each)
 * @param recvbuff Output buffers on each GPU (count elements each)
 * @param count Number of elements per output buffer (input is count*gpus)
 * @param op The reduction operation to perform (SUM, PROD, MIN, MAX, AVG)
 *
 * Algorithm:
 * 1. Divide input data into chunks (one per GPU)
 * 2. Reduce corresponding chunks from all GPUs
 * 3. Each GPU gets one reduced chunk as output
 */
template <typename T>
void MCCL<T>::reduceScatter(const std::vector<T *> &sendbuff,
                            std::vector<T *> &recvbuff, uint count,
                            Operation op) {
    ensureTmpBuffers(count);
    const uint bytesPerChunk = count * sizeof(T);

    for (uint gpu = 0; gpu < this->gpus; gpu++) {
        const uint offset = gpu * bytesPerChunk;

        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaMemcpyAsync(recvbuff[gpu], sendbuff[gpu] + offset,
                                   bytesPerChunk, cudaMemcpyDeviceToDevice,
                                   this->streams[gpu]));

        for (uint src = 0; src < this->gpus; src++) {
            if (src != gpu) {
                CHECK_CUDA(cudaSetDevice(gpu));
                CHECK_CUDA(cudaMemcpyPeerAsync(
                    this->tmpbuff[gpu], gpu, sendbuff[src] + offset, src,
                    bytesPerChunk, this->streams[gpu]));
                CHECK_CUDA(cudaStreamSynchronize(this->streams[gpu]));
                performReduction(recvbuff[gpu], this->tmpbuff[gpu],
                                 recvbuff[gpu], count, op, this->streams[gpu],
                                 this->gpus);
            }
        }

        if (op == AVG) {
            CHECK_CUDA(cudaSetDevice(gpu));
            finalizeAverage(recvbuff[gpu], count, this->gpus,
                            this->streams[gpu]);
        }
    }

    for (uint gpu = 0; gpu < this->gpus; gpu++) {
        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaStreamSynchronize(this->streams[gpu]));
    }
}

template <typename T>
uint MCCL<T>::getGpus() const {
    return this->gpus;
}

template <typename T>
std::vector<cudaStream_t> &MCCL<T>::getStreams() {
    return this->streams;
}

// Explicit template instantiation for common data types
template class MCCL<float>;
template class MCCL<double>;
template class MCCL<int>;
template class MCCL<long>;

// Explicit instantiation of helper functions
template void performReduction<float>(float *, float *, float *, uint,
                                      Operation, cudaStream_t, uint);
template void performReduction<double>(double *, double *, double *, uint,
                                       Operation, cudaStream_t, uint);
template void performReduction<int>(int *, int *, int *, uint, Operation,
                                    cudaStream_t, uint);
template void performReduction<long>(long *, long *, long *, uint, Operation,
                                     cudaStream_t, uint);

template void finalizeAverage<float>(float *, uint, uint, cudaStream_t);
template void finalizeAverage<double>(double *, uint, uint, cudaStream_t);
template void finalizeAverage<int>(int *, uint, uint, cudaStream_t);
template void finalizeAverage<long>(long *, uint, uint, cudaStream_t);
