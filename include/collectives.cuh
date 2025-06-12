#pragma once

#include <vector>
#include <iostream>
#include <cuda_runtime.h>

/**
 * @brief Operation types supported by the collective communication library
 */
enum Operation
{
    SUM,  ///< Element-wise sum
    PROD, ///< Element-wise product
    MIN,  ///< Element-wise minimum
    MAX,  ///< Element-wise maximum
    AVG   ///< Element-wise average
};

/**
 * @brief Microsoft Collective Communication Library (MCCL)
 *
 * A template-based multi-GPU collective communication library that provides
 * efficient implementations of common collective operations including allreduce,
 * broadcast, reduce, allgather, and reduce-scatter.
 *
 * Features:
 * - Multi-GPU support with peer-to-peer memory access
 * - Asynchronous operations using CUDA streams
 * - Template-based design supporting multiple data types
 * - NCCL-style API where buffers are passed as parameters
 * - Support for various reduction operations (sum, product, min, max, average)
 *
 * Usage:
 * @code
 * // Create MCCL instance for available GPUs
 * MCCL<float> mccl(8);
 *
 * // Allocate buffers on each GPU
 * std::vector<float*> sendbuff, recvbuff;
 * // ... allocate buffers ...
 *
 * // Perform allreduce with sum operation
 * Operation op = SUM;
 * mccl.allReduce(sendbuff, recvbuff, 1024*1024, op);
 *
 * // Broadcast from GPU 0
 * mccl.broadcast(sendbuff, recvbuff, 1024*1024, 0);
 * @endcode
 *
 * @tparam T Data type (float, double, int, long, etc.)
 */
template <typename T>
class MCCL
{
private:
    uint gpus;                         ///< Number of available GPUs
    std::vector<cudaStream_t> streams; ///< CUDA streams for each GPU
    std::vector<T *> tmpbuff;          ///< Temporary buffers for intermediate operations

    /**
     * @brief Helper function to ensure temporary buffers are allocated
     * @param count Number of elements needed
     */
    void ensureTmpBuffers(uint count);

public:
    /**
     * @brief Constructor - initializes multi-GPU environment
     */
    MCCL(const uint &gpus);

    /**
     * @brief Destructor - cleans up GPU memory and streams
     */
    ~MCCL();

    /**
     * @brief AllReduce operation: reduce and broadcast result to all GPUs
     * @param sendbuff Input buffers on each GPU
     * @param recvbuff Output buffers on each GPU (results stored here)
     * @param count Number of elements per buffer
     * @param op Reduction operation to perform
     */
    void allReduce(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, uint count, Operation op);

    /**
     * @brief Broadcast operation: distribute data from root to all GPUs
     * @param sendbuff Input buffer (only root's buffer is used)
     * @param recvbuff Output buffers on each GPU (results stored here)
     * @param count Number of elements per buffer
     * @param root Source GPU ID (0-indexed)
     */
    void broadcast(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, uint count, uint root);

    /**
     * @brief Reduce operation: reduce data from all GPUs to root
     * @param sendbuff Input buffers on each GPU
     * @param recvbuff Output buffer (only root's buffer receives result)
     * @param count Number of elements per buffer
     * @param root Destination GPU ID (0-indexed)
     * @param op Reduction operation to perform
     */
    void reduce(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, uint count, uint root, Operation op);

    /**
     * @brief All-gather operation: gather data from all GPUs to all GPUs
     * @param sendbuff Input buffers on each GPU (count elements each)
     * @param recvbuff Output buffers on each GPU (count*gpus elements each)
     * @param count Number of elements per input buffer
     * Note: Output buffers should be sized to hold data from all GPUs (count*gpus)
     */
    void allGather(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, uint count);

    /**
     * @brief Reduce-scatter operation: reduce and scatter chunks to GPUs
     * @param sendbuff Input buffers on each GPU (count*gpus elements each)
     * @param recvbuff Output buffers on each GPU (count elements each)
     * @param count Number of elements per output buffer (input is count*gpus)
     * @param op Reduction operation to perform
     */
    void reduceScatter(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, uint count, Operation op);

    /**
     * @brief Get number of GPUs
     * @return Number of available GPUs
     */
    uint getGpus() const;

    /**
     * @brief Get reference to CUDA streams vector
     * @return Reference to streams vector
     */
    std::vector<cudaStream_t> &getStreams();
};