/**
 * @file mccl.cuh
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Collective communication operations for multi-GPU environments
 * @version 0.1
 * @date 2025-06-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef COLLECTIVES_CUH
#define COLLECTIVES_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <vector>

/**
 * @brief Enumeration for reduction operations
 *
 */
enum Operation { SUM, PROD, MIN, MAX, AVG };

/**
 * @brief Microsoft Collective Communication Library (MCCL)
 *
 * @tparam T Data type for collective operations (e.g., float, double, int)
 */
template <typename T>
class MCCL {
   private:
    std::size_t num_gpus_;
    std::vector<cudaStream_t> streams_;
    std::vector<T *> tmpbuff_;

    /**
     * @brief Helper function to ensure temporary buffers are allocated
     *
     * @param count Number of elements needed
     */
    void EnsureTmpBuffers(const std::size_t &count);

   public:
    /**
     * @brief Construct a new MCCL object
     *
     * @param gpus Number of GPUs available for collective operations
     */
    explicit MCCL(const std::size_t &gpus);

    /**
     * @brief Destroy the MCCL object
     *
     */
    ~MCCL();

    /**
     * @brief AllReduce operation: reduce and broadcast result to all GPUs
     *
     * @param sendbuff Input buffers on each GPU
     * @param recvbuff Output buffers on each GPU (results stored here)
     * @param count Number of elements per buffer
     * @param op Reduction operation to perform
     */
    void AllReduce(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, const std::size_t &count, const Operation &op);

    /**
     * @brief Broadcast operation: distribute data from root to all GPUs
     *
     * @param sendbuff Input buffer (only root's buffer is used)
     * @param recvbuff Output buffers on each GPU (results stored here)
     * @param count Number of elements per buffer
     * @param root Source GPU ID (0-indexed)
     */
    void Broadcast(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, const std::size_t &count, const std::size_t &root);

    /**
     * @brief Reduce operation: reduce data from all GPUs to root
     *
     * @param sendbuff Input buffers on each GPU
     * @param recvbuff Output buffer (only root's buffer receives result)
     * @param count Number of elements per buffer
     * @param root Destination GPU ID (0-indexed)
     * @param op Reduction operation to perform
     */
    void Reduce(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, const std::size_t &count, const std::size_t &root, const Operation &op);

    /**
     * @brief All-gather operation: gather data from all GPUs to all GPUs
     *
     * @param sendbuff Input buffers on each GPU (count elements each)
     * @param recvbuff Output buffers on each GPU (count*gpus elements each)
     * @param count Number of elements per input buffer
     */
    void AllGather(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, const std::size_t &count);

    /**
     * @brief Reduce-scatter operation: reduce and scatter chunks to GPUs
     *
     * @param sendbuff Input buffers on each GPU (count*gpus elements each)
     * @param recvbuff Output buffers on each GPU (count elements each)
     * @param count Number of elements per output buffer (input is count*gpus)
     * @param op Reduction operation to perform
     */
    void ReduceScatter(const std::vector<T *> &sendbuff, std::vector<T *> &recvbuff, const std::size_t &count, const Operation &op);

    /**
     * @brief Get number of GPUs
     *
     * @return Number of available GPUs
     */
    const std::size_t &GetGpus() const;

    /**
     * @brief Get reference to CUDA streams vector
     *
     * @return Reference to streams vector
     */
    const std::vector<cudaStream_t> &GetStreams() const;
};

#endif