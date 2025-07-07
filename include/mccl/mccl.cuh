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

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <vector>

struct PeerLink {
    int peer_rank;
    int cuda_dev;
    void *peer_recvbuff;
};

struct mcclChannel {
    PeerLink prev;
    PeerLink next;
};

struct mcclComm_t {
    int rank;
    int size;
    int cuda_dev;
    mcclChannel *channels;
    cudaStream_t stream;
};

struct mcclUniqueId {
    unsigned int token[2];
};

enum mcclDataType_t { mcclFloat, mcclFloat32 };

void mcclGetUniqueId(mcclUniqueId *id);

void mcclCommInitRank(mcclComm_t *comm, int size, mcclUniqueId id, int rank);

void mcclCommDestroy(mcclComm_t *comm);

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
    int rank_;
    int size_;
    cudaStream_t stream_;
    std::vector<cudaStream_t> streams_;

    /**
     * @brief Exchange IPC memory handles between GPUs
     *
     * @param buffer Pointer to the buffer containing data to exchange
     * @param bytes Size of the buffer in bytes
     * @param remote_handles Vector to store remote memory handles
     * @param local_handles Vector to store local memory handles
     */
    void ExchangeIpc(const T *buffer, unsigned int bytes, std::vector<T *> &remote_handles, std::vector<cudaIpcMemHandle_t> &local_handles);

    /**
     * @brief Close remote handles for IPC memory
     *
     * @param remote_handles Vector of remote memory handles to close
     */
    void CloseRemoteHandles(std::vector<T *> &remote_handles);

    /**
     * @brief Perform reduction operation on two buffers and store the result in a third buffer
     *
     * @param a First input buffer
     * @param b Second input buffer
     * @param result Output buffer for the result
     * @param n Number of elements in the buffers
     * @param op Reduction operation to perform
     * @param stream CUDA stream for asynchronous execution
     */
    void PerformReduction(const T *a, const T *b, T *result, const unsigned int &n, const Operation &op, const cudaStream_t &stream);

   public:
    /**
     * @brief Construct a new MCCL object
     *
     * @param rank The rank of the process
     * @param size The total number of processes
     */
    explicit MCCL(int rank, int size);

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
    void AllReduce(const T *sendbuff, T *recvbuff, const unsigned int &count, const Operation &op);

    /**
     * @brief Broadcast operation: distribute data from root to all GPUs
     *
     * @param buffer Buffer for the operation.
     * @param count Number of elements per buffer
     * @param root Source GPU ID (0-indexed)
     */
    void Broadcast(T *buffer, const unsigned int &count, const unsigned int &root);

    /**
     * @brief Reduce operation: reduce data from all GPUs to root
     *
     * @param sendbuff Input buffers on each GPU
     * @param recvbuff Output buffer (only root's buffer receives result)
     * @param count Number of elements per buffer
     * @param root Destination GPU ID (0-indexed)
     * @param op Reduction operation to perform
     */
    void Reduce(const T *sendbuff, T *recvbuff, const unsigned int &count, const unsigned int &root, const Operation &op);

    /**
     * @brief All-gather operation: gather data from all GPUs to all GPUs
     *
     * @param sendbuff Input buffers on each GPU (count elements each)
     * @param recvbuff Output buffers on each GPU (count*gpus elements each)
     * @param count Number of elements per input buffer
     */
    void AllGather(const T *sendbuff, T *recvbuff, const unsigned int &count);

    /**
     * @brief Reduce-scatter operation: reduce and scatter chunks to GPUs
     *
     * @param sendbuff Input buffers on each GPU (count*gpus elements each)
     * @param recvbuff Output buffers on each GPU (count elements each)
     * @param count Number of elements per output buffer (input is count*gpus)
     * @param op Reduction operation to perform
     */
    void ReduceScatter(const T *sendbuff, T *recvbuff, const unsigned int &count, const Operation &op);

    /**
     * @brief Get the Rank object
     *
     * @return const unsigned int&
     */
    const unsigned int &GetRank() const;

    /**
     * @brief Get the Size object
     *
     * @return const unsigned int&
     */
    const unsigned int &GetSize() const;

    /**
     * @brief Get the Stream object
     *
     * @return const cudaStream_t&
     */
    const cudaStream_t &GetStream() const;

    /**
     * @brief Get the Streams object
     *
     * @return const std::vector<cudaStream_t>&
     */
    const std::vector<cudaStream_t> &GetStreams() const;
};