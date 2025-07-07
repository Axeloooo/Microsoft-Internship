/**
 * @file broadcast.cuh
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Header file for the broadcast collective operation in MCCL.
 * @version 0.1
 * @date 2025-06-30
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include "mccl/mccl.cuh"

template <typename T>
__global__ void ringBroadcastKernel(const T* __restrict__ sendbuff, T* __restrict__ recvbuff, size_t n, int root, PeerLink next, int rank, int size);

void mcclBroadcast(const void* sendbuff, void* recvbuff, size_t count, mcclDataType_t datatype, int root, mcclComm_t* comm, cudaStream_t stream);