/**
 * @file ipcSocket.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief IPC socket management for NCCL
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#if ENABLE_MULTI_DEVICE

#include <cstdint>
#include <memory>

namespace trt_allreduce::runtime {
////////////////////////////////////////////////////////////////
// Functions for sharing file descriptors between processes
////////////////////////////////////////////////////////////////
struct NcclIpcSocket;

std::shared_ptr<NcclIpcSocket> ncclIpcSocketInit(int rank, uint64_t hash, uint32_t volatile* abortFlag);

void ncclIpcSocketClose(std::shared_ptr<NcclIpcSocket> handle);

int ncclIpcSocketRecvFd(std::shared_ptr<NcclIpcSocket> handle);

void ncclIpcSocketSendFd(std::shared_ptr<NcclIpcSocket> handle, int fd, int rank, uint64_t hash);

}  // namespace trt_allreduce::runtime

#endif  // ENABLE_MULTI_DEVICE