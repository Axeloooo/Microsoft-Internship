/**
 * @file ipcNvlsMemory.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief IPC NVLS memory management header
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <set>
#include <vector>

namespace trt_allreduce::runtime {
struct IpcNvlsHandle {
    // Begin internal kernel visible fields
    // Changes to these fields must sync with ipcNvlsMemory.h in internal kernel repo
    size_t size = 0;
    // Device pointers used by kernels
    uintptr_t uc_ptr = 0;
    uintptr_t mc_ptr = 0;
    // End internal kernel visible fields
    std::vector<uintptr_t> ipc_uc_ptrs;
    // Device pointers
    CUdeviceptr uc_va;
    CUdeviceptr mc_va;
    std::vector<CUdeviceptr> ipc_uc_vas;
    // Device allocation handles
    CUmemGenericAllocationHandle uc_handle;
    CUmemGenericAllocationHandle mc_handle;
    std::vector<CUmemGenericAllocationHandle> ipc_uc_handles;
};

void MPI_group_barrier(std::set<int> ranks);

bool ipcNvlsSupported();

IpcNvlsHandle* ipcNvlsAllocate(size_t size, std::set<int> ranks);

void ipcNvlsFree(IpcNvlsHandle* handle);

template <typename T>
class DeviceAllocationNvls {
   public:
    DeviceAllocationNvls() = default;

    ~DeviceAllocationNvls() { this->free(); }

    void reset(size_t size, std::set<int> ranks) {
        this->free();
        _handle = ipcNvlsAllocate(size * sizeof(T), ranks);
        _capacity = size;
    }

    // Return device pointer to multicast memory
    [[nodiscard]] T* getMulticastPointer() const { return reinterpret_cast<T*>(_handle->mc_ptr); }

    // Return device pointer for current rank
    [[nodiscard]] T* getUnicastPointer() const { return reinterpret_cast<T*>(_handle->uc_ptr); }

    // Return host list of device pointers to memory on each rank
    [[nodiscard]] T** getIpcUnicastPointers() { return reinterpret_cast<T**>(_handle->ipc_uc_ptrs.data()); }

    [[nodiscard]] size_t getCapacity() const { return _capacity; }

    void free() {
        if (_capacity > 0) {
            ipcNvlsFree(_handle);
            _capacity = 0;
        }
    }

   private:
    size_t _capacity = 0;
    IpcNvlsHandle* _handle;
};
}  // namespace trt_allreduce::runtime