/**
 * @file ub_interface.cpp
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Interface for user buffer operations in NVIDIA's TensorRT All-Reduce.
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "kernels/userbuffers/ub_interface.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "common/cudaDriverWrapper.h"

#if ENABLE_MULTI_DEVICE
namespace trt_allreduce::runtime::ub {
void ub_initialize(trt_allreduce::runtime::WorldConfig const& world_config) { UserBufferAllocator::Instance().initialize(world_config); }

void ub_initialize(int tp_size) {
    int num_devices;
    TLLM_CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    trt_allreduce::runtime::WorldConfig world_config(tp_size, 1, 1, COMM_SESSION.getRank(), num_devices);
    UserBufferAllocator::Instance().initialize(world_config);
}

bool ub_is_initialized() { return UserBufferAllocator::Instance().is_initialized(); }

UBBuffer ub_allocate(size_t bytes) { return UserBufferAllocator::Instance().allocate(bytes); }

void ub_deallocate(void* addr) { UserBufferAllocator::Instance().deallocate(addr); }

UBBuffer ub_get(int idx) { return UserBufferAllocator::Instance().get(idx); }

communicator* ub_comm() { return UserBufferAllocator::Instance().comm(); }

bool ub_supported() {
    int cur_dev;
    TLLM_CUDA_CHECK(cudaGetDevice(&cur_dev));
    // UB requires Multicast support
    int mc_support;
    TLLM_CU_CHECK(trt_allreduce::common::CUDADriverWrapper::getInstance()->cuDeviceGetAttribute(&mc_support, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cur_dev));
    return mc_support;
}
};  // namespace trt_allreduce::runtime::ub

namespace trt_allreduce::kernels::ub {
using namespace trt_allreduce::runtime::ub;

void allreduce2_userbuff_inplace_launcher(int const handler, size_t const offset, size_t const elements, nvinfer1::DataType dataType, communicator* comm,
                                          cudaStream_t stream) {
    allreduce2_userbuff_inplace_impl(handler, offset, elements, dataType, comm, stream);
}

int allgather2_userbuff_residual_launcher(int const handler, size_t const offset, size_t const elements, int const hidden_size, void* residual, nvinfer1::DataType dataType,
                                          communicator* comm, cudaStream_t stream, bool force_enable) {
    return allgather2_userbuff_residual_impl(handler, offset, elements, hidden_size, residual, dataType, comm, stream, force_enable);
}

int allreduce2_userbuff_rmsnorm_launcher(int const handler, size_t const offset, int const out_handler, size_t const out_offset, size_t const elements, int const hidden_size,
                                         void* beta, void* gamma, float eps, void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm,
                                         cudaStream_t stream) {
    return allreduce2_userbuff_rmsnorm_impl(handler, offset, out_handler, out_offset, elements, hidden_size, beta, gamma, eps, residual_in, residual_out, dataType, comm,
                                            stream);
}

int allreduce2_userbuff_inplace_rmsnorm_quant_launcher(int const handler, size_t const offset, int const out_handler, size_t const out_offset, size_t const elements,
                                                       int const hidden_size, void* beta, void* gamma, float eps, float* scalefactor, void* residual_in, void* residual_out,
                                                       nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream) {
    return allreduce2_userbuff_inplace_rmsnorm_quant_impl(handler, offset, out_handler, out_offset, elements, hidden_size, beta, gamma, eps, scalefactor, residual_in,
                                                          residual_out, dataType, comm, stream);
}

int allreduce2_userbuff_inplace_rmsnorm_quant_fp4_launcher(int const handler, size_t const offset, int const out_handler, size_t const out_offset, int const scale_handler,
                                                           size_t const scale_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
                                                           float* scalefactor, void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm,
                                                           cudaStream_t stream) {
    return allreduce2_userbuff_inplace_rmsnorm_quant_fp4_impl(handler, offset, out_handler, out_offset, scale_handler, scale_offset, elements, hidden_size, beta, gamma, eps,
                                                              scalefactor, residual_in, residual_out, dataType, comm, stream);
}
}  // namespace trt_allreduce::kernels::ub
#else
namespace trt_allreduce::runtime::ub {
void ub_initialize(trt_allreduce::runtime::WorldConfig const& world_config) {}

void ub_initialize(int tp_size) {}

bool ub_is_initialized() { return false; }

UBBuffer ub_allocate(size_t bytes) { return UBBuffer(); }

void ub_deallocate(void* addr) {}

UBBuffer ub_get(int idx) { return UBBuffer(); }

communicator* ub_comm() { return nullptr; }

bool ub_supported() { return false; }
};  // namespace trt_allreduce::runtime::ub

namespace trt_allreduce::kernels::ub {
using namespace trt_allreduce::runtime::ub;

void allreduce2_userbuff_inplace_launcher(int const handler, size_t const offset, size_t const elements, nvinfer1::DataType dataType, communicator* comm,
                                          cudaStream_t stream) {}

int allgather2_userbuff_residual_launcher(int const handler, size_t const offset, size_t const elements, int const hidden_size, void* residual, nvinfer1::DataType dataType,
                                          communicator* comm, cudaStream_t stream, bool force_enable) {
    return 0;
}

int allreduce2_userbuff_inplace_rmsnorm_quant_launcher(int const handler, size_t const offset, int const out_handler, size_t const out_offset, size_t const elements,
                                                       int const hidden_size, void* beta, void* gamma, float eps, float* scalefactor, void* residual_in, void* residual_out,
                                                       nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream) {
    return 0;
}

int allreduce2_userbuff_inplace_rmsnorm_quant_fp4_launcher(int const handler, size_t const offset, int const out_handler, size_t const out_offset, int const scale_handler,
                                                           size_t const scale_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
                                                           float* scalefactor, void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm,
                                                           cudaStream_t stream) {
    return 0;
}
}  // namespace trt_allreduce::kernels::ub
#endif
