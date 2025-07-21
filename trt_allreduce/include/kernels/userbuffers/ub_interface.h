/**
 * @file ub_interface.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Header file for user buffer interface in the allreduce framework.
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once
#include "common/cudaUtils.h"
#include "common/dataType.h"
#include "cuda_runtime.h"
#include "ub_allocator.h"

namespace trt_allreduce::runtime::ub {
void ub_initialize(trt_allreduce::runtime::WorldConfig const& world_config);
void ub_initialize(int tp_size);
bool ub_is_initialized();
UBBuffer ub_allocate(size_t bytes);
void ub_deallocate(void* addr);
UBBuffer ub_get(int idx);
communicator* ub_comm();
bool ub_supported();
};  // namespace trt_allreduce::runtime::ub

namespace trt_allreduce::kernels::ub {
using namespace trt_allreduce::runtime::ub;
void allreduce2_userbuff_inplace_launcher(int const handler, size_t const offset, size_t const elements, nvinfer1::DataType dataType, communicator* comm,
                                          cudaStream_t stream = 0);

int allgather2_userbuff_residual_launcher(int const handler, size_t const offset, size_t const elements, int const hidden_size, void* residual, nvinfer1::DataType dataType,
                                          communicator* comm, cudaStream_t stream, bool force_enable = false);

int allreduce2_userbuff_rmsnorm_launcher(int const handler, size_t const offset, int const out_handler, size_t const out_offset, size_t const elements, int const hidden_size,
                                         void* beta, void* gamma, float eps, void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm,
                                         cudaStream_t stream);

int allreduce2_userbuff_inplace_rmsnorm_quant_launcher(int const handler, size_t const offset, int const out_handler, size_t const out_offset, size_t const elements,
                                                       int const hidden_size, void* beta, void* gamma, float eps, float* scalefactor, void* residual_in, void* residual_out,
                                                       nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream);
int allreduce2_userbuff_inplace_rmsnorm_quant_fp4_launcher(int const handler, size_t const offset, int const out_handler, size_t const out_offset, int const scale_handler,
                                                           size_t const scale_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
                                                           float* scalefactor, void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm,
                                                           cudaStream_t stream);
}  // namespace trt_allreduce::kernels::ub