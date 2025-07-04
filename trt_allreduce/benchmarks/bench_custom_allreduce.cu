/**
 * @file benchmark_custom_allreduce.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Benchmark for custom all-reduce implementations using NCCL and CUDA IPC.
 * @version 0.1
 * @date 2025-07-02
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "kernels/customAllReduceKernels.h"
#include "utils/checker.h"
#include "utils/csv_writer.h"
#include "utils/cuda_timer.cuh"
#include "utils/data.h"

using tensorrt_llm::common::datatype_enum;
using tensorrt_llm::kernels::AllReduceParams;
using tensorrt_llm::kernels::AllReduceStrategyType;
using tensorrt_llm::kernels::customAllReduce;
using tensorrt_llm::kernels::invokeMultiGpuBarrier;
using trt_allreduce::utils::CsvWriter;
using trt_allreduce::utils::CudaTimer;
using trt_allreduce::utils::InitVector;

/**
 * @brief Get the local rank object
 *
 * @return int
 */
static int GetLocalRank() {
    // Portable node-local rank using hostname hashing
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    int world_rank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    int color;
    MPI_Comm node_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm));
    MPI_CHECK(MPI_Comm_rank(node_comm, &color));
    MPI_CHECK(MPI_Comm_free(&node_comm));
    return color;  // rank inside the node
}

/**
 * @brief Share IPC pointers across ranks in a node.
 *
 * @tparam PtrT Type of the pointer (e.g., void*, float*, __half*)
 * @param local_ptr Pointer to the local device memory
 * @param remote_ptrs Vector to hold remote pointers
 * @param local_rank Rank of the local process within the node
 * @param ranks_per_node Number of ranks per node
 */
template <typename PtrT>
void ShareIpcPtrs(PtrT local_ptr, std::vector<PtrT> &remote_ptrs, int local_rank, int ranks_per_node) {
    cudaIpcMemHandle_t local_handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&local_handle, (void *)local_ptr));

    std::vector<cudaIpcMemHandle_t> handles(ranks_per_node);
    MPI_CHECK(MPI_Allgather(&local_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, handles.data(), sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD));

    remote_ptrs.resize(ranks_per_node);
    for (int r = 0; r < ranks_per_node; ++r) {
        if (r == local_rank) {
            remote_ptrs[r] = local_ptr;
        } else {
            void *tmp;
            CUDA_CHECK(cudaIpcOpenMemHandle(&tmp, handles[r], cudaIpcMemLazyEnablePeerAccess));
            remote_ptrs[r] = static_cast<PtrT>(tmp);
        }
    }
}

/**
 * @brief Validate the results of the all-reduce operation.
 *
 * @tparam T fp32 or __half
 * @param d_actual
 * @param h_golden
 * @param world_size
 * @param world_rank
 * @param nElts
 */
template <typename T>
void ValidateResults(void *d_buff, const std::vector<T> &h_golden, int world_size, int world_rank, int nElts) {
    // Copy the results from device to host
    std::vector<T> h_actual(nElts);
    CUDA_CHECK(cudaMemcpy(h_actual.data(), d_buff, nElts * sizeof(T), cudaMemcpyDeviceToHost));

    // Adust epsilon based on data type
    const float eps = std::is_same_v<T, float> ? 1e-5f : 1e-2f;

    // Initialize counter, sums, and max errors
    std::size_t correct = 0;
    double abs_sum = 0.0;
    double sq_err_sum = 0.0;
    float max_abs_err = 0.f;
    float max_rel_err = 0.f;

    // Perform element-wise comparison and compute statistics
    for (int i = 0; i < nElts; ++i) {
        float g = std::is_same_v<T, float> ? static_cast<float>(h_golden[i]) : __half2float(h_golden[i]);
        float a = std::is_same_v<T, float> ? static_cast<float>(h_actual[i]) : __half2float(h_actual[i]);

        const float diff = fabsf(g - a);
        const float rel = (fabsf(g) > 0.f) ? diff / fabsf(g) : diff;  // guard g==0

        if (diff < eps) ++correct;

        abs_sum += diff;
        sq_err_sum += static_cast<double>(diff) * diff;

        max_abs_err = fmaxf(max_abs_err, diff);
        max_rel_err = fmaxf(max_rel_err, rel);
    }

    // Perform global reduction to gather statistics across all ranks
    std::size_t global_correct;
    double global_abs_sum, global_sq_sum;
    float global_max_abs, global_max_rel;

    MPI_CHECK(MPI_Allreduce(&correct, &global_correct, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&abs_sum, &global_abs_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&sq_err_sum, &global_sq_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&max_abs_err, &global_max_abs, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&max_rel_err, &global_max_rel, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD));

    const std::size_t total_elems = static_cast<std::size_t>(nElts) * static_cast<std::size_t>(world_size);

    // Report global results
    if (world_rank == 0) {
        const double meanAbsErr = global_abs_sum / total_elems;
        const double rmse = std::sqrt(global_sq_sum / total_elems);
        const double accuracy = 100.0 * global_correct / total_elems;

        std::cout << "Accuracy: " << accuracy << "%" << std::endl;
        std::cout << "Max Absolute Error: " << global_max_abs << std::endl;
        std::cout << "Mean Absolute Error: " << meanAbsErr << std::endl;
        std::cout << "Max Relative Error: " << global_max_rel << std::endl;
        std::cout << "Root Mean Square Error: " << rmse << std::endl;
    }
}

/**
 * @brief Main function to benchmark custom all-reduce implementations.
 *
 * @param argc Argument count
 * @param argv Argument vector
 * @return int
 */
int main(int argc, char **argv) {
    // Set random seed for reproducibility
    srand(42);

    // MPI intialization
    MPI_CHECK(MPI_Init(&argc, &argv));

    // Setup default hyperparameters
    std::size_t nElts = 1024 * 1024 * 4;
    std::string dtype_s = "fp32";
    int iters = 10;

    // Get hyperparameters from command line arguments
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--elts")) nElts = std::stoul(argv[++i]);
        if (!strcmp(argv[i], "--dtype")) dtype_s = argv[++i];
        if (!strcmp(argv[i], "--iters")) iters = std::atoi(argv[++i]);
    }

    // Setup datatype and element size
    datatype_enum dtype = datatype_enum::TYPE_FP32;
    std::size_t eltSize = sizeof(float);
    if (dtype_s == "fp16") {
        dtype = datatype_enum::TYPE_FP16;
        eltSize = sizeof(__half);
    }

    // Rank and size setup
    int world_size, world_rank;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    const int local_rank = GetLocalRank();
    const int ranks_node = world_size;

    // Set device and create CUDA stream
    CUDA_CHECK(cudaSetDevice(local_rank));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // NCCL initialization
    ncclUniqueId id{};
    if (world_rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Initialize NCCL communicator
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, world_rank));

    // Build message-size multipliers
    std::vector<int> n_values;
    for (int n = 1; n < 16; ++n) n_values.push_back(n);
    for (int n = 16; n < 128; n += 16) n_values.push_back(n);
    for (int n = 128; n < 1024; n += 128) n_values.push_back(n);

    // =====================  per-message-size loop  =====================
    for (int n_mult : n_values) {
        const std::size_t bytes = static_cast<std::size_t>(n_mult) * 10240;
        const std::size_t nElts = bytes / eltSize;

        if (world_rank == 0) {
            std::cout << "\n===== Benchmarking message size " << bytes / 1024 << " KiB (n = " << n_mult << ", elements = " << nElts << ") =====\n";
        }

        // Allocate memory for this size
        void *d_sendbuff = nullptr, *d_recvbuff = nullptr, *d_commBuf = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sendbuff, bytes));
        CUDA_CHECK(cudaMalloc(&d_recvbuff, bytes));
        CUDA_CHECK(cudaMalloc(&d_commBuf, bytes));

        // Barrier work-spaces
        constexpr std::size_t BARRIER_WORDS = 1024;
        uint32_t *d_barrier_in = nullptr;
        uint32_t *d_barrier_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_barrier_in, BARRIER_WORDS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_barrier_out, BARRIER_WORDS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_barrier_in, 0, BARRIER_WORDS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_barrier_out, 0, BARRIER_WORDS * sizeof(uint32_t)));

        // Initialize host send buffers for float and __half types with random values
        std::vector<float> h_send_float;
        std::vector<__half> h_send_half;

        if (dtype == datatype_enum::TYPE_FP32) {
            h_send_float.resize(nElts);
            InitVector<float>(h_send_float.data(), nElts);
            CUDA_CHECK(cudaMemcpy(d_sendbuff, h_send_float.data(), bytes, cudaMemcpyHostToDevice));
        } else {
            h_send_half.resize(nElts);
            InitVector<__half>(h_send_half.data(), nElts);  // random values
            CUDA_CHECK(cudaMemcpy(d_sendbuff, h_send_half.data(), bytes, cudaMemcpyHostToDevice));
        }

        // Exchange IPC
        std::vector<void *> peer_comm_ptrs;
        std::vector<uint32_t *> peer_barrier_in_ptrs;
        std::vector<uint32_t *> peer_barrier_out_ptrs;

        ShareIpcPtrs(d_commBuf, peer_comm_ptrs, local_rank, ranks_node);
        ShareIpcPtrs(d_barrier_in, peer_barrier_in_ptrs, local_rank, ranks_node);
        ShareIpcPtrs(d_barrier_out, peer_barrier_out_ptrs, local_rank, ranks_node);

        // Prepare parameters for custom all-reduce
        AllReduceParams params{};
        for (int r = 0; r < ranks_node; ++r) {
            params.peer_comm_buffer_ptrs[r] = peer_comm_ptrs[r];
            params.peer_barrier_ptrs_in[r] = peer_barrier_in_ptrs[r];
            params.peer_barrier_ptrs_out[r] = peer_barrier_out_ptrs[r];
        }
        params.ranks_per_node = ranks_node;
        params.rank = local_rank;
        params.local_rank = local_rank;

        // Prepare timers to measure performance
        CudaTimer t_nccl("NCCL AllReduce");
        CudaTimer t_one("Custom One-Shot AllReduce");
        CudaTimer t_two("Custom Two-Shot AllReduce");

        // Barrier counter to ensure correct synchronization
        uint32_t barrier_counter = 1;

        // Benchmark NCCL AllReduce
        for (int it = 0; it < iters; ++it) {
            t_nccl.Start();
            NCCL_CHECK(ncclAllReduce(d_sendbuff, d_recvbuff, nElts, (dtype == datatype_enum::TYPE_FP32) ? ncclFloat32 : ncclHalf, ncclSum, comm, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            t_nccl.Stop();
        }
        if (world_rank == 0) t_nccl.Print();

        // Copy NCCL AllReduce result to host as golden reference and validate results
        std::vector<float> h_gold_float;
        std::vector<__half> h_gold_half;

        if (dtype == datatype_enum::TYPE_FP32) {
            h_gold_float.resize(nElts);
            CUDA_CHECK(cudaMemcpy(h_gold_float.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            ValidateResults<float>(d_recvbuff, h_gold_float, world_size, world_rank, nElts);
        } else {
            h_gold_half.resize(nElts);
            CUDA_CHECK(cudaMemcpy(h_gold_half.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            ValidateResults<__half>(d_recvbuff, h_gold_half, world_size, world_rank, nElts);
        }

        // Reset the receive buffer
        CUDA_CHECK(cudaMemset(d_recvbuff, 0, bytes));

        // Benchmark Custom One-Shot AllReduce
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(cudaMemcpyAsync(peer_comm_ptrs[local_rank], d_sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));

            params.barrier_flag = barrier_counter++;
            invokeMultiGpuBarrier(params, stream);

            t_one.Start();
            customAllReduce(params, d_recvbuff, nElts, eltSize, dtype, AllReduceStrategyType::ONESHOT, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            t_one.Stop();
        }
        if (world_rank == 0) t_one.Print();

        // Validate results of Custom One-Shot AllReduce
        if (dtype == datatype_enum::TYPE_FP32) {
            ValidateResults<float>(d_recvbuff, h_gold_float, world_size, world_rank, nElts);
        } else {
            ValidateResults<__half>(d_recvbuff, h_gold_half, world_size, world_rank, nElts);
        }

        // Reset the receive buffer
        CUDA_CHECK(cudaMemset(d_recvbuff, 0, bytes));

        // Benchmark Custom Two-Shot AllReduce
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(cudaMemcpyAsync(peer_comm_ptrs[local_rank], d_sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));

            params.barrier_flag = barrier_counter++;
            invokeMultiGpuBarrier(params, stream);

            t_two.Start();
            customAllReduce(params, d_recvbuff, nElts, eltSize, dtype, AllReduceStrategyType::TWOSHOT, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            t_two.Stop();
        }
        if (world_rank == 0) t_two.Print();

        // Validate results of Custom Two-Shot AllReduce
        if (dtype == datatype_enum::TYPE_FP32) {
            ValidateResults<float>(d_recvbuff, h_gold_float, world_size, world_rank, nElts);
        } else {
            ValidateResults<__half>(d_recvbuff, h_gold_half, world_size, world_rank, nElts);
        }

        // -------- per-size clean-up --------
        CUDA_CHECK(cudaFree(d_sendbuff));
        CUDA_CHECK(cudaFree(d_recvbuff));
        CUDA_CHECK(cudaFree(d_commBuf));
        CUDA_CHECK(cudaFree(d_barrier_in));
        CUDA_CHECK(cudaFree(d_barrier_out));
    }  // end for each message size
    // =====================  end loop  =====================

    // Global clean-up
    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
