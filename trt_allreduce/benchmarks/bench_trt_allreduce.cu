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
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "kernels/customAllReduceKernels.h"
#include "utils/checker.h"
#include "utils/csv_writer.h"
#include "utils/cuda_timer.cuh"

using trt_allreduce::common::datatype_enum;
using trt_allreduce::kernels::AllReduceParams;
using trt_allreduce::kernels::AllReduceStrategyType;
using trt_allreduce::kernels::customAllReduce;
using trt_allreduce::kernels::invokeMultiGpuBarrier;
using trt_allreduce::utils::CsvWriter;
using trt_allreduce::utils::CudaTimer;

const std::size_t kWarmup = 10;
const std::string kNcclLabel = "NCCL";
const std::string kOneShotLabel = "One-Shot";
const std::string kTwoShotLabel = "Two-Shot";
const std::string kFileName = "benchmark_trt_allreduce";
const std::string kDirectoryPath = "data";

// Forward declarations
struct BenchmarkConfig;
struct BenchmarkResults;

/**
 * @brief Structure to hold benchmark configuration
 *
 */
struct BenchmarkConfig {
    datatype_enum dtype;
    std::size_t element_size;
    int iterations;
    int world_size;
    int world_rank;
    int local_rank;
    int ranks_per_node;
    cudaStream_t stream;
    ncclComm_t nccl_comm;
};

/**
 * @brief Structure to hold benchmark results
 *
 */
struct BenchmarkResults {
    std::string implementation;
    std::size_t num_elements;
    std::size_t message_bytes;
    float latency_ms;
    double bandwidth_gbps;
    double accuracy_pct;
    float max_abs_error;
    double mean_abs_error;
    float max_rel_error;
    double rmse;
};

/**
 * @brief Initialize a vector with random values
 *
 * @tparam T
 * @param vector Pointer to the vector data
 * @param n Number of elements in the vector
 */
template <typename T>
void InitVector(T* vector, const std::size_t& n);

/**
 * @brief Specialization for float type
 *
 * @tparam float
 * @param vector Pointer to the vector data
 * @param n Number of elements in the vector
 */
template <>
void InitVector<float>(float* vector, const std::size_t& n) {
    for (std::size_t i = 0; i < n; ++i) {
        vector[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

/**
 * @brief Specialization for __half type
 *
 * @tparam __half
 * @param vector Pointer to the vector data
 * @param n Number of elements in the vector
 */
template <>
void InitVector<__half>(__half* vector, const std::size_t& n) {
    for (std::size_t i = 0; i < n; ++i) {
        float val = static_cast<float>(rand()) / RAND_MAX;
        vector[i] = __float2half(val);
    }
}

/**
 * @brief Generate a timestamp string in the format YYYYMMDD-HHMMSS
 *
 * @return std::string
 */
std::string GenerateTimestamp() {
    auto now = std::time(nullptr);
    auto* tm_now = std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(tm_now, "%Y%m%d-%H%M%S");
    return oss.str();
}

/**
 * @brief Calculate bandwidth in GB/s
 *
 * @param bytes Number of bytes transferred
 * @param latency_ms Latency in milliseconds
 * @return double
 */
double CalculateBandwidth(std::size_t bytes, float latency_ms) { return static_cast<double>(bytes) / (latency_ms * 1e-3) / 1e9; }

/**
 * @brief Get the Local Rank object
 *
 * @return int
 */
static int GetLocalRank() {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    int world_rank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    int color;
    MPI_Comm node_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm));
    MPI_CHECK(MPI_Comm_rank(node_comm, &color));
    MPI_CHECK(MPI_Comm_free(&node_comm));
    return color;
}

/**
 * @brief Share IPC pointers across ranks in a node
 *
 * @tparam PtrT Type of the pointer (e.g., void*, float*, __half*)
 * @param local_ptr Pointer to the local device memory
 * @param remote_ptrs Vector to hold remote pointers
 * @param local_rank Rank of the local process within the node
 * @param ranks_per_node Number of ranks per node
 */
template <typename PtrT>
void ShareIpcPtrs(PtrT local_ptr, std::vector<PtrT>& remote_ptrs, int local_rank, int ranks_per_node) {
    cudaIpcMemHandle_t local_handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&local_handle, static_cast<void*>(local_ptr)));

    std::vector<cudaIpcMemHandle_t> handles(ranks_per_node);
    MPI_CHECK(MPI_Allgather(&local_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, handles.data(), sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD));

    remote_ptrs.resize(ranks_per_node);
    for (int r = 0; r < ranks_per_node; ++r) {
        if (r == local_rank) {
            remote_ptrs[r] = local_ptr;
        } else {
            void* tmp;
            CUDA_CHECK(cudaIpcOpenMemHandle(&tmp, handles[r], cudaIpcMemLazyEnablePeerAccess));
            remote_ptrs[r] = static_cast<PtrT>(tmp);
        }
    }
}

/**
 * @brief Validate the results of the all-reduce operation
 *
 * @tparam T Data type (float or __half)
 * @param d_buff Device buffer with actual results
 * @param h_golden Host buffer with expected results
 * @param config Benchmark configuration
 * @param n_elts Number of elements
 * @return BenchmarkResults with validation metrics
 */
template <typename T>
BenchmarkResults ValidateResults(void* d_buff, const std::vector<T>& h_golden, const BenchmarkConfig& config, std::size_t n_elts) {
    std::vector<T> h_actual(n_elts);
    CUDA_CHECK(cudaMemcpy(h_actual.data(), d_buff, n_elts * sizeof(T), cudaMemcpyDeviceToHost));

    const float eps = std::is_same_v<T, float> ? 1e-5f : 1e-2f;
    std::size_t correct = 0;
    double abs_sum = 0.0;
    double sq_err_sum = 0.0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    for (std::size_t i = 0; i < n_elts; ++i) {
        float g = std::is_same_v<T, float> ? static_cast<float>(h_golden[i]) : __half2float(h_golden[i]);
        float a = std::is_same_v<T, float> ? static_cast<float>(h_actual[i]) : __half2float(h_actual[i]);

        const float diff = std::abs(g - a);
        const float rel = (std::abs(g) > 0.0f) ? diff / std::abs(g) : diff;

        if (diff < eps) ++correct;

        abs_sum += diff;
        sq_err_sum += static_cast<double>(diff) * diff;
        max_abs_err = std::max(max_abs_err, diff);
        max_rel_err = std::max(max_rel_err, rel);
    }

    // Global reduction for statistics
    std::size_t global_correct;
    double global_abs_sum, global_sq_sum;
    float global_max_abs, global_max_rel;

    MPI_CHECK(MPI_Allreduce(&correct, &global_correct, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&abs_sum, &global_abs_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&sq_err_sum, &global_sq_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&max_abs_err, &global_max_abs, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&max_rel_err, &global_max_rel, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD));

    const std::size_t total_elems = n_elts * static_cast<std::size_t>(config.world_size);

    BenchmarkResults results{};
    results.accuracy_pct = 100.0 * global_correct / total_elems;
    results.max_abs_error = global_max_abs;
    results.mean_abs_error = global_abs_sum / total_elems;
    results.max_rel_error = global_max_rel;
    results.rmse = std::sqrt(global_sq_sum / total_elems);

    return results;
}

/**
 * @brief Benchmark NCCL AllReduce
 *
 * @param d_sendbuff Send buffer for data to reduce
 * @param d_recvbuff Receive buffer for results
 * @param n_elts Number of elements to reduce
 * @param config Configuration for the benchmark
 * @return BenchmarkResults
 */
BenchmarkResults BenchmarkNccl(void* d_sendbuff, void* d_recvbuff, std::size_t n_elts, const BenchmarkConfig& config) {
    // Warm-up phase
    for (int i = 0; i < kWarmup; ++i) {
        NCCL_CHECK(
            ncclAllReduce(d_sendbuff, d_recvbuff, n_elts, (config.dtype == datatype_enum::TYPE_FP32) ? ncclFloat32 : ncclHalf, ncclSum, config.nccl_comm, config.stream));
        CUDA_CHECK(cudaStreamSynchronize(config.stream));
    }

    CudaTimer timer;
    float total_time = 0.0f;

    // Benchmark iterations
    timer.Start();
    for (int iter = 0; iter < config.iterations; ++iter) {
        NCCL_CHECK(
            ncclAllReduce(d_sendbuff, d_recvbuff, n_elts, (config.dtype == datatype_enum::TYPE_FP32) ? ncclFloat32 : ncclHalf, ncclSum, config.nccl_comm, config.stream));
    }
    total_time = timer.Stop();

    BenchmarkResults results{};
    results.implementation = kNcclLabel;
    results.num_elements = n_elts;
    results.message_bytes = n_elts * config.element_size;
    results.latency_ms = total_time / config.iterations;
    results.bandwidth_gbps = CalculateBandwidth(results.message_bytes, results.latency_ms);

    return results;
}

/**
 * @brief Benchmark Custom One-Shot AllReduce
 *
 * @param d_sendbuff Send buffer for data to reduce
 * @param d_recvbuff Receive buffer for results
 * @param n_elts Number of elements to reduce
 * @param params AllReduce parameters including peer communication buffers and barrier flags
 * @param config Configuration for the benchmark
 * @param barrier_counter Counter for multi-GPU barrier synchronization
 * @return BenchmarkResults
 */
BenchmarkResults BenchmarkOneshot(void* d_sendbuff, void* d_recvbuff, std::size_t n_elts, const AllReduceParams& params, const BenchmarkConfig& config,
                                  uint32_t& barrier_counter) {
    // Warm-up phase
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(params.peer_comm_buffer_ptrs[config.local_rank], d_sendbuff, n_elts * config.element_size, cudaMemcpyDeviceToDevice, config.stream));
        AllReduceParams wp = params;
        wp.barrier_flag = barrier_counter++;
        invokeMultiGpuBarrier(wp, config.stream);
        customAllReduce(wp, d_recvbuff, n_elts, config.element_size, config.dtype, AllReduceStrategyType::ONESHOT, config.stream);
    }

    CudaTimer timer;
    float total_time = 0.0f;

    // Benchmark iterations
    timer.Start();
    for (int iter = 0; iter < config.iterations; ++iter) {
        CUDA_CHECK(cudaMemcpyAsync(params.peer_comm_buffer_ptrs[config.local_rank], d_sendbuff, n_elts * config.element_size, cudaMemcpyDeviceToDevice, config.stream));

        AllReduceParams iter_params = params;
        iter_params.barrier_flag = barrier_counter++;

        invokeMultiGpuBarrier(iter_params, config.stream);

        customAllReduce(iter_params, d_recvbuff, n_elts, config.element_size, config.dtype, AllReduceStrategyType::ONESHOT, config.stream);
    }
    total_time += timer.Stop();

    BenchmarkResults results{};
    results.implementation = kOneShotLabel;
    results.num_elements = n_elts;
    results.message_bytes = n_elts * config.element_size;
    results.latency_ms = total_time / config.iterations;
    results.bandwidth_gbps = CalculateBandwidth(results.message_bytes, results.latency_ms);

    return results;
}

/**
 * @brief Benchmark Custom Two-Shot AllReduce
 *
 * @param d_sendbuff Send buffer for data to reduce
 * @param d_recvbuff Receive buffer for results
 * @param n_elts Number of elements to reduce
 * @param params AllReduce parameters including peer communication buffers and barrier flags
 * @param config Configuration for the benchmark
 * @param barrier_counter Counter for multi-GPU barrier synchronization
 * @return BenchmarkResults
 */
BenchmarkResults BenchmarkTwoshot(void* d_sendbuff, void* d_recvbuff, std::size_t n_elts, const AllReduceParams& params, const BenchmarkConfig& config,
                                  uint32_t& barrier_counter) {
    // Warm-up phase
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(params.peer_comm_buffer_ptrs[config.local_rank], d_sendbuff, n_elts * config.element_size, cudaMemcpyDeviceToDevice, config.stream));
        AllReduceParams wp = params;
        wp.barrier_flag = barrier_counter++;
        invokeMultiGpuBarrier(wp, config.stream);
        customAllReduce(wp, d_recvbuff, n_elts, config.element_size, config.dtype, AllReduceStrategyType::TWOSHOT, config.stream);
    }

    CudaTimer timer;
    float total_time = 0.0f;

    // Benchmark iterations
    timer.Start();
    for (int iter = 0; iter < config.iterations; ++iter) {
        CUDA_CHECK(cudaMemcpyAsync(params.peer_comm_buffer_ptrs[config.local_rank], d_sendbuff, n_elts * config.element_size, cudaMemcpyDeviceToDevice, config.stream));

        AllReduceParams iter_params = params;
        iter_params.barrier_flag = barrier_counter++;

        invokeMultiGpuBarrier(iter_params, config.stream);

        customAllReduce(iter_params, d_recvbuff, n_elts, config.element_size, config.dtype, AllReduceStrategyType::TWOSHOT, config.stream);
    }
    total_time += timer.Stop();

    BenchmarkResults results{};
    results.implementation = kTwoShotLabel;
    results.num_elements = n_elts;
    results.message_bytes = n_elts * config.element_size;
    results.latency_ms = total_time / config.iterations;
    results.bandwidth_gbps = CalculateBandwidth(results.message_bytes, results.latency_ms);

    return results;
}

/**
 * @brief Cleanup allocated resources
 *
 * @param d_sendbuff Send buffer for data to reduce
 * @param d_recvbuff Receive buffer for results
 * @param d_comm_buf Communication buffer for peer communication
 * @param d_barrier_in Barrier input buffer for synchronization
 * @param d_barrier_out Barrier output buffer for synchronization
 */
void Cleanup(void* d_sendbuff, void* d_recvbuff, void* d_comm_buf, uint32_t* d_barrier_in, uint32_t* d_barrier_out) {
    CUDA_CHECK(cudaFree(d_sendbuff));
    CUDA_CHECK(cudaFree(d_recvbuff));
    CUDA_CHECK(cudaFree(d_comm_buf));
    CUDA_CHECK(cudaFree(d_barrier_in));
    CUDA_CHECK(cudaFree(d_barrier_out));
}

/**
 * @brief Launch benchmarks for different message sizes
 *
 * @param n_values Vector of multipliers for message sizes
 * @param config Benchmark configuration
 * @param csv_writer Pointer to CSV writer for results
 */
void LaunchBenchmarks(std::vector<int> n_values, const BenchmarkConfig& config, std::unique_ptr<CsvWriter>& csv_writer) {
    // Benchmark loop
    for (int n_mult : n_values) {
        const std::size_t bytes = static_cast<std::size_t>(n_mult) * 10240;
        const std::size_t current_n_elts = bytes / config.element_size;

        if (config.world_rank == 0) {
            std::cout << "Benchmarking " << current_n_elts << " elements (" << bytes << " Bytes)..." << std::endl;
        }

        void* d_sendbuff = nullptr;
        void* d_recvbuff = nullptr;
        void* d_comm_buf = nullptr;
        uint32_t* d_barrier_in = nullptr;
        uint32_t* d_barrier_out = nullptr;

        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_sendbuff, bytes));
        CUDA_CHECK(cudaMalloc(&d_recvbuff, bytes));
        CUDA_CHECK(cudaMalloc(&d_comm_buf, bytes));
        CUDA_CHECK(cudaMalloc(&d_barrier_in, 1024 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_barrier_out, 1024 * sizeof(uint32_t)));

        // Set device barrier to zero
        CUDA_CHECK(cudaMemset(d_barrier_in, 0, 1024 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_barrier_out, 0, 1024 * sizeof(uint32_t)));

        // Initialize data
        if (config.dtype == datatype_enum::TYPE_FP32) {
            std::vector<float> h_send(current_n_elts);
            InitVector<float>(h_send.data(), current_n_elts);
            CUDA_CHECK(cudaMemcpy(d_sendbuff, h_send.data(), bytes, cudaMemcpyHostToDevice));
        } else {
            std::vector<__half> h_send(current_n_elts);
            InitVector<__half>(h_send.data(), current_n_elts);
            CUDA_CHECK(cudaMemcpy(d_sendbuff, h_send.data(), bytes, cudaMemcpyHostToDevice));
        }

        // Setup IPC
        std::vector<void*> peer_comm_ptrs;
        std::vector<uint32_t*> peer_barrier_in_ptrs;
        std::vector<uint32_t*> peer_barrier_out_ptrs;

        ShareIpcPtrs(d_comm_buf, peer_comm_ptrs, config.local_rank, config.ranks_per_node);
        ShareIpcPtrs(d_barrier_in, peer_barrier_in_ptrs, config.local_rank, config.ranks_per_node);
        ShareIpcPtrs(d_barrier_out, peer_barrier_out_ptrs, config.local_rank, config.ranks_per_node);

        // Setup AllReduce parameters
        AllReduceParams params{};
        for (int r = 0; r < config.ranks_per_node; ++r) {
            params.peer_comm_buffer_ptrs[r] = peer_comm_ptrs[r];
            params.peer_barrier_ptrs_in[r] = peer_barrier_in_ptrs[r];
            params.peer_barrier_ptrs_out[r] = peer_barrier_out_ptrs[r];
        }
        params.ranks_per_node = config.ranks_per_node;
        params.rank = config.local_rank;
        params.local_rank = config.local_rank;

        uint32_t barrier_counter = 1;

        // Benchmark NCCL
        BenchmarkResults nccl_results = BenchmarkNccl(d_sendbuff, d_recvbuff, current_n_elts, config);

        // Get golden reference from NCCL result
        BenchmarkResults validation_results{};
        if (config.dtype == datatype_enum::TYPE_FP32) {
            std::vector<float> h_golden(current_n_elts);
            CUDA_CHECK(cudaMemcpy(h_golden.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            validation_results = ValidateResults<float>(d_recvbuff, h_golden, config, current_n_elts);
        } else {
            std::vector<__half> h_golden(current_n_elts);
            CUDA_CHECK(cudaMemcpy(h_golden.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            validation_results = ValidateResults<__half>(d_recvbuff, h_golden, config, current_n_elts);
        }

        // Merge validation results with NCCL results
        nccl_results.accuracy_pct = validation_results.accuracy_pct;
        nccl_results.max_abs_error = validation_results.max_abs_error;
        nccl_results.mean_abs_error = validation_results.mean_abs_error;
        nccl_results.max_rel_error = validation_results.max_rel_error;
        nccl_results.rmse = validation_results.rmse;

        // Write NCCL results to CSV
        if (csv_writer) {
            csv_writer->WriteRow(nccl_results.implementation, nccl_results.message_bytes, nccl_results.latency_ms, nccl_results.bandwidth_gbps, nccl_results.accuracy_pct,
                                 nccl_results.max_abs_error, nccl_results.mean_abs_error, nccl_results.max_rel_error, nccl_results.rmse);
        }

        // Reset buffer and benchmark OneShot
        CUDA_CHECK(cudaMemset(d_recvbuff, 0, bytes));
        BenchmarkResults oneshot_results = BenchmarkOneshot(d_sendbuff, d_recvbuff, current_n_elts, params, config, barrier_counter);

        // Validate OneShot results
        if (config.dtype == datatype_enum::TYPE_FP32) {
            std::vector<float> h_golden(current_n_elts);
            CUDA_CHECK(cudaMemcpy(h_golden.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            validation_results = ValidateResults<float>(d_recvbuff, h_golden, config, current_n_elts);
        } else {
            std::vector<__half> h_golden(current_n_elts);
            CUDA_CHECK(cudaMemcpy(h_golden.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            validation_results = ValidateResults<__half>(d_recvbuff, h_golden, config, current_n_elts);
        }

        oneshot_results.accuracy_pct = validation_results.accuracy_pct;
        oneshot_results.max_abs_error = validation_results.max_abs_error;
        oneshot_results.mean_abs_error = validation_results.mean_abs_error;
        oneshot_results.max_rel_error = validation_results.max_rel_error;
        oneshot_results.rmse = validation_results.rmse;

        if (csv_writer) {
            csv_writer->WriteRow(oneshot_results.implementation, oneshot_results.message_bytes, oneshot_results.latency_ms, oneshot_results.bandwidth_gbps,
                                 oneshot_results.accuracy_pct, oneshot_results.max_abs_error, oneshot_results.mean_abs_error, oneshot_results.max_rel_error,
                                 oneshot_results.rmse);
        }

        // Reset buffer and benchmark TwoShot
        CUDA_CHECK(cudaMemset(d_recvbuff, 0, bytes));
        BenchmarkResults twoshot_results = BenchmarkTwoshot(d_sendbuff, d_recvbuff, current_n_elts, params, config, barrier_counter);

        // Validate TwoShot results (similar to above)
        if (config.dtype == datatype_enum::TYPE_FP32) {
            std::vector<float> h_golden(current_n_elts);
            CUDA_CHECK(cudaMemcpy(h_golden.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            validation_results = ValidateResults<float>(d_recvbuff, h_golden, config, current_n_elts);
        } else {
            std::vector<__half> h_golden(current_n_elts);
            CUDA_CHECK(cudaMemcpy(h_golden.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            validation_results = ValidateResults<__half>(d_recvbuff, h_golden, config, current_n_elts);
        }

        twoshot_results.accuracy_pct = validation_results.accuracy_pct;
        twoshot_results.max_abs_error = validation_results.max_abs_error;
        twoshot_results.mean_abs_error = validation_results.mean_abs_error;
        twoshot_results.max_rel_error = validation_results.max_rel_error;
        twoshot_results.rmse = validation_results.rmse;

        if (csv_writer) {
            csv_writer->WriteRow(twoshot_results.implementation, twoshot_results.message_bytes, twoshot_results.latency_ms, twoshot_results.bandwidth_gbps,
                                 twoshot_results.accuracy_pct, twoshot_results.max_abs_error, twoshot_results.mean_abs_error, twoshot_results.max_rel_error,
                                 twoshot_results.rmse);
            csv_writer->Flush();
        }

        // Cleanup memory for this iteration
        Cleanup(d_sendbuff, d_recvbuff, d_comm_buf, d_barrier_in, d_barrier_out);
    }
}

/**
 * @brief Main function to benchmark custom all-reduce implementations
 *
 * @param argc Argument count
 * @param argv Argument vector
 * @return int
 */
int main(int argc, char** argv) {
    // Set random seed for reproducibility
    srand(42);

    MPI_CHECK(MPI_Init(&argc, &argv));

    // Parse command line arguments
    std::string dtype_s = "fp32";
    int iterations = 10;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dtype")) dtype_s = argv[++i];
        if (!strcmp(argv[i], "--iters")) iterations = std::atoi(argv[++i]);
    }

    // Setup benchmark configuration
    BenchmarkConfig config{};
    config.dtype = (dtype_s == "fp16") ? datatype_enum::TYPE_FP16 : datatype_enum::TYPE_FP32;
    config.element_size = (dtype_s == "fp16") ? sizeof(__half) : sizeof(float);
    config.iterations = iterations;

    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &config.world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &config.world_rank));
    config.local_rank = GetLocalRank();
    config.ranks_per_node = config.world_size;

    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(config.local_rank));
    CUDA_CHECK(cudaStreamCreate(&config.stream));

    // Initialize NCCL
    ncclUniqueId id{};
    if (config.world_rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCL_CHECK(ncclCommInitRank(&config.nccl_comm, config.world_size, id, config.world_rank));

    // Create CSV writer (only on rank 0)
    std::unique_ptr<CsvWriter> csv_writer;
    if (config.world_rank == 0) {
        std::filesystem::create_directories(kDirectoryPath);
        std::string filename = kDirectoryPath + "/" + kFileName + "_" + GenerateTimestamp() + "_" + dtype_s + ".csv";
        csv_writer = std::make_unique<CsvWriter>(filename, true);
    }

    // Message size multipliers
    std::vector<int> n_values;
    for (int n = 1; n < 16; ++n) n_values.push_back(n);
    for (int n = 16; n < 128; n += 16) n_values.push_back(n);
    for (int n = 128; n < 1024; n += 128) n_values.push_back(n);

    // Launch benchmarks
    LaunchBenchmarks(n_values, config, csv_writer);

    // Final cleanup
    if (csv_writer) {
        csv_writer->Close();
    }

    NCCL_CHECK(ncclCommDestroy(config.nccl_comm));
    CUDA_CHECK(cudaStreamDestroy(config.stream));
    MPI_CHECK(MPI_Finalize());

    if (config.world_rank == 0) {
        std::cout << "Benchmark completed successfully!" << std::endl;
    }

    return 0;
}
