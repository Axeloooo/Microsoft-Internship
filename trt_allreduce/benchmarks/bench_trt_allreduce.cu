/**
 * @file bench_trt_allreduce.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Benchmarking for TensorRT AllReduce implementations.
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "common/dataType.h"
#include "kernels/customAllReduceKernels.h"
#include "kernels/userbuffers/ub_allocator.h"
#include "kernels/userbuffers/ub_interface.h"
#include "kernels/userbuffers/userbuffers.h"
#include "utils/checker.h"
#include "utils/csv_writer.h"
#include "utils/cuda_timer.cuh"
#include "utils/data.h"

using trt_allreduce::common::getDTypeSize;
using trt_allreduce::kernels::AllReduceFusionOp;
using trt_allreduce::kernels::AllReduceParams;
using trt_allreduce::kernels::AllReduceStrategyConfig;
using trt_allreduce::kernels::AllReduceStrategyType;
using trt_allreduce::kernels::customAllReduce;
using trt_allreduce::kernels::ub::allreduce2_userbuff_inplace_launcher;
using trt_allreduce::kernels::ub::communicator;
using trt_allreduce::runtime::WorldConfig;
using trt_allreduce::runtime::ub::ub_allocate;
using trt_allreduce::runtime::ub::ub_comm;
using trt_allreduce::runtime::ub::ub_deallocate;
using trt_allreduce::runtime::ub::ub_get;
using trt_allreduce::runtime::ub::ub_initialize;
using trt_allreduce::runtime::ub::ub_is_initialized;
using trt_allreduce::runtime::ub::UBBuffer;
using trt_allreduce::utils::CsvWriter;
using trt_allreduce::utils::CudaTimer;
using trt_allreduce::utils::InitVector;

namespace trt_allreduce::benchmark {

// -------------------------------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------------------------------

constexpr std::size_t kSweepSize = 10240;
constexpr std::size_t kWarmupIters = 10;
constexpr double kMilliToSec = 1e-3;
constexpr double kBytesToGigabytes = 1e-9;
constexpr char const* kNcclLabel = "NCCL";
constexpr char const* kOneShotLabel = "One-Shot";
constexpr char const* kTwoShotLabel = "Two-Shot";
constexpr char const* kUbLabel = "User-Buffers";
constexpr char const* kFileName = "benchmark_trt_allreduce";
constexpr char const* kDirectoryPath = "data";

// -------------------------------------------------------------------------------------------------
// Benchmark configuration and results
// -------------------------------------------------------------------------------------------------

/**
 * @brief Structure to hold benchmark configuration parameters.
 *
 */
struct BenchmarkConfig {
    nvinfer1::DataType dtype{};
    std::size_t element_size{0};
    int iterations{10};
    int world_size{0};
    int world_rank{0};
    int local_rank{0};
    int ranks_per_node{0};
    cudaStream_t stream{};
    ncclComm_t nccl_comm{};
};

/**
 * @brief Structure to hold benchmark results.
 *
 */
struct BenchmarkResults {
    std::string implementation;
    std::size_t num_elements{0};
    std::size_t message_bytes{0};
    float latency_ms{0.0F};
    double bandwidth_gbps{0.0};
    double accuracy_pct{0.0};
    float max_abs_error{0.0F};
    double mean_abs_error{0.0};
    float max_rel_error{0.0F};
    double rmse{0.0};
};

// -------------------------------------------------------------------------------------------------
// Small helpers
// -------------------------------------------------------------------------------------------------

/**
 * @brief Generate a timestamp string in the format YYYYMMDD-HHMMSS.
 *
 * @return std::string
 */
std::string GenerateTimestamp() {
    // Get the current time
    std::time_t const now = std::time(nullptr);

    // Convert to local time and format it
    std::tm* const tm_now = std::localtime(&now);

    // Create a string stream to format the time
    std::ostringstream oss;
    oss << std::put_time(tm_now, "%Y%m%d-%H%M%S");
    return oss.str();
}

/**
 * @brief Calculate bandwidth in GB/s given bytes transferred and latency (ms).
 *
 * @param bytes Number of bytes transferred
 * @param latency_ms Latency in milliseconds
 * @return double
 */
double CalculateBandwidth(std::size_t bytes, float latency_ms) {
    if (latency_ms <= 0.0F) {
        return 0.0;
    }
    return static_cast<double>(bytes) / (latency_ms * kMilliToSec) * kBytesToGigabytes;
}

/**
 * @brief Get the Local Rank object
 *
 * @return int
 */
int GetLocalRank() {
    // Obtain the rank in the global communicator (MPI_COMM_WORLD)
    int world_rank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    // Split the communicator by shared memory type to get the local rank
    MPI_Comm node_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm));

    // Get the local rank within the node communicator
    int local_rank;
    MPI_CHECK(MPI_Comm_rank(node_comm, &local_rank));

    // Free the node communicator
    MPI_CHECK(MPI_Comm_free(&node_comm));

    return local_rank;
}

/**
 * @brief Get the Local Size object
 *
 * @tparam T
 * @param local_ptr Local pointer to the IPC memory
 * @param remote_ptrs Vector to hold remote pointers
 * @param local_rank Local rank of the process
 * @param ranks_per_node Number of ranks per node
 */
template <typename T>
void ShareIpcPtrs(T local_ptr, std::vector<T>& remote_ptrs, int local_rank, int ranks_per_node) {
    // Create a vector to hold handles for all ranks
    std::vector<cudaIpcMemHandle_t> handles(ranks_per_node);

    // Create a local handle for the current rank
    cudaIpcMemHandle_t local_handle{};
    CUDA_CHECK(cudaIpcGetMemHandle(&local_handle, static_cast<void*>(local_ptr)));

    // Gather handles on host using MPI
    MPI_CHECK(MPI_Allgather(&local_handle, sizeof(local_handle), MPI_BYTE, handles.data(), sizeof(local_handle), MPI_BYTE, MPI_COMM_WORLD));

    // Resize remote pointers vector to hold pointers for all ranks
    remote_ptrs.resize(ranks_per_node);

    // Open IPC memory handles for all ranks
    for (std::size_t r = 0; r < ranks_per_node; ++r) {
        if (r == local_rank) {
            remote_ptrs[r] = local_ptr;
        } else {
            void* tmp = nullptr;
            CUDA_CHECK(cudaIpcOpenMemHandle(&tmp, handles[r], cudaIpcMemLazyEnablePeerAccess));
            remote_ptrs[r] = static_cast<T>(tmp);
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Validation utilities
// -------------------------------------------------------------------------------------------------

/**
 * @brief Validate results against a golden reference.
 *
 * @tparam T
 * @param d_buff Device buffer containing results
 * @param golden Golden reference data
 * @param config Benchmark configuration parameters
 * @param elements Number of elements to validate
 * @return BenchmarkResults
 */
template <typename T>
BenchmarkResults ValidateResults(void* d_buff, std::vector<T> const& golden, BenchmarkConfig const& config, std::size_t elements) {
    // Store actual results in a vector
    std::vector<T> actual(elements);
    CUDA_CHECK(cudaMemcpy(actual.data(), d_buff, elements * sizeof(T), cudaMemcpyDeviceToHost));

    // Calculate error metrics
    float const eps = std::is_same_v<T, float> ? 1e-5F : 1e-2F;
    std::size_t correct = 0;
    double abs_sum = 0.0;
    double sq_err_sum = 0.0;
    float max_abs_err = 0.0F;
    float max_rel_err = 0.0F;

    // Compare actual results with golden reference
    for (std::size_t i = 0; i < elements; ++i) {
        float diff = static_cast<float>(actual[i]) - static_cast<float>(golden[i]);
        float abs_err = std::fabs(diff);
        abs_sum += abs_err;
        sq_err_sum += diff * diff;
        max_abs_err = std::max(max_abs_err, abs_err);
        float denom = std::fabs(static_cast<float>(golden[i])) + 1e-12F;
        float rel_err = abs_err / denom;
        max_rel_err = std::max(max_rel_err, rel_err);
        if (abs_err < eps) {
            ++correct;
        }
    }

    // Calculate results
    BenchmarkResults r{};
    r.num_elements = elements;
    r.message_bytes = elements * config.element_size;
    r.accuracy_pct = 100.0 * static_cast<double>(correct) / static_cast<double>(elements);
    r.max_abs_error = max_abs_err;
    r.mean_abs_error = abs_sum / static_cast<double>(elements);
    r.max_rel_error = max_rel_err;
    r.rmse = std::sqrt(sq_err_sum / static_cast<double>(elements));
    return r;
}

// -------------------------------------------------------------------------------------------------
// Helper: fill AllReduceParams for this benchmark.
// -------------------------------------------------------------------------------------------------

/**
 * @brief Create AllReduceParams structure for the custom all-reduce operation.
 *
 * @param sendbuff Input data buffer (device memory)
 * @param recvbuff Output data buffer (device memory)
 * @param elements Number of elements to reduce
 * @param config Benchmark configuration parameters
 * @param peer_comm_ptrs List of peer communication buffer pointers
 * @param peer_barrier_in_ptrs List of peer barrier input pointers
 * @param peer_barrier_out_ptrs List of peer barrier output pointers
 * @param barrier_flag Barrier flag to synchronize peers
 * @return AllReduceParams
 */
inline AllReduceParams MakeMultiShotParams(void* sendbuff, void* recvbuff, std::size_t elements, BenchmarkConfig const& config, std::vector<void*> const& peer_comm_ptrs,
                                           std::vector<uint32_t*> const& peer_barrier_in_ptrs, std::vector<uint32_t*> const& peer_barrier_out_ptrs, uint32_t barrier_flag) {
    // Create AllReduceParams structure
    AllReduceParams p{};
    p.elts_total = elements;
    p.ranks_per_node = static_cast<std::size_t>(config.ranks_per_node);
    p.local_rank = static_cast<std::size_t>(config.local_rank);
    p.barrier_flag = barrier_flag;
    p.local_input_buffer_ptr = sendbuff;
    p.local_output_buffer_ptr = recvbuff;

    // Set peer pointers
    for (std::size_t r = 0; r < config.ranks_per_node; ++r) {
        p.peer_comm_buffer_ptrs[r] = (r < static_cast<int>(peer_comm_ptrs.size())) ? peer_comm_ptrs[r] : nullptr;
        p.peer_barrier_ptrs_in[r] = (r < static_cast<int>(peer_barrier_in_ptrs.size())) ? peer_barrier_in_ptrs[r] : nullptr;
        p.peer_barrier_ptrs_out[r] = (r < static_cast<int>(peer_barrier_out_ptrs.size())) ? peer_barrier_out_ptrs[r] : nullptr;
    }

    // Set additional parameters
    p.elts_per_rank = 0;
    p.elts_per_block = 0;
    p.rank_offset = 0;

    // Disable fusion
    p.fusion_params = trt_allreduce::kernels::AllReduceFusionParams{};
    p.fusion_params.eps = 0.0F;
    p.fusion_params.hidden_size = 0;
    return p;
}

// -------------------------------------------------------------------------------------------------
// NCCL baseline
// -------------------------------------------------------------------------------------------------

/**
 * @brief Run a benchmark using NCCL all-reduce.
 *
 * @param d_sendbuff Input data buffer (device memory)
 * @param d_recvbuff Output data buffer (device memory)
 * @param elements Number of elements to reduce
 * @param config Benchmark configuration parameters
 * @return BenchmarkResults
 */
BenchmarkResults BenchmarkNccl(void* d_sendbuff, void* d_recvbuff, std::size_t elements, BenchmarkConfig const& config) {
    // Warm-up phase
    for (std::size_t i = 0; i < kWarmupIters; ++i) {
        NCCL_CHECK(
            ncclAllReduce(d_sendbuff, d_recvbuff, elements, (config.dtype == nvinfer1::DataType::kFLOAT ? ncclFloat32 : ncclHalf), ncclSum, config.nccl_comm, config.stream));
    }

    // Perform NCCL all-reduce
    CudaTimer timer(config.stream);
    timer.Start();
    for (std::size_t it = 0; it < config.iterations; ++it) {
        NCCL_CHECK(
            ncclAllReduce(d_sendbuff, d_recvbuff, elements, (config.dtype == nvinfer1::DataType::kFLOAT ? ncclFloat32 : ncclHalf), ncclSum, config.nccl_comm, config.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
    float total_time = timer.Stop();

    // Store results
    BenchmarkResults r{};
    r.implementation = kNcclLabel;
    r.num_elements = elements;
    r.message_bytes = elements * config.element_size;
    r.latency_ms = total_time / static_cast<float>(config.iterations);
    r.bandwidth_gbps = CalculateBandwidth(r.message_bytes, r.latency_ms);
    return r;
}

// -------------------------------------------------------------------------------------------------
// Custom One‑shot
// -------------------------------------------------------------------------------------------------

/**
 * @brief Run a benchmark using the custom one-shot all-reduce implementation.
 *
 * @param d_sendbuff Input data buffer (device memory)
 * @param d_recvbuff Output data buffer (device memory)
 * @param elements Number of elements to reduce
 * @param peer_comm_ptrs List of peer communication buffer pointers
 * @param peer_barrier_in_ptrs List of peer barrier input pointers
 * @param peer_barrier_out_ptrs List of peer barrier output pointers
 * @param config Benchmark configuration parameters
 * @param barrier_counter Barrier counter to synchronize peers
 * @return BenchmarkResults
 */
BenchmarkResults BenchmarkOneshot(void* d_sendbuff, void* d_recvbuff, std::size_t elements, std::vector<void*> const& peer_comm_ptrs,
                                  std::vector<uint32_t*> const& peer_barrier_in_ptrs, std::vector<uint32_t*> const& peer_barrier_out_ptrs, BenchmarkConfig const& config,
                                  uint32_t& barrier_counter) {
    // Warm-up phase
    for (std::size_t i = 0; i < kWarmupIters; ++i) {
        AllReduceParams p = MakeMultiShotParams(d_sendbuff, d_recvbuff, elements, config, peer_comm_ptrs, peer_barrier_in_ptrs, peer_barrier_out_ptrs, barrier_counter++);
        customAllReduce(p, config.dtype, AllReduceStrategyType::ONESHOT, static_cast<AllReduceStrategyConfig>(0), AllReduceFusionOp::NONE, config.stream);
    }

    // Perform One-shot all-reduce
    CudaTimer timer(config.stream);
    timer.Start();
    for (std::size_t it = 0; it < config.iterations; ++it) {
        AllReduceParams p = MakeMultiShotParams(d_sendbuff, d_recvbuff, elements, config, peer_comm_ptrs, peer_barrier_in_ptrs, peer_barrier_out_ptrs, barrier_counter++);
        customAllReduce(p, config.dtype, AllReduceStrategyType::ONESHOT, static_cast<AllReduceStrategyConfig>(0), AllReduceFusionOp::NONE, config.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
    float total_time = timer.Stop();

    // Store results
    BenchmarkResults r{};
    r.implementation = kOneShotLabel;
    r.num_elements = elements;
    r.message_bytes = elements * config.element_size;
    r.latency_ms = total_time / static_cast<float>(config.iterations);
    r.bandwidth_gbps = CalculateBandwidth(r.message_bytes, r.latency_ms);
    return r;
}

// -------------------------------------------------------------------------------------------------
// Custom Two‑shot
// -------------------------------------------------------------------------------------------------

/**
 * @brief Run a benchmark using the custom two-shot all-reduce implementation.
 *
 * @param d_sendbuff Input data buffer (device memory)
 * @param d_recvbuff Output data buffer (device memory)
 * @param elements Number of elements to reduce
 * @param peer_comm_ptrs List of peer communication buffer pointers
 * @param peer_barrier_in_ptrs List of peer barrier input pointers
 * @param peer_barrier_out_ptrs List of peer barrier output pointers
 * @param config Benchmark configuration parameters
 * @param barrier_counter Barrier counter to synchronize peers
 * @return BenchmarkResults
 */
BenchmarkResults BenchmarkTwoshot(void* d_sendbuff, void* d_recvbuff, std::size_t elements, std::vector<void*> const& peer_comm_ptrs,
                                  std::vector<uint32_t*> const& peer_barrier_in_ptrs, std::vector<uint32_t*> const& peer_barrier_out_ptrs, BenchmarkConfig const& config,
                                  uint32_t& barrier_counter) {
    // Warm-up phase
    for (std::size_t i = 0; i < kWarmupIters; ++i) {
        AllReduceParams p = MakeMultiShotParams(d_sendbuff, d_recvbuff, elements, config, peer_comm_ptrs, peer_barrier_in_ptrs, peer_barrier_out_ptrs, barrier_counter++);
        customAllReduce(p, config.dtype, AllReduceStrategyType::TWOSHOT, static_cast<AllReduceStrategyConfig>(0), AllReduceFusionOp::NONE, config.stream);
    }

    // Perform Two-shot all-reduce
    CudaTimer timer(config.stream);
    timer.Start();
    for (std::size_t it = 0; it < config.iterations; ++it) {
        AllReduceParams p = MakeMultiShotParams(d_sendbuff, d_recvbuff, elements, config, peer_comm_ptrs, peer_barrier_in_ptrs, peer_barrier_out_ptrs, barrier_counter++);
        customAllReduce(p, config.dtype, AllReduceStrategyType::TWOSHOT, static_cast<AllReduceStrategyConfig>(0), AllReduceFusionOp::NONE, config.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
    float total_time = timer.Stop();

    // Store results
    BenchmarkResults r{};
    r.implementation = kTwoShotLabel;
    r.num_elements = elements;
    r.message_bytes = elements * config.element_size;
    r.latency_ms = total_time / static_cast<float>(config.iterations);
    r.bandwidth_gbps = CalculateBandwidth(r.message_bytes, r.latency_ms);
    return r;
}

// -------------------------------------------------------------------------------------------------
// User-buffer all-reduce
// -------------------------------------------------------------------------------------------------

/**
 * @brief Run a benchmark using the custom user-buffer all-reduce implementation.
 *
 * @param d_sendbuff Input data buffer (device memory)
 * @param d_recvbuff Output data buffer (device memory)
 * @param elements Number of elements to reduce
 * @param config Benchmark configuration parameters
 * @param userbuff User-buffer to use for the all-reduce operation
 * @return BenchmarkResults
 */
BenchmarkResults BenchmarkUb(void* d_sendbuff, void* d_recvbuff, std::size_t elements, BenchmarkConfig const& config, UBBuffer& userbuff) {
    std::size_t bytes = elements * config.element_size;

    // Extra safety-check
    if (userbuff.invalid() || userbuff.size < bytes) {
        throw std::runtime_error("User-buffer is invalid or too small for current benchmark.");
    }

    communicator* comm = ub_comm();

    // Copy input data to user-buffer (first time, before warm-up)
    CUDA_CHECK(cudaMemcpyAsync(userbuff.addr, d_sendbuff, bytes, cudaMemcpyDeviceToDevice, config.stream));

    // Warm-up phase
    for (std::size_t i = 0; i < kWarmupIters; ++i) {
        allreduce2_userbuff_inplace_launcher(userbuff.handle, 0, elements, config.dtype, comm, config.stream);
    }

    // Perform user-buffer all-reduce
    CudaTimer timer(config.stream);
    timer.Start();
    for (std::size_t it = 0; it < config.iterations; ++it) {
        CUDA_CHECK(cudaMemcpyAsync(userbuff.addr, d_sendbuff, bytes, cudaMemcpyDeviceToDevice, config.stream));
        allreduce2_userbuff_inplace_launcher(userbuff.handle, 0, elements, config.dtype, comm, config.stream);
        CUDA_CHECK(cudaMemcpy(d_recvbuff, userbuff.addr, bytes, cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
    float total_time = timer.Stop();

    // Store results
    BenchmarkResults r{};
    r.implementation = kUbLabel;
    r.num_elements = elements;
    r.message_bytes = bytes;
    r.latency_ms = total_time / static_cast<float>(config.iterations);
    r.bandwidth_gbps = CalculateBandwidth(bytes, r.latency_ms);
    return r;
}
}  // namespace trt_allreduce::benchmark

// -------------------------------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------------------------------

using namespace trt_allreduce::benchmark;

/**
 * @brief Main function for the benchmark.
 *
 * @param argc Argument count
 * @param argv Argument vector
 * @return int
 */
int main(int argc, char** argv) {
    // Initialize random seed for reproducibility
    std::srand(42);

    // Disable [TensorRT-LLM][DEBUG] AllReduceDispatch enabled
    setenv("TLLM_LOG_LEVEL", "WARNING", 1);

    // Initialize MPI
    MPI_CHECK(MPI_Init(&argc, &argv));

    // Set default hyperparameters
    std::string dtype_s = "fp32";
    int iterations = 10;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--dtype") && i + 1 < argc) {
            dtype_s = argv[++i];
        } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        }
    }

    // Create benchmark configuration
    BenchmarkConfig config{};
    config.dtype = (dtype_s == "fp16") ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
    config.element_size = getDTypeSize(config.dtype);
    config.iterations = iterations;

    // Set up local and global ranks
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &config.world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &config.world_rank));
    config.local_rank = GetLocalRank();
    config.ranks_per_node = config.world_size;

    // Set CUDA device and create stream
    CUDA_CHECK(cudaSetDevice(config.local_rank));
    CUDA_CHECK(cudaStreamCreate(&config.stream));

    // Get NCCL id
    ncclUniqueId nccl_id;
    if (config.world_rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }

    // Broadcast NCCL ID to all ranks
    MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Initialize NCCL communicator
    NCCL_CHECK(ncclCommInitRank(&config.nccl_comm, config.world_size, nccl_id, config.world_rank));

    // Create directory for results
    std::unique_ptr<CsvWriter> csv_writer;
    if (config.world_rank == 0) {
        std::filesystem::create_directories(kDirectoryPath);
        std::string filename = std::string(kDirectoryPath) + "/" + kFileName + "_" + GenerateTimestamp() + "_" + dtype_s + ".csv";
        csv_writer = std::make_unique<CsvWriter>(filename, true);
    }

    // Message size sweep
    std::vector<int> n_values;
    for (int n = 1; n < 16; ++n) n_values.push_back(n);
    for (int n = 16; n < 128; n += 16) n_values.push_back(n);
    for (int n = 128; n < 1024; n += 128) n_values.push_back(n);

    // Calculate maximum number of elements and bytes
    std::size_t const max_n = static_cast<std::size_t>(n_values.back()) * kSweepSize;
    std::size_t const max_bytes = max_n * config.element_size;

    // Initialize user-buffers (fp16 only) and allocate once
    UBBuffer ubuff;
    if (config.dtype == nvinfer1::DataType::kHALF) {
        ub_initialize(config.world_size);
        ubuff = ub_allocate(max_bytes);
    }

    // Allocate device buffers for send and receive
    void* d_sendbuff = nullptr;
    void* d_recvbuff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sendbuff, max_bytes));
    CUDA_CHECK(cudaMalloc(&d_recvbuff, max_bytes));

    // Fill send buffer with values
    if (config.dtype == nvinfer1::DataType::kFLOAT) {
        std::vector<float> h(max_n);
        InitVector<float>(h.data(), max_n);
        CUDA_CHECK(cudaMemcpy(d_sendbuff, h.data(), max_bytes, cudaMemcpyHostToDevice));
    } else {
        std::vector<__half> h(max_n);
        InitVector<__half>(h.data(), max_n);
        CUDA_CHECK(cudaMemcpy(d_sendbuff, h.data(), max_bytes, cudaMemcpyHostToDevice));
    }

    // Allocate shared memory for communication buffers and barriers
    void* d_comm_buf = nullptr;
    uint32_t* d_barrier_in = nullptr;
    uint32_t* d_barrier_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_comm_buf, max_bytes));
    CUDA_CHECK(cudaMalloc(&d_barrier_in, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_barrier_out, sizeof(uint32_t)));

    // Initialize communication buffers and barriers
    std::vector<void*> peer_comm_ptrs;
    std::vector<uint32_t*> peer_barrier_in_ptrs;
    std::vector<uint32_t*> peer_barrier_out_ptrs;
    ShareIpcPtrs(d_comm_buf, peer_comm_ptrs, config.local_rank, config.ranks_per_node);
    ShareIpcPtrs(d_barrier_in, peer_barrier_in_ptrs, config.local_rank, config.ranks_per_node);
    ShareIpcPtrs(d_barrier_out, peer_barrier_out_ptrs, config.local_rank, config.ranks_per_node);

    // Set initial barrier values
    uint32_t barrier_counter = 1;

    // Loop over message sizes
    for (int n_mult : n_values) {
        // Calculate number of elements and bytes for this sweep
        std::size_t const elements = static_cast<std::size_t>(n_mult) * kSweepSize;
        std::size_t const bytes = elements * config.element_size;

        // Create device buffers for this sweep
        std::vector<float> golden_fp32;
        std::vector<__half> golden_fp16;

        // Create a validation results object
        BenchmarkResults val_res{};

        if (config.world_rank == 0) {
            std::cout << "Running benchmark for " << elements << " elements (" << bytes << " Bytes)..." << std::endl;
        }

        // NCCL baseline
        BenchmarkResults nccl_res = BenchmarkNccl(d_sendbuff, d_recvbuff, elements, config);

        // Validate NCCL results and capture golden reference for later validation
        if (config.dtype == nvinfer1::DataType::kFLOAT) {
            golden_fp32.resize(elements);
            CUDA_CHECK(cudaMemcpy(golden_fp32.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            val_res = ValidateResults<float>(d_recvbuff, golden_fp32, config, elements);
        } else {
            golden_fp16.resize(elements);
            CUDA_CHECK(cudaMemcpy(golden_fp16.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            val_res = ValidateResults<__half>(d_recvbuff, golden_fp16, config, elements);
        }

        // Store NCCL results
        nccl_res.accuracy_pct = val_res.accuracy_pct;
        nccl_res.max_abs_error = val_res.max_abs_error;
        nccl_res.mean_abs_error = val_res.mean_abs_error;
        nccl_res.max_rel_error = val_res.max_rel_error;
        nccl_res.rmse = val_res.rmse;

        if (csv_writer) {
            csv_writer->WriteRow(nccl_res.implementation, nccl_res.message_bytes, nccl_res.latency_ms, nccl_res.bandwidth_gbps, nccl_res.accuracy_pct, nccl_res.max_abs_error,
                                 nccl_res.mean_abs_error, nccl_res.max_rel_error, nccl_res.rmse);
        }

        // Reset and benchmark one-shot
        CUDA_CHECK(cudaMemset(d_recvbuff, 0, bytes));
        BenchmarkResults oneshot_res =
            BenchmarkOneshot(d_sendbuff, d_recvbuff, elements, peer_comm_ptrs, peer_barrier_in_ptrs, peer_barrier_out_ptrs, config, barrier_counter);

        // Validate one-shot (reuse golden)
        if (config.dtype == nvinfer1::DataType::kFLOAT) {
            val_res = ValidateResults<float>(d_recvbuff, golden_fp32, config, elements);
        } else {
            val_res = ValidateResults<__half>(d_recvbuff, golden_fp16, config, elements);
        }

        // Store one-shot results
        oneshot_res.accuracy_pct = val_res.accuracy_pct;
        oneshot_res.max_abs_error = val_res.max_abs_error;
        oneshot_res.mean_abs_error = val_res.mean_abs_error;
        oneshot_res.max_rel_error = val_res.max_rel_error;
        oneshot_res.rmse = val_res.rmse;

        if (csv_writer) {
            csv_writer->WriteRow(oneshot_res.implementation, oneshot_res.message_bytes, oneshot_res.latency_ms, oneshot_res.bandwidth_gbps, oneshot_res.accuracy_pct,
                                 oneshot_res.max_abs_error, oneshot_res.mean_abs_error, oneshot_res.max_rel_error, oneshot_res.rmse);
        }

        // Reset and benchmark two-shot
        CUDA_CHECK(cudaMemset(d_recvbuff, 0, bytes));
        BenchmarkResults twoshot_res =
            BenchmarkTwoshot(d_sendbuff, d_recvbuff, elements, peer_comm_ptrs, peer_barrier_in_ptrs, peer_barrier_out_ptrs, config, barrier_counter);

        // Validate two-shot (reuse golden)
        if (config.dtype == nvinfer1::DataType::kFLOAT) {
            val_res = ValidateResults<float>(d_recvbuff, golden_fp32, config, elements);
        } else {
            val_res = ValidateResults<__half>(d_recvbuff, golden_fp16, config, elements);
        }

        // Store two-shot results
        twoshot_res.accuracy_pct = val_res.accuracy_pct;
        twoshot_res.max_abs_error = val_res.max_abs_error;
        twoshot_res.mean_abs_error = val_res.mean_abs_error;
        twoshot_res.max_rel_error = val_res.max_rel_error;
        twoshot_res.rmse = val_res.rmse;

        if (csv_writer) {
            csv_writer->WriteRow(twoshot_res.implementation, twoshot_res.message_bytes, twoshot_res.latency_ms, twoshot_res.bandwidth_gbps, twoshot_res.accuracy_pct,
                                 twoshot_res.max_abs_error, twoshot_res.mean_abs_error, twoshot_res.max_rel_error, twoshot_res.rmse);
        }

        // Reset and benchmark user-buffers
        if (config.dtype == nvinfer1::DataType::kHALF) {
            CUDA_CHECK(cudaMemset(d_recvbuff, 0, bytes));
            BenchmarkResults ub_res = BenchmarkUb(d_sendbuff, d_recvbuff, elements, config, ubuff);

            // Validate user-buffers (reuse golden)
            val_res = ValidateResults<__half>(d_recvbuff, golden_fp16, config, elements);

            // Store user-buffers results
            ub_res.accuracy_pct = val_res.accuracy_pct;
            ub_res.max_abs_error = val_res.max_abs_error;
            ub_res.mean_abs_error = val_res.mean_abs_error;
            ub_res.max_rel_error = val_res.max_rel_error;
            ub_res.rmse = val_res.rmse;

            if (csv_writer) {
                csv_writer->WriteRow(ub_res.implementation, ub_res.message_bytes, ub_res.latency_ms, ub_res.bandwidth_gbps, ub_res.accuracy_pct, ub_res.max_abs_error,
                                     ub_res.mean_abs_error, ub_res.max_rel_error, ub_res.rmse);
            }
        }
    }

    // Clean-up resources
    if (config.dtype == nvinfer1::DataType::kHALF) {
        ub_deallocate(ubuff.addr);
    }

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
