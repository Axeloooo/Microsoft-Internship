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

#include "common/dataType.h"
#include "kernels/customAllReduceKernels.h"
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
using trt_allreduce::utils::CsvWriter;
using trt_allreduce::utils::CudaTimer;
using trt_allreduce::utils::InitVector;

// -------------------------------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------------------------------

constexpr std::size_t kSweepSize = 10240;
constexpr std::size_t kWarmupIters = 10;
constexpr char const* kNcclLabel = "NCCL";
constexpr char const* kOneShotLabel = "One-Shot";
constexpr char const* kTwoShotLabel = "Two-Shot";
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
    std::time_t const now = std::time(nullptr);
    std::tm* const tm_now = std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(tm_now, "%Y%m%d-%H%M%S");
    return oss.str();
}

/**
 * @brief Calculate bandwidth in GB/s given the number of bytes and latency in milliseconds.
 *
 * @param bytes
 * @param latency_ms
 * @return double
 */
double CalculateBandwidth(std::size_t bytes, float latency_ms) { return static_cast<double>(bytes) / (latency_ms * 1e-3) / 1e9; }

/**
 * @brief Get the Local Rank object
 *
 * @return int
 */
int GetLocalRank() {
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
 * @brief Get the Local Size object
 *
 * @tparam T
 * @param local_ptr
 * @param remote_ptrs
 * @param local_rank
 * @param ranks_per_node
 */
template <typename T>
void ShareIpcPtrs(T local_ptr, std::vector<T>& remote_ptrs, int local_rank, int ranks_per_node) {
    // Gather handles on host using MPI.
    std::vector<cudaIpcMemHandle_t> handles(ranks_per_node);
    cudaIpcMemHandle_t local_handle{};
    CUDA_CHECK(cudaIpcGetMemHandle(&local_handle, static_cast<void*>(local_ptr)));

    MPI_CHECK(MPI_Allgather(&local_handle, sizeof(local_handle), MPI_BYTE, handles.data(), sizeof(local_handle), MPI_BYTE, MPI_COMM_WORLD));

    remote_ptrs.resize(ranks_per_node);
    for (int r = 0; r < ranks_per_node; ++r) {
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
 * @param d_buff
 * @param golden
 * @param config
 * @param n_elts
 * @return BenchmarkResults
 */
template <typename T>
BenchmarkResults ValidateResults(void* d_buff, std::vector<T> const& golden, BenchmarkConfig const& config, std::size_t n_elts) {
    std::vector<T> actual(n_elts);
    CUDA_CHECK(cudaMemcpy(actual.data(), d_buff, n_elts * sizeof(T), cudaMemcpyDeviceToHost));

    float const eps = std::is_same_v<T, float> ? 1e-5F : 1e-2F;
    std::size_t correct = 0;
    double abs_sum = 0.0;
    double sq_err_sum = 0.0;
    float max_abs_err = 0.0F;
    float max_rel_err = 0.0F;

    for (std::size_t i = 0; i < n_elts; ++i) {
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

    BenchmarkResults r{};
    r.num_elements = n_elts;
    r.message_bytes = n_elts * config.element_size;
    r.accuracy_pct = 100.0 * static_cast<double>(correct) / static_cast<double>(n_elts);
    r.max_abs_error = max_abs_err;
    r.mean_abs_error = abs_sum / static_cast<double>(n_elts);
    r.max_rel_error = max_rel_err;
    r.rmse = std::sqrt(sq_err_sum / static_cast<double>(n_elts));
    return r;
}

// -------------------------------------------------------------------------------------------------
// Helper: fill AllReduceParams for this benchmark.
// -------------------------------------------------------------------------------------------------

/**
 * @brief Create AllReduceParams structure for the custom all-reduce operation.
 *
 * @param sendbuff
 * @param recvbuff
 * @param n_elts
 * @param cfg
 * @param peer_comm_ptrs
 * @param peer_barrier_in_ptrs
 * @param peer_barrier_out_ptrs
 * @param barrier_flag
 * @return AllReduceParams
 */
inline AllReduceParams MakeParams(void* sendbuff, void* recvbuff, std::size_t n_elts, BenchmarkConfig const& cfg, std::vector<void*> const& peer_comm_ptrs,
                                  std::vector<uint32_t*> const& peer_barrier_in_ptrs, std::vector<uint32_t*> const& peer_barrier_out_ptrs, uint32_t barrier_flag) {
    AllReduceParams p{};
    p.elts_total = n_elts;
    p.ranks_per_node = static_cast<std::size_t>(cfg.ranks_per_node);
    p.local_rank = static_cast<std::size_t>(cfg.local_rank);
    p.barrier_flag = barrier_flag;
    p.local_input_buffer_ptr = sendbuff;
    p.local_output_buffer_ptr = recvbuff;

    // peer arrays
    for (int r = 0; r < cfg.ranks_per_node; ++r) {
        p.peer_comm_buffer_ptrs[r] = (r < static_cast<int>(peer_comm_ptrs.size())) ? peer_comm_ptrs[r] : nullptr;
        p.peer_barrier_ptrs_in[r] = (r < static_cast<int>(peer_barrier_in_ptrs.size())) ? peer_barrier_in_ptrs[r] : nullptr;
        p.peer_barrier_ptrs_out[r] = (r < static_cast<int>(peer_barrier_out_ptrs.size())) ? peer_barrier_out_ptrs[r] : nullptr;
    }

    // Remaining fields (`elts_per_rank`, `elts_per_block`, `rank_offset`) are computed
    // inside the kernel launch helper; initialize to zero to be safe.
    p.elts_per_rank = 0;
    p.elts_per_block = 0;
    p.rank_offset = 0;

    // Fusion disabled.
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
 * @param d_sendbuff
 * @param d_recvbuff
 * @param n_elts
 * @param cfg
 * @return BenchmarkResults
 */
BenchmarkResults BenchmarkNccl(void* d_sendbuff, void* d_recvbuff, std::size_t n_elts, BenchmarkConfig const& cfg) {
    // warmup
    for (int i = 0; i < kWarmupIters; ++i) {
        NCCL_CHECK(ncclAllReduce(d_sendbuff, d_recvbuff, n_elts, (cfg.dtype == nvinfer1::DataType::kFLOAT ? ncclFloat32 : ncclHalf), ncclSum, cfg.nccl_comm, cfg.stream));
    }

    CudaTimer timer;
    timer.Start();
    for (int it = 0; it < cfg.iterations; ++it) {
        NCCL_CHECK(ncclAllReduce(d_sendbuff, d_recvbuff, n_elts, (cfg.dtype == nvinfer1::DataType::kFLOAT ? ncclFloat32 : ncclHalf), ncclSum, cfg.nccl_comm, cfg.stream));
    }
    float total_time = timer.Stop();

    BenchmarkResults r{};
    r.implementation = kNcclLabel;
    r.num_elements = n_elts;
    r.message_bytes = n_elts * cfg.element_size;
    r.latency_ms = total_time / static_cast<float>(cfg.iterations);
    r.bandwidth_gbps = CalculateBandwidth(r.message_bytes, r.latency_ms);
    return r;
}

// -------------------------------------------------------------------------------------------------
// Custom One‑shot
// -------------------------------------------------------------------------------------------------

/**
 * @brief Run a benchmark using the custom one-shot all-reduce implementation.
 *
 * @param d_sendbuff
 * @param d_recvbuff
 * @param n_elts
 * @param peer_comm_ptrs
 * @param peer_barrier_in_ptrs
 * @param peer_barrier_out_ptrs
 * @param cfg
 * @param barrier_counter
 * @return BenchmarkResults
 */
BenchmarkResults BenchmarkOneshot(void* d_sendbuff, void* d_recvbuff, std::size_t n_elts, std::vector<void*> const& peer_comm_ptrs,
                                  std::vector<uint32_t*> const& peer_barrier_in_ptrs, std::vector<uint32_t*> const& peer_barrier_out_ptrs, BenchmarkConfig const& cfg,
                                  uint32_t& barrier_counter) {
    // warmup
    for (int i = 0; i < kWarmupIters; ++i) {
        AllReduceParams p = MakeParams(d_sendbuff, d_recvbuff, n_elts, cfg, peer_comm_ptrs, peer_barrier_in_ptrs, peer_barrier_out_ptrs, barrier_counter++);
        customAllReduce(p, cfg.dtype, AllReduceStrategyType::ONESHOT, static_cast<AllReduceStrategyConfig>(0), AllReduceFusionOp::NONE, cfg.stream);
    }

    CudaTimer timer;
    timer.Start();
    for (int it = 0; it < cfg.iterations; ++it) {
        AllReduceParams p = MakeParams(d_sendbuff, d_recvbuff, n_elts, cfg, peer_comm_ptrs, peer_barrier_in_ptrs, peer_barrier_out_ptrs, barrier_counter++);
        customAllReduce(p, cfg.dtype, AllReduceStrategyType::ONESHOT, static_cast<AllReduceStrategyConfig>(0), AllReduceFusionOp::NONE, cfg.stream);
    }
    float total_time = timer.Stop();

    BenchmarkResults r{};
    r.implementation = kOneShotLabel;
    r.num_elements = n_elts;
    r.message_bytes = n_elts * cfg.element_size;
    r.latency_ms = total_time / static_cast<float>(cfg.iterations);
    r.bandwidth_gbps = CalculateBandwidth(r.message_bytes, r.latency_ms);
    return r;
}

// -------------------------------------------------------------------------------------------------
// Custom Two‑shot
// -------------------------------------------------------------------------------------------------

/**
 * @brief Run a benchmark using the custom two-shot all-reduce implementation.
 *
 * @param d_sendbuff
 * @param d_recvbuff
 * @param n_elts
 * @param peer_comm_ptrs
 * @param peer_barrier_in_ptrs
 * @param peer_barrier_out_ptrs
 * @param cfg
 * @param barrier_counter
 * @return BenchmarkResults
 */
BenchmarkResults BenchmarkTwoshot(void* d_sendbuff, void* d_recvbuff, std::size_t n_elts, std::vector<void*> const& peer_comm_ptrs,
                                  std::vector<uint32_t*> const& peer_barrier_in_ptrs, std::vector<uint32_t*> const& peer_barrier_out_ptrs, BenchmarkConfig const& cfg,
                                  uint32_t& barrier_counter) {
    // warmup
    for (int i = 0; i < kWarmupIters; ++i) {
        AllReduceParams p = MakeParams(d_sendbuff, d_recvbuff, n_elts, cfg, peer_comm_ptrs, peer_barrier_in_ptrs, peer_barrier_out_ptrs, barrier_counter++);
        customAllReduce(p, cfg.dtype, AllReduceStrategyType::TWOSHOT, static_cast<AllReduceStrategyConfig>(0), AllReduceFusionOp::NONE, cfg.stream);
    }

    CudaTimer timer;
    timer.Start();
    for (int it = 0; it < cfg.iterations; ++it) {
        AllReduceParams p = MakeParams(d_sendbuff, d_recvbuff, n_elts, cfg, peer_comm_ptrs, peer_barrier_in_ptrs, peer_barrier_out_ptrs, barrier_counter++);
        customAllReduce(p, cfg.dtype, AllReduceStrategyType::TWOSHOT, static_cast<AllReduceStrategyConfig>(0), AllReduceFusionOp::NONE, cfg.stream);
    }
    float total_time = timer.Stop();

    BenchmarkResults r{};
    r.implementation = kTwoShotLabel;
    r.num_elements = n_elts;
    r.message_bytes = n_elts * cfg.element_size;
    r.latency_ms = total_time / static_cast<float>(cfg.iterations);
    r.bandwidth_gbps = CalculateBandwidth(r.message_bytes, r.latency_ms);
    return r;
}

// -------------------------------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------------------------------

/**
 * @brief Main function for the benchmark.
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char** argv) {
    std::srand(42);

    MPI_CHECK(MPI_Init(&argc, &argv));

    std::string dtype_s = "fp32";
    int iterations = 10;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--dtype") && i + 1 < argc) {
            dtype_s = argv[++i];
        } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        }
    }

    BenchmarkConfig cfg{};
    cfg.dtype = (dtype_s == "fp16") ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
    cfg.element_size = getDTypeSize(cfg.dtype);
    cfg.iterations = iterations;

    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &cfg.world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &cfg.world_rank));
    cfg.local_rank = GetLocalRank();
    cfg.ranks_per_node = cfg.world_size;

    CUDA_CHECK(cudaSetDevice(cfg.local_rank));
    CUDA_CHECK(cudaStreamCreate(&cfg.stream));

    ncclUniqueId nccl_id;
    if (cfg.world_rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }
    MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCL_CHECK(ncclCommInitRank(&cfg.nccl_comm, cfg.world_size, nccl_id, cfg.world_rank));

    std::unique_ptr<CsvWriter> csv_writer;
    if (cfg.world_rank == 0) {
        std::filesystem::create_directories(kDirectoryPath);
        std::string filename = std::string(kDirectoryPath) + "/" + kFileName + "_" + GenerateTimestamp() + "_" + dtype_s + ".csv";
        csv_writer = std::make_unique<CsvWriter>(filename, true);
    }

    // Message size sweep (in multiples of 1024 elements just like original)
    std::vector<int> n_values;
    for (int n = 1; n < 16; ++n) n_values.push_back(n);
    for (int n = 16; n < 128; n += 16) n_values.push_back(n);
    for (int n = 128; n < 1024; n += 128) n_values.push_back(n);

    // Device buffers used for all message sizes (max allocate)
    std::size_t const max_n = static_cast<std::size_t>(n_values.back()) * kSweepSize;
    std::size_t const max_bytes = max_n * cfg.element_size;

    void* d_sendbuff = nullptr;
    void* d_recvbuff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sendbuff, max_bytes));
    CUDA_CHECK(cudaMalloc(&d_recvbuff, max_bytes));

    // Fill send buffer with values
    if (cfg.dtype == nvinfer1::DataType::kFLOAT) {
        std::vector<float> h(max_n);
        InitVector<float>(h.data(), max_n);
        CUDA_CHECK(cudaMemcpy(d_sendbuff, h.data(), max_bytes, cudaMemcpyHostToDevice));
    } else {
        std::vector<__half> h(max_n);
        InitVector<__half>(h.data(), max_n);
        CUDA_CHECK(cudaMemcpy(d_sendbuff, h.data(), max_bytes, cudaMemcpyHostToDevice));
    }

    // Allocate & share peer comm / barrier buffers (each per max size)
    void* d_comm_buf = nullptr;
    uint32_t* d_barrier_in = nullptr;
    uint32_t* d_barrier_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_comm_buf, max_bytes));
    CUDA_CHECK(cudaMalloc(&d_barrier_in, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_barrier_out, sizeof(uint32_t)));

    std::vector<void*> peer_comm_ptrs;
    std::vector<uint32_t*> peer_barrier_in_ptrs;
    std::vector<uint32_t*> peer_barrier_out_ptrs;
    ShareIpcPtrs(d_comm_buf, peer_comm_ptrs, cfg.local_rank, cfg.ranks_per_node);
    ShareIpcPtrs(d_barrier_in, peer_barrier_in_ptrs, cfg.local_rank, cfg.ranks_per_node);
    ShareIpcPtrs(d_barrier_out, peer_barrier_out_ptrs, cfg.local_rank, cfg.ranks_per_node);

    uint32_t barrier_counter = 1;

    for (int n_mult : n_values) {
        std::size_t const n_elts = static_cast<std::size_t>(n_mult) * kSweepSize;
        std::size_t const bytes = n_elts * cfg.element_size;

        if (cfg.world_rank == 0) {
            std::cout << "Running benchmark for " << n_elts << " elements (" << bytes << " Bytes)..." << std::endl;
        }

        // NCCL baseline
        BenchmarkResults nccl_res = BenchmarkNccl(d_sendbuff, d_recvbuff, n_elts, cfg);

        // Golden copy from NCCL result for validation
        BenchmarkResults val_res{};
        if (cfg.dtype == nvinfer1::DataType::kFLOAT) {
            std::vector<float> golden(n_elts);
            CUDA_CHECK(cudaMemcpy(golden.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            val_res = ValidateResults<float>(d_recvbuff, golden, cfg, n_elts);
        } else {
            std::vector<__half> golden(n_elts);
            CUDA_CHECK(cudaMemcpy(golden.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            val_res = ValidateResults<__half>(d_recvbuff, golden, cfg, n_elts);
        }
        // annotate nccl with validation
        nccl_res.accuracy_pct = val_res.accuracy_pct;
        nccl_res.max_abs_error = val_res.max_abs_error;
        nccl_res.mean_abs_error = val_res.mean_abs_error;
        nccl_res.max_rel_error = val_res.max_rel_error;
        nccl_res.rmse = val_res.rmse;

        if (csv_writer) {
            csv_writer->WriteRow(nccl_res.implementation, nccl_res.message_bytes, nccl_res.latency_ms, nccl_res.bandwidth_gbps, nccl_res.accuracy_pct, nccl_res.max_abs_error,
                                 nccl_res.mean_abs_error, nccl_res.max_rel_error, nccl_res.rmse);
        }

        // Reset and benchmark one‑shot
        CUDA_CHECK(cudaMemset(d_recvbuff, 0, bytes));
        BenchmarkResults oneshot_res = BenchmarkOneshot(d_sendbuff, d_recvbuff, n_elts, peer_comm_ptrs, peer_barrier_in_ptrs, peer_barrier_out_ptrs, cfg, barrier_counter);
        // Validate one‑shot
        if (cfg.dtype == nvinfer1::DataType::kFLOAT) {
            std::vector<float> golden(n_elts);
            CUDA_CHECK(cudaMemcpy(golden.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            val_res = ValidateResults<float>(d_recvbuff, golden, cfg, n_elts);
        } else {
            std::vector<__half> golden(n_elts);
            CUDA_CHECK(cudaMemcpy(golden.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            val_res = ValidateResults<__half>(d_recvbuff, golden, cfg, n_elts);
        }
        oneshot_res.accuracy_pct = val_res.accuracy_pct;
        oneshot_res.max_abs_error = val_res.max_abs_error;
        oneshot_res.mean_abs_error = val_res.mean_abs_error;
        oneshot_res.max_rel_error = val_res.max_rel_error;
        oneshot_res.rmse = val_res.rmse;

        if (csv_writer) {
            csv_writer->WriteRow(oneshot_res.implementation, oneshot_res.message_bytes, oneshot_res.latency_ms, oneshot_res.bandwidth_gbps, oneshot_res.accuracy_pct,
                                 oneshot_res.max_abs_error, oneshot_res.mean_abs_error, oneshot_res.max_rel_error, oneshot_res.rmse);
        }

        // Reset and benchmark two‑shot
        CUDA_CHECK(cudaMemset(d_recvbuff, 0, bytes));
        BenchmarkResults twoshot_res = BenchmarkTwoshot(d_sendbuff, d_recvbuff, n_elts, peer_comm_ptrs, peer_barrier_in_ptrs, peer_barrier_out_ptrs, cfg, barrier_counter);
        // Validate two‑shot
        if (cfg.dtype == nvinfer1::DataType::kFLOAT) {
            std::vector<float> golden(n_elts);
            CUDA_CHECK(cudaMemcpy(golden.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            val_res = ValidateResults<float>(d_recvbuff, golden, cfg, n_elts);
        } else {
            std::vector<__half> golden(n_elts);
            CUDA_CHECK(cudaMemcpy(golden.data(), d_recvbuff, bytes, cudaMemcpyDeviceToHost));
            val_res = ValidateResults<__half>(d_recvbuff, golden, cfg, n_elts);
        }
        twoshot_res.accuracy_pct = val_res.accuracy_pct;
        twoshot_res.max_abs_error = val_res.max_abs_error;
        twoshot_res.mean_abs_error = val_res.mean_abs_error;
        twoshot_res.max_rel_error = val_res.max_rel_error;
        twoshot_res.rmse = val_res.rmse;

        if (csv_writer) {
            csv_writer->WriteRow(twoshot_res.implementation, twoshot_res.message_bytes, twoshot_res.latency_ms, twoshot_res.bandwidth_gbps, twoshot_res.accuracy_pct,
                                 twoshot_res.max_abs_error, twoshot_res.mean_abs_error, twoshot_res.max_rel_error, twoshot_res.rmse);
        }
    }

    if (csv_writer) {
        csv_writer->Close();
    }

    NCCL_CHECK(ncclCommDestroy(cfg.nccl_comm));
    CUDA_CHECK(cudaStreamDestroy(cfg.stream));
    MPI_CHECK(MPI_Finalize());

    if (cfg.world_rank == 0) {
        std::cout << "Benchmark completed successfully!" << std::endl;
    }
    return 0;
}
