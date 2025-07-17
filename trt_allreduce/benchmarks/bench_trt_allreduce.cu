/**
 * @file bench_trt_allreduce.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Benchmark for TensorRT AllReduce implementations using NCCL, custom kernels, and user buffers.
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "common/dataType.h"
#include "kernels/customAllReduceKernels.h"
#include "kernels/userbuffers/ub_interface.h"
#include "utils/checker.h"
#include "utils/csv_writer.h"
#include "utils/cuda_timer.cuh"
#include "utils/data.h"

namespace {

constexpr std::size_t kWarmupIters = 10;
constexpr std::size_t kBarrierElems = 1024;
constexpr std::size_t kSweepUnit = 10240ULL;
constexpr const char* kFileName = "benchmark_trt_allreduce";
constexpr const char* kDirectoryPath = "data";

struct Options {
    std::string dtype_str{"fp16"};
    int iters{10};
};

std::string GenerateTimestamp() {
    auto now = std::time(nullptr);
    auto* tm_now = std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(tm_now, "%Y%m%d-%H%M%S");
    return oss.str();
}

Options ParseCommandLine(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg{argv[i]};
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after " << arg << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            return argv[++i];
        };
        if (arg == "--dtype") {
            opt.dtype_str = next();
        } else if (arg == "--iters") {
            opt.iters = std::atoi(next());
        }
    }
    return opt;
}
struct Metrics {
    double pct_correct{100.0};
    float max_abs{0.F};
    double mean_abs{0.0};
    float max_rel{0.F};
    double rmse{0.0};
};

Metrics ComputeMetrics(void* d_buff, const std::vector<float>& h_golden, nvinfer1::DataType dtype, std::size_t n_elts) {
    std::vector<float> h_actual(n_elts);

    if (dtype == nvinfer1::DataType::kFLOAT) {
        CUDA_CHECK(cudaMemcpy(h_actual.data(), d_buff, n_elts * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        std::vector<__half> h_tmp(n_elts);
        CUDA_CHECK(cudaMemcpy(h_tmp.data(), d_buff, n_elts * sizeof(__half), cudaMemcpyDeviceToHost));
        std::transform(h_tmp.begin(), h_tmp.end(), h_actual.begin(), [](const __half& x) { return __half2float(x); });
    }

    const float eps = (dtype == nvinfer1::DataType::kFLOAT) ? 1e-5f : 1e-2f;

    std::size_t correct = 0;
    double abs_sum = 0.0;
    double sq_err_sum = 0.0;
    float max_abs_err = 0.0F;
    float max_rel_err = 0.0F;

    for (std::size_t i = 0; i < n_elts; ++i) {
        const float g = h_golden[i];
        const float a = h_actual[i];
        const float diff = std::abs(g - a);
        const float rel = (std::abs(g) > 0.0F) ? diff / std::abs(g) : diff;
        if (diff <= eps) {
            ++correct;
        }
        abs_sum += diff;
        sq_err_sum += static_cast<double>(diff) * diff;
        max_abs_err = std::max(max_abs_err, diff);
        max_rel_err = std::max(max_rel_err, rel);
    }

    std::size_t global_correct = 0;
    double global_abs_sum = 0.0;
    double global_sq_sum = 0.0;
    float global_max_abs = 0.0F;
    float global_max_rel = 0.0F;

    MPI_CHECK(MPI_Allreduce(&correct, &global_correct, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&abs_sum, &global_abs_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&sq_err_sum, &global_sq_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&max_abs_err, &global_max_abs, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&max_rel_err, &global_max_rel, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD));

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const std::size_t total_elems = n_elts * static_cast<std::size_t>(world_size);

    Metrics m{};
    m.pct_correct = 100.0 * static_cast<double>(global_correct) / static_cast<double>(total_elems);
    m.max_abs = global_max_abs;
    m.mean_abs = global_abs_sum / static_cast<double>(total_elems);
    m.max_rel = global_max_rel;
    m.rmse = std::sqrt(global_sq_sum / static_cast<double>(total_elems));
    return m;
}

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

class NcclEnv {
   public:
    NcclEnv(int world_size, int world_rank) {
        if (world_rank == 0) {
            NCCL_CHECK(ncclGetUniqueId(&id_));
        }
        MPI_Bcast(&id_, sizeof(id_), MPI_BYTE, 0, MPI_COMM_WORLD);
        NCCL_CHECK(ncclCommInitRank(&comm_, world_size, id_, world_rank));
    }

    NcclEnv(const NcclEnv&) = delete;
    NcclEnv& operator=(const NcclEnv&) = delete;

    ~NcclEnv() { ncclCommDestroy(comm_); }

    ncclComm_t comm() const { return comm_; }

   private:
    ncclComm_t comm_{};
    static ncclUniqueId id_;
};

ncclUniqueId NcclEnv::id_{};

void InitUserBuffers(std::size_t max_elts, nvinfer1::DataType dtype, int rank, int size) {
    if (!trt_allreduce::runtime::ub::ub_is_initialized()) {
        trt_allreduce::runtime::ub::ub_initialize(size);

        const std::size_t elt_sz = trt_allreduce::common::getDTypeSize(dtype);
        const std::size_t buf_bytes = max_elts * elt_sz;

        for (std::size_t id = 0; id < 3; ++id) {
            (void)trt_allreduce::runtime::ub::ub_allocate(buf_bytes);
        }
    }
}

using ImplFn = std::function<void(void*, void*, std::size_t, cudaStream_t)>;

ImplFn MakeNcclImpl(ncclComm_t comm, nvinfer1::DataType dtype) {
    const ncclDataType_t nccl_dtype = (dtype == nvinfer1::DataType::kHALF) ? ncclHalf : ncclFloat32;
    return
        [comm, nccl_dtype](void* send, void* recv, std::size_t elts, cudaStream_t stream) { NCCL_CHECK(ncclAllReduce(send, recv, elts, nccl_dtype, ncclSum, comm, stream)); };
}

ImplFn MakeCustomImpl(trt_allreduce::kernels::AllReduceStrategyType strat, trt_allreduce::kernels::AllReduceStrategyConfig cfg, nvinfer1::DataType dtype) {
    return [strat, cfg, dtype](void* send, void* recv, std::size_t elts, cudaStream_t stream) {
        /* ---------- scratch buffers on the local GPU ---------- */
        const std::size_t bytes = elts * trt_allreduce::common::getDTypeSize(dtype);

        void* d_comm_buf = nullptr;
        uint32_t* d_barrier_in = nullptr;
        uint32_t* d_barrier_out = nullptr;

        CUDA_CHECK(cudaMalloc(&d_comm_buf, bytes));
        CUDA_CHECK(cudaMalloc(&d_barrier_in, kBarrierElems * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_barrier_out, kBarrierElems * sizeof(uint32_t)));

        CUDA_CHECK(cudaMemsetAsync(d_barrier_in, 0, kBarrierElems * sizeof(uint32_t), stream));
        CUDA_CHECK(cudaMemsetAsync(d_barrier_out, 0, kBarrierElems * sizeof(uint32_t), stream));

        /* ---------- IPC mapping to peer buffers ---------- */
        int ranks_per_node = 0;
        int local_rank = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &ranks_per_node);
        MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);

        std::vector<void*> peer_comm_ptrs;
        std::vector<uint32_t*> peer_barrier_in_ptrs;
        std::vector<uint32_t*> peer_barrier_out_ptrs;

        ShareIpcPtrs(d_comm_buf, peer_comm_ptrs, local_rank, ranks_per_node);
        ShareIpcPtrs(d_barrier_in, peer_barrier_in_ptrs, local_rank, ranks_per_node);
        ShareIpcPtrs(d_barrier_out, peer_barrier_out_ptrs, local_rank, ranks_per_node);

        /* ---------- build param struct ---------- */
        trt_allreduce::kernels::AllReduceParams p{};  // zero-initialize
        p.local_output_buffer_ptr = recv;
        p.local_input_buffer_ptr = send;
        p.elts_total = elts;

        /* per-rank / per-block information */
        p.elts_per_rank = elts / static_cast<std::size_t>(ranks_per_node);
        p.elts_per_block = strat == trt_allreduce::kernels::AllReduceStrategyType::ONESHOT ? p.elts_per_rank : std::ceil(p.elts_per_rank / 2);
        p.rank_offset = static_cast<std::size_t>(local_rank) * p.elts_per_rank;
        p.ranks_per_node = ranks_per_node;
        p.local_rank = local_rank;

        /* simple ping-pong barrier flag */
        p.barrier_flag = 0;

        /* pointer tables */
        for (int r = 0; r < ranks_per_node; ++r) {
            p.peer_comm_buffer_ptrs[r] = peer_comm_ptrs[r];
            p.peer_barrier_ptrs_in[r] = peer_barrier_in_ptrs[r];
            p.peer_barrier_ptrs_out[r] = peer_barrier_out_ptrs[r];
        }

        /* no fusion for this benchmark */
        p.fusion_params = {};

        /* ---------- launch kernel ---------- */
        trt_allreduce::kernels::customAllReduce(p, dtype, strat, cfg, trt_allreduce::kernels::AllReduceFusionOp::NONE, stream);

        /* ---------- cleanup ---------- */
        // Un-map peer buffers created with cudaIpcOpenMemHandle to prevent
        // “resource already mapped” errors on the next iteration.
        for (int r = 0; r < ranks_per_node; ++r) {
            if (r != local_rank) {
                CUDA_CHECK(cudaIpcCloseMemHandle(static_cast<void*>(peer_comm_ptrs[r])));
                CUDA_CHECK(cudaIpcCloseMemHandle(static_cast<void*>(peer_barrier_in_ptrs[r])));
                CUDA_CHECK(cudaIpcCloseMemHandle(static_cast<void*>(peer_barrier_out_ptrs[r])));
            }
        }

        CUDA_CHECK(cudaFree(d_comm_buf));
        CUDA_CHECK(cudaFree(d_barrier_in));
        CUDA_CHECK(cudaFree(d_barrier_out));
    };
}

ImplFn MakeUbImpl(int rank, int size, nvinfer1::DataType dtype) {
    return [rank, size, dtype](void*, void* recv, std::size_t elts, cudaStream_t stream) {
        auto buf0 = trt_allreduce::runtime::ub::ub_get(0);
        auto comm = trt_allreduce::runtime::ub::ub_comm();

        if (buf0.addr != recv) {
            CUDA_CHECK(cudaMemcpyAsync(buf0.addr, recv, elts * trt_allreduce::common::getDTypeSize(dtype), cudaMemcpyDeviceToDevice, stream));
        }
        trt_allreduce::kernels::ub::allreduce2_userbuff_inplace_launcher(buf0.handle, 0, elts, dtype, comm, stream);
    };
}

void RunBenchmark(const std::string& name, const ImplFn& fn, std::size_t bytes, int iters, int warmup, cudaStream_t stream, trt_allreduce::utils::CsvWriter* csv,
                  nvinfer1::DataType dtype, int world_size, int world_rank) {
    const std::size_t elt_sz = trt_allreduce::common::getDTypeSize(dtype);
    const std::size_t elts = bytes / elt_sz;

    void* d_send = nullptr;
    void* d_recv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send, bytes));
    CUDA_CHECK(cudaMalloc(&d_recv, bytes));

    std::vector<float> h_local(elts);
    if (dtype == nvinfer1::DataType::kFLOAT) {
        std::vector<float> h_in(elts);
        trt_allreduce::utils::InitVector<float>(h_in.data(), elts);
        CUDA_CHECK(cudaMemcpy(d_send, h_in.data(), bytes, cudaMemcpyHostToDevice));
        h_local.assign(h_in.begin(), h_in.end());
    } else {
        std::vector<__half> h_in(elts);
        trt_allreduce::utils::InitVector<__half>(h_in.data(), elts);
        CUDA_CHECK(cudaMemcpy(d_send, h_in.data(), bytes, cudaMemcpyHostToDevice));
    }

    std::vector<float> h_golden(elts);
    MPI_CHECK(MPI_Allreduce(h_local.data(), h_golden.data(), static_cast<int>(elts), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));

    trt_allreduce::utils::CudaTimer timer;

    for (std::size_t i = 0; i < warmup; ++i) {
        fn(d_send, d_recv, elts, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemsetAsync(d_recv, 0, bytes, stream));

    timer.Start();
    for (std::size_t i = 0; i < iters; ++i) {
        fn(d_send, d_recv, elts, stream);
    }
    float total_ms = timer.Stop();

    Metrics metrics = ComputeMetrics(d_recv, h_golden, dtype, elts);

    const double latency_ms = static_cast<double>(total_ms) / iters;
    const double bandwidth_gbps = static_cast<double>(bytes) / (latency_ms * 1e-3) / 1e9;

    if (world_rank == 0 && csv != nullptr) {
        csv->WriteRow(name, bytes, latency_ms, bandwidth_gbps, metrics.pct_correct, metrics.max_abs, metrics.mean_abs, metrics.max_rel, metrics.rmse);
    }

    CUDA_CHECK(cudaFree(d_send));
    CUDA_CHECK(cudaFree(d_recv));
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size = 0;
    int world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    Options opt = ParseCommandLine(argc, argv);

    const nvinfer1::DataType dtype = (opt.dtype_str == "fp16") ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
    const std::size_t elt_sz = trt_allreduce::common::getDTypeSize(dtype);

    CUDA_CHECK(cudaSetDevice(world_rank));

    for (int dev = 0; dev < world_size; ++dev) {
        if (dev != world_rank) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(dev, 0));
        }
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    NcclEnv nccl(world_size, world_rank);

    const std::size_t max_bytes = kSweepUnit * 1023ULL;
    const std::size_t max_elts = max_bytes / elt_sz;
    InitUserBuffers(max_elts, dtype, world_rank, world_size);

    std::unique_ptr<trt_allreduce::utils::CsvWriter> csv;
    if (world_rank == 0) {
        std::filesystem::create_directories(kDirectoryPath);
        std::string filename = std::string(kDirectoryPath) + "/" + kFileName + "_" + GenerateTimestamp() + "_" + opt.dtype_str + ".csv";
        csv = std::make_unique<trt_allreduce::utils::CsvWriter>(filename, true);
    }

    std::vector<std::pair<std::string, ImplFn>> impls = {
        {"NCCL", MakeNcclImpl(nccl.comm(), dtype)},
        {"One-Shot", MakeCustomImpl(trt_allreduce::kernels::AllReduceStrategyType::ONESHOT, trt_allreduce::kernels::AllReduceStrategyConfig::USE_MEMCPY, dtype)},
        {"Two-Shot", MakeCustomImpl(trt_allreduce::kernels::AllReduceStrategyType::TWOSHOT, trt_allreduce::kernels::AllReduceStrategyConfig::USE_MEMCPY, dtype)}};

    if (dtype == nvinfer1::DataType::kHALF) {
        impls.emplace_back("UserBuffer", MakeUbImpl(world_rank, world_size, dtype));
    }

    std::vector<int> n_values;
    for (int n = 1; n < 16; ++n) n_values.push_back(n);
    for (int n = 16; n < 128; n += 16) n_values.push_back(n);
    for (int n = 128; n < 1024; n += 128) n_values.push_back(n);

    for (int n_mult : n_values) {
        const std::size_t bytes = static_cast<std::size_t>(n_mult) * kSweepUnit;
        if (world_rank == 0) {
            std::cout << "Message Size " << bytes << " Bytes" << std::endl;
        }
        for (const auto& impl : impls) {
            RunBenchmark(impl.first, impl.second, bytes, opt.iters, kWarmupIters, stream, csv.get(), dtype, world_size, world_rank);
        }
        if (csv) {
            csv->Flush();
        }
    }

    if (csv) {
        csv->Close();
    }
    if (world_rank == 0) {
        std::cout << "Benchmark completed!" << std::endl;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Finalize();
    return 0;
}
