/**
 * @file benchmark_broadcast.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief A benchmark for the Broadcast operation using MCCL and NCCL.
 * @version 0.1
 * @date 2025-06-13
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cstddef>
#include <vector>

#include "mccl/mccl.cuh"
#include "mccl/mccl_utils.cuh"
#include "utils/checker.cuh"
#include "utils/data.cuh"
#include "utils/timer.cuh"

constexpr unsigned int kN = 4194304;
constexpr unsigned int kBytes = kN * sizeof(float);
constexpr unsigned int kRootGpu = 0;

/**
 * @brief Main function for the broadcast example using CUDA IPC and MPI
 *
 * @param argc argument count
 * @param argv argument vector
 * @return int
 */
int main(int argc, char **argv) {
    CollectiveComm mccl_comm;
    Init(argc, argv, &mccl_comm);

    float *d_buff_mccl = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buff_mccl, kBytes));

    float *d_buff_nccl = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buff_nccl, kBytes));

    if (mccl_comm.world_rank == kRootGpu) {
        std::vector<float> h_buff(kN);
        InitVector<float>(h_buff.data(), kN);
        CUDA_CHECK(cudaMemcpy(d_buff_mccl, h_buff.data(), kBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_buff_nccl, h_buff.data(), kBytes, cudaMemcpyHostToDevice));
    }

    // ----------   M C C L   ---------- //
    MCCL<float> mccl(mccl_comm.world_rank, mccl_comm.world_size);

    // warm-up (not timed)
    mccl.Broadcast(d_buff_mccl, kN, kRootGpu);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CudaTimer mccl_timer("MCCL Broadcast", kBytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_timer.Start();
    mccl.Broadcast(d_buff_mccl, kN, kRootGpu);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_timer.Stop();

    if (mccl_comm.world_rank == kRootGpu) {
        mccl_timer.Print();
    }

    std::vector<float> h_ref(kN);
    if (mccl_comm.world_rank == kRootGpu) {
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_buff_mccl, kBytes, cudaMemcpyDeviceToHost));
    }
    MPI_CHECK(MPI_Bcast(h_ref.data(), kN, MPI_FLOAT, kRootGpu, MPI_COMM_WORLD));
    VerifyCollective("MCCL", d_buff_mccl, h_ref, mccl_comm.world_rank, kN);

    // ----------   N C C L   ---------- //
    NCCL_CHECK(ncclBroadcast(d_buff_nccl, d_buff_nccl, kN, ncclFloat32, kRootGpu, mccl_comm.comm, mccl_comm.stream));
    CUDA_CHECK(cudaStreamSynchronize(mccl_comm.stream));

    CudaTimer nccl_timer("NCCL Broadcast", kBytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_timer.Start();
    NCCL_CHECK(ncclBroadcast(d_buff_nccl, d_buff_nccl, kN, ncclFloat32, kRootGpu, mccl_comm.comm, mccl_comm.stream));
    CUDA_CHECK(cudaStreamSynchronize(mccl_comm.stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_timer.Stop();

    if (mccl_comm.world_rank == kRootGpu) {
        nccl_timer.Print();
    }

    VerifyCollective("NCCL", d_buff_nccl, h_ref, mccl_comm.world_rank, kN);

    CUDA_CHECK(cudaFree(d_buff_mccl));
    CUDA_CHECK(cudaFree(d_buff_nccl));

    Finalize(&mccl_comm);

    return 0;
}