/**
 * @file benchmark_allreduce.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief A benchmark for the AllGather operation using MCCL and NCCL.
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

/**
 * @brief Main function for the AllReduce example using CUDA IPC and MPI
 *
 * @param argc argument count
 * @param argv argument vector
 * @return int
 */
int main(int argc, char* argv[]) {
    // Set random seed for reproducibility
    srand(42);

    CollectiveComm mccl_comm;
    Init(argc, argv, &mccl_comm);

    float *d_send = nullptr, *d_recv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send, kBytes));
    CUDA_CHECK(cudaMalloc(&d_recv, kBytes));

    std::vector<float> h_init(kN);
    InitVector<float>(h_init.data(), kN);
    CUDA_CHECK(cudaMemcpy(d_send, h_init.data(), kBytes, cudaMemcpyHostToDevice));

    float* d_send_nccl = nullptr;
    float* d_recv_nccl = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send_nccl, kBytes));
    CUDA_CHECK(cudaMalloc(&d_recv_nccl, kBytes));
    CUDA_CHECK(cudaMemcpy(d_send_nccl, h_init.data(), kBytes, cudaMemcpyHostToDevice));

    // ------------------ MCCL -----------------------------------------------
    MCCL<float> mccl(mccl_comm.world_rank, mccl_comm.world_size);

    mccl.AllReduce(d_send, d_recv, kN, Operation::SUM);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CudaTimer mccl_timer("MCCL AllReduce", kBytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_timer.Start();
    mccl.AllReduce(d_send, d_recv, kN, Operation::SUM);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_timer.Stop();
    if (mccl_comm.world_rank == 0) {
        mccl_timer.Print();
    }

    std::vector<float> h_ref(kN);
    MPI_CHECK(MPI_Allreduce(h_init.data(), h_ref.data(), kN, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
    VerifyCollective("MCCL", d_recv, h_ref, mccl_comm.world_rank, kN);

    // ------------------ NCCL -----------------------------------------------
    NCCL_CHECK(ncclAllReduce(d_send_nccl, d_recv_nccl, kN, ncclFloat32, ncclSum, mccl_comm.comm, mccl_comm.stream));
    CUDA_CHECK(cudaStreamSynchronize(mccl_comm.stream));
    VerifyCollective("NCCL", d_recv_nccl, h_ref, mccl_comm.world_rank, kN);

    CudaTimer nccl_timer("NCCL AllReduce", kBytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_timer.Start();
    NCCL_CHECK(ncclAllReduce(d_send_nccl, d_recv_nccl, kN, ncclFloat32, ncclSum, mccl_comm.comm, mccl_comm.stream));
    CUDA_CHECK(cudaStreamSynchronize(mccl_comm.stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_timer.Stop();
    if (mccl_comm.world_rank == 0) {
        nccl_timer.Print();
    }

    CUDA_CHECK(cudaFree(d_send));
    CUDA_CHECK(cudaFree(d_recv));
    CUDA_CHECK(cudaFree(d_send_nccl));
    CUDA_CHECK(cudaFree(d_recv_nccl));

    Finalize(&mccl_comm);

    return 0;
}
