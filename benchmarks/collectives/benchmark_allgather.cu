/**
 * @file benchmark_allgather.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief  A benchmark for the AllGather operation using MCCL and NCCL.
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
 * @brief Main function for the AllGather example using CUDA IPC and MPI.
 *
 * @param argc argument count
 * @param argv argument vector
 * @return int
 */
int main(int argc, char *argv[]) {
    // Set random seed for reproducibility
    srand(42);

    CollectiveComm mccl_comm;
    Init(argc, argv, &mccl_comm);

    const unsigned int slice_elems = kN / mccl_comm.world_size;
    const unsigned int slice_bytes = slice_elems * sizeof(float);

    float *d_send = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send, slice_bytes));

    std::vector<float> h_init(slice_elems);
    InitVector<float>(h_init.data(), slice_elems);
    CUDA_CHECK(cudaMemcpy(d_send, h_init.data(), slice_bytes, cudaMemcpyHostToDevice));

    float *d_recv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_recv, kBytes));

    float *d_send_nccl = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send_nccl, slice_bytes));
    CUDA_CHECK(cudaMemcpy(d_send_nccl, h_init.data(), slice_bytes, cudaMemcpyHostToDevice));
    float *d_recv_nccl = nullptr;
    CUDA_CHECK(cudaMalloc(&d_recv_nccl, kBytes));

    // ------------------ MCCL -----------------------------------------------
    MCCL<float> mccl(mccl_comm.world_rank, mccl_comm.world_size);

    mccl.AllGather(d_send, d_recv, slice_elems);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CudaTimer mccl_timer("MCCL AllGather", kBytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_timer.Start();
    mccl.AllGather(d_send, d_recv, slice_elems);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    mccl_timer.Stop();

    if (mccl_comm.world_rank == 0) {
        mccl_timer.Print();
    }

    std::vector<float> h_ref(kN);
    MPI_CHECK(MPI_Allgather(h_init.data(), slice_elems, MPI_FLOAT, h_ref.data(), slice_elems, MPI_FLOAT, MPI_COMM_WORLD));
    VerifyCollective("MCCL", d_recv, h_ref, mccl_comm.world_rank, kN);

    // ------------------ NCCL -----------------------------------------------
    NCCL_CHECK(ncclAllGather(d_send_nccl, d_recv_nccl, slice_elems, ncclFloat32, mccl_comm.comm, mccl_comm.stream));
    CUDA_CHECK(cudaStreamSynchronize(mccl_comm.stream));
    VerifyCollective("NCCL", d_recv_nccl, h_ref, mccl_comm.world_rank, kN);

    CudaTimer nccl_timer("NCCL AllGather", kBytes);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    nccl_timer.Start();
    NCCL_CHECK(ncclAllGather(d_send_nccl, d_recv_nccl, slice_elems, ncclFloat32, mccl_comm.comm, mccl_comm.stream));
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
