/**
 * @file reducescatter.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief ReduceScatter example using CUDA IPC and MPI
 * @version 0.1
 * @date 2025-06-13
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "checker.cuh"
#include "data.cuh"

constexpr std::size_t kN = 4194304;
constexpr std::size_t kBlockDim = 256;

/**
 * @brief Kernel to add two vectors in place
 *
 * @param dst Pointer to the destination vector
 * @param src Pointer to the source vector
 * @param n Number of elements in the vectors
 * @return __global__
 */
__global__ void VecAddInPlace(float *dst, const float *src, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] += src[i];
    }
}

/**
 * @brief Main function for the ReduceScatter example using CUDA IPC and MPI
 *
 * @param argc argument count
 * @param argv argument vector
 * @return int
 */
int main(int argc, char **argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    CUDA_CHECK(cudaSetDevice(world_rank));

    for (std::size_t rank = 0; rank < world_size; ++rank) {
        if (rank != world_rank) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(rank, 0));
        }
    }

    const int recvcount = kN / world_size;
    const std::size_t sendbytes = kN * sizeof(float);
    const std::size_t recvbytes = recvcount * sizeof(float);
    const std::size_t gridSlice = (recvcount + kBlockDim - 1) / kBlockDim;

    float *d_sendbuff = nullptr, *d_recvbuff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sendbuff, sendbytes));
    CUDA_CHECK(cudaMalloc(&d_recvbuff, recvbytes));

    std::vector<float> h_in(kN);
    InitVector<float>(h_in.data(), kN);
    CUDA_CHECK(cudaMemcpy(d_sendbuff, h_in.data(), sendbytes, cudaMemcpyHostToDevice));

    cudaIpcMemHandle_t handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, d_sendbuff));

    std::vector<cudaIpcMemHandle_t> locals(world_size);
    MPI_CHECK(MPI_Allgather(&handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, locals.data(), sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD));

    std::vector<float *> remotes(world_size, nullptr);
    for (std::size_t rank = 0; rank < world_size; ++rank) {
        if (rank != world_rank) {
            CUDA_CHECK(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&remotes[rank]), locals[rank], cudaIpcMemLazyEnablePeerAccess));
        }
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const std::size_t offset_elems = static_cast<std::size_t>(world_rank) * recvcount;
    CUDA_CHECK(cudaMemcpyAsync(d_recvbuff,\ d_sendbuff + offset_elems, recvbytes, cudaMemcpyDeviceToDevice, stream));

    float *d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, recvbytes));

    for (std::size_t src = 0; src < world_size; ++src) {
        if (src != world_rank) {
            const float *remote_slice = (remotes[src] + offset_elems);
            CUDA_CHECK(cudaMemcpyPeerAsync(d_tmp, world_rank, remote_slice, src, recvbytes, stream));

            VecAddInPlace<<<gridSlice, kBlockDim, 0, stream>>>(d_recvbuff, d_tmp, recvcount);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_tmp));

    std::vector<float> h_result(recvcount);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_recvbuff, recvbytes, cudaMemcpyDeviceToHost));

    std::vector<int> recvcounts(world_size, recvcount);
    std::vector<float> h_expected(recvcount);
    MPI_CHECK(MPI_Reduce_scatter(h_in.data(), h_expected.data(), recvcounts.data(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));

    bool success = true;
    for (std::size_t i = 0; i < recvcount; ++i) {
        if (std::fabs(h_result[i] - h_expected[i]) > 1e-4f) {
            success = false;
            break;
        }
    }
    std::cout << "Device " << world_rank << " : " << (success ? "PASS" : "FAIL") << '\n';

    for (std::size_t rank = 0; rank < world_size; ++rank) {
        if (remotes[rank] != nullptr) {
            CUDA_CHECK(cudaIpcCloseMemHandle(remotes[rank]));
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_sendbuff));
    CUDA_CHECK(cudaFree(d_recvbuff));
    MPI_CHECK(MPI_Finalize());
    return 0;
}