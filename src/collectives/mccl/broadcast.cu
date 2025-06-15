/**
 * @file broadcast.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Example of using MCCL for broadcast operation across multiple GPUs
 * @version 0.1
 * @date 2025-06-13
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "checker.cuh"
#include "data.cuh"
#include "mccl.cuh"

constexpr std::size_t kN = 4194304;
constexpr std::size_t kBytes = (kN * sizeof(float));
constexpr std::size_t kRootGpu = 0;

/**
 * @brief Main function to demonstrate broadcast operation using MCCL
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char **argv) {
    float *h_a = new float[kN];
    InitVector<float>(h_a, kN);

    int num_gpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
    std::cout << "Number of GPUs available: " << num_gpus << std::endl;

    // Initialize MCCL library
    MCCL<float> mccl(num_gpus);

    // Allocate buffers for NCCL-style API
    std::vector<float *> sendbuff(num_gpus);
    std::vector<float *> recvbuff(num_gpus);

    // Allocate GPU memory
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&sendbuff[i], kBytes));
        CHECK_CUDA(cudaMalloc(&recvbuff[i], kBytes));
    }

    // Initialize data on root device (only root's sendbuff is used in
    // broadcast)
    CHECK_CUDA(cudaSetDevice(kRootGpu));
    CHECK_CUDA(cudaMemcpy(sendbuff[kRootGpu], h_a, kBytes, cudaMemcpyHostToDevice));

    // Perform broadcast operation using MCCL
    mccl.Broadcast(sendbuff, recvbuff, kN, kRootGpu);

    // Verify results
    std::cout << "Broadcast completed. Verifying results...\n";
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        float *verify_data = new float[kN];
        CHECK_CUDA(cudaMemcpy(verify_data, recvbuff[i], kBytes, cudaMemcpyDeviceToHost));

        bool success = true;
        for (int j = 0; j < kN; j++) {
            if (std::fabs(verify_data[j] - h_a[j]) > 1e-5f) {
                success = false;
                break;
            }
        }
        std::cout << "Device " << i << ": " << (success ? "PASS" : "FAIL") << std::endl;
        delete[] verify_data;
    }

    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(sendbuff[i]));
        CHECK_CUDA(cudaFree(recvbuff[i]));
    }
    delete[] h_a;

    return 0;
}