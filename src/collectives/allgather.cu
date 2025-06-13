/**
 * @file allgather.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Example of AllGather operation using MCCL
 * @version 0.1
 * @date 2025-06-13
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "checker.cuh"
#include "collectives.cuh"
#include "data.cuh"

constexpr std::size_t kN = 4194304;
constexpr std::size_t kBytes = (kN * sizeof(float));

/**
 * @brief Main function to demonstrate AllGather operation using MCCL
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char **argv) {
    int num_gpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
    std::cout << "Number of GPUs available: " << num_gpus << std::endl;

    if (num_gpus == 0) {
        std::cerr << "No GPUs available!" << std::endl;
        return 1;
    }

    // Initialize MCCL library
    MCCL<float> mccl(num_gpus);

    // Host data for each GPU
    std::vector<float *> h_in(num_gpus);
    for (int i = 0; i < num_gpus; i++) {
        h_in[i] = new float[kN];
        InitVector<float>(h_in[i], kN);
    }

    // Allocate buffers for MCCL-style API
    std::vector<float *> sendbuff(num_gpus);
    std::vector<float *> recvbuff(num_gpus);

    // Allocate GPU memory
    const int total_bytes = kBytes * num_gpus;
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&sendbuff[i], kBytes));       // Input: N elements per GPU
        CHECK_CUDA(cudaMalloc(&recvbuff[i], total_bytes));  // Output: N*NUM_GPUS elements per GPU
    }

    // Initialize data on each device
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMemcpy(sendbuff[i], h_in[i], kBytes, cudaMemcpyHostToDevice));
    }

    // Perform allgather operation using MCCL
    std::cout << "Performing AllGather operation..." << std::endl;
    mccl.AllGather(sendbuff, recvbuff, kN);

    // Verify results
    std::cout << "AllGather completed. Verifying results..." << std::endl;
    bool all_success = true;
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        float *verify_data = new float[kN * num_gpus];
        CHECK_CUDA(cudaMemcpy(verify_data, recvbuff[i], total_bytes, cudaMemcpyDeviceToHost));

        bool success = true;
        for (int src = 0; src < num_gpus; src++) {
            for (int j = 0; j < kN; j++) {
                const int idx = src * kN + j;
                const float expected = h_in[src][j];
                if (std::fabs(verify_data[idx] - expected) > 1e-5f) {
                    success = false;
                    std::cout << "Mismatch at GPU " << i << ", index " << idx << ": expected " << expected << ", got " << verify_data[idx] << std::endl;
                    break;
                }
            }
            if (!success) break;
        }
        std::cout << "GPU " << i << ": " << (success ? "PASS" : "FAIL") << std::endl;
        if (!success) all_success = false;
        delete[] verify_data;
    }

    std::cout << "Overall result: " << (all_success ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(sendbuff[i]));
        CHECK_CUDA(cudaFree(recvbuff[i]));
        delete[] h_in[i];
    }

    return 0;
}
