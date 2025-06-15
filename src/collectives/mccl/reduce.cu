/**
 * @file reduce.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Example of using MCCL for a Reduce operation across multiple GPUs
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
#include "data.cuh"
#include "mccl.cuh"

constexpr std::size_t kN = 4194304;
constexpr std::size_t kBytes = (kN * sizeof(float));
constexpr std::size_t kRootGpu = 0;

/**
 * @brief Main function to demonstrate Reduce operation using MCCL
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char **argv) {
    int num_gpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
    std::cout << "Number of GPUs available: " << num_gpus << std::endl;

    // Initialize MCCL library
    MCCL<float> mccl(num_gpus);

    // Host data for each GPU
    std::vector<float *> h_in(num_gpus);
    float *h_expected = new float[kN];

    // Initialize expected result to zero
    for (int i = 0; i < kN; i++) {
        h_expected[i] = 0.0f;
    }

    // Initialize input data and calculate expected result
    for (int i = 0; i < num_gpus; i++) {
        h_in[i] = new float[kN];
        InitVector<float>(h_in[i], kN);

        for (int j = 0; j < kN; j++) {
            h_expected[j] += h_in[i][j];
        }
    }

    // Allocate buffers for NCCL-style API
    std::vector<float *> sendbuff(num_gpus);
    std::vector<float *> recvbuff(num_gpus);

    // Allocate GPU memory
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&sendbuff[i], kBytes));
        CHECK_CUDA(cudaMalloc(&recvbuff[i], kBytes));
    }

    // Initialize data on each device
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMemcpy(sendbuff[i], h_in[i], kBytes, cudaMemcpyHostToDevice));
    }

    // Perform reduce operation using MCCL (SUM operation)
    const Operation op = SUM;
    mccl.Reduce(sendbuff, recvbuff, kN, kRootGpu, op);

    // Verify results - only check root GPU
    std::cout << "Reduce completed. Verifying result on root GPU " << kRootGpu << "...\n";

    CHECK_CUDA(cudaSetDevice(kRootGpu));
    float *verify_data = new float[kN];
    CHECK_CUDA(cudaMemcpy(verify_data, recvbuff[kRootGpu], kBytes, cudaMemcpyDeviceToHost));

    bool success = true;
    for (int j = 0; j < kN; j++) {
        if (std::fabs(verify_data[j] - h_expected[j]) > 1e-5f) {
            success = false;
            std::cout << "Mismatch at index " << j << ": expected " << h_expected[j] << ", got " << verify_data[j] << std::endl;
            break;
        }
    }
    std::cout << "Root GPU " << kRootGpu << ": " << (success ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(sendbuff[i]));
        CHECK_CUDA(cudaFree(recvbuff[i]));
        delete[] h_in[i];
    }
    delete[] h_expected;
    delete[] verify_data;

    return 0;
}
