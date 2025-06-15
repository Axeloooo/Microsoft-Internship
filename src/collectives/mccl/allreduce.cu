/**
 * @file allreduce.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Example of AllReduce operation using MCCL library
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

/**
 * @brief Main function to demonstrate AllReduce operation using MCCL library
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

    // Allocate buffers for MCCL-style API
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

    // Perform allreduce operation using MCCL (SUM operation)
    std::cout << "Performing AllReduce operation (SUM)..." << std::endl;
    const Operation op = SUM;
    mccl.AllReduce(sendbuff, recvbuff, kN, op);

    // Verify results
    std::cout << "AllReduce completed. Verifying results..." << std::endl;
    bool all_success = true;
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        float *verify_data = new float[kN];
        CHECK_CUDA(cudaMemcpy(verify_data, recvbuff[i], kBytes, cudaMemcpyDeviceToHost));

        bool success = true;
        for (int j = 0; j < kN; j++) {
            if (std::fabs(verify_data[j] - h_expected[j]) > 1e-5f) {
                success = false;
                std::cout << "Mismatch at GPU " << i << ", index " << j << ": expected " << h_expected[j] << ", got " << verify_data[j] << std::endl;
                break;
            }
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
    delete[] h_expected;

    return 0;
}
