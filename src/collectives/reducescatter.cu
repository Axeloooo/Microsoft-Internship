/**
 * @file reducescatter.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Example of ReduceScatter operation using MCCL library
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
 * @brief Main function to demonstrate ReduceScatter operation using MCCL library
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

    // Calculate chunk size per GPU
    const int chunk_size = kN / num_gpus;
    const int chunk_bytes = chunk_size * sizeof(float);
    const int total_bytes = kBytes * num_gpus;

    // Host data for each GPU
    std::vector<float *> h_in(num_gpus);
    std::vector<float *> h_expected(num_gpus);

    // Initialize input data and expected results
    for (int i = 0; i < num_gpus; i++) {
        h_in[i] = new float[kN * num_gpus];
        h_expected[i] = new float[chunk_size];

        // Initialize input data
        InitVector<float>(h_in[i], kN * num_gpus);
    }

    for (int i = 0; i < num_gpus; i++) {
        // Calculate expected results for ReduceScatter (SUM operation)
        for (int j = 0; j < chunk_size; j++) {
            h_expected[i][j] = 0.0f;
            for (int k = 0; k < num_gpus; k++) {
                h_expected[i][j] += h_in[k][i * chunk_size + j];
            }
        }
    }

    // Allocate buffers for MCCL-style API
    std::vector<float *> sendbuff(num_gpus);
    std::vector<float *> recvbuff(num_gpus);

    // Allocate GPU memory
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&sendbuff[i], total_bytes));
        CHECK_CUDA(cudaMalloc(&recvbuff[i], chunk_bytes));
    }

    // Initialize data on each device
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMemcpy(sendbuff[i], h_in[i], total_bytes, cudaMemcpyHostToDevice));
    }

    // Perform reduce-scatter operation using MCCL (SUM operation)
    std::cout << "Performing ReduceScatter operation (SUM)..." << std::endl;
    const Operation op = SUM;
    mccl.ReduceScatter(sendbuff, recvbuff, chunk_size, op);

    // Verify results
    std::cout << "ReduceScatter completed. Verifying results..." << std::endl;
    bool all_success = true;
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CHECK_CUDA(cudaSetDevice(gpu));
        float *verify_data = new float[chunk_size];
        CHECK_CUDA(cudaMemcpy(verify_data, recvbuff[gpu], chunk_bytes, cudaMemcpyDeviceToHost));

        bool success = true;
        for (int j = 0; j < chunk_size; j++) {
            if (std::fabs(verify_data[j] - h_expected[gpu][j]) > 1e-5f) {
                success = false;
                std::cout << "Mismatch at GPU " << gpu << ", index " << j << ": expected " << h_expected[gpu][j] << ", got " << verify_data[j] << std::endl;
                break;
            }
        }
        std::cout << "GPU " << gpu << ": " << (success ? "PASS" : "FAIL") << std::endl;
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
        delete[] h_expected[i];
    }

    return 0;
}