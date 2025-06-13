#include <cuda_runtime.h>

#include <iostream>

#include "checker.cuh"
#include "collectives.cuh"
#include "data.cuh"

#define N 4194304
#define BYTES (N * sizeof(float))
#define ROOT_GPU 0

int main(int argc, char **argv) {
    int NUM_GPUS = 0;
    CHECK_CUDA(cudaGetDeviceCount(&NUM_GPUS));
    std::cout << "Number of GPUs available: " << NUM_GPUS << std::endl;

    // Initialize MCCL library
    MCCL<float> mccl(NUM_GPUS);

    // Host data for each GPU
    std::vector<float *> h_in(NUM_GPUS);
    float *h_expected = new float[N];

    // Initialize expected result to zero
    for (int i = 0; i < N; i++) {
        h_expected[i] = 0.0f;
    }

    // Initialize input data and calculate expected result
    for (int i = 0; i < NUM_GPUS; i++) {
        h_in[i] = new float[N];
        InitVector<float>(h_in[i], N);

        for (int j = 0; j < N; j++) {
            h_expected[j] += h_in[i][j];
        }
    }

    // Allocate buffers for NCCL-style API
    std::vector<float *> sendbuff(NUM_GPUS);
    std::vector<float *> recvbuff(NUM_GPUS);

    // Allocate GPU memory
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&sendbuff[i], BYTES));
        CHECK_CUDA(cudaMalloc(&recvbuff[i], BYTES));
    }

    // Initialize data on each device
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(
            cudaMemcpy(sendbuff[i], h_in[i], BYTES, cudaMemcpyHostToDevice));
    }
    // Initialize data on each device
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(
            cudaMemcpy(sendbuff[i], h_in[i], BYTES, cudaMemcpyHostToDevice));
    }

    // Perform reduce operation using MCCL (SUM operation)
    Operation op = SUM;
    mccl.reduce(sendbuff, recvbuff, N, ROOT_GPU, op);

    // Verify results - only check root GPU
    std::cout << "Reduce completed. Verifying result on root GPU " << ROOT_GPU
              << "...\n";

    CHECK_CUDA(cudaSetDevice(ROOT_GPU));
    float *verify_data = new float[N];
    CHECK_CUDA(cudaMemcpy(verify_data, recvbuff[ROOT_GPU], BYTES,
                          cudaMemcpyDeviceToHost));

    bool success = true;
    for (int j = 0; j < N; j++) {
        if (abs(verify_data[j] - h_expected[j]) > 1e-5) {
            success = false;
            std::cout << "Mismatch at index " << j << ": expected "
                      << h_expected[j] << ", got " << verify_data[j]
                      << std::endl;
            break;
        }
    }
    std::cout << "Root GPU " << ROOT_GPU << ": " << (success ? "PASS" : "FAIL")
              << std::endl;

    // Cleanup
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(sendbuff[i]));
        CHECK_CUDA(cudaFree(recvbuff[i]));
        delete[] h_in[i];
    }
    delete[] h_expected;
    delete[] verify_data;

    return 0;
}
