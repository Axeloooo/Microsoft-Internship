#include <cuda_runtime.h>

#include <iostream>

#include "checker.cuh"
#include "collectives.cuh"
#include "data.cuh"

#define N 4194304
#define BYTES (N * sizeof(float))
#define ROOT 0

int main(int argc, char **argv) {
    float *h_a = new float[N];
    InitVector<float>(h_a, N);

    int NUM_GPUS = 0;
    CHECK_CUDA(cudaGetDeviceCount(&NUM_GPUS));
    std::cout << "Number of GPUs available: " << NUM_GPUS << std::endl;

    // Initialize MCCL library
    MCCL<float> mccl(NUM_GPUS);

    // Allocate buffers for NCCL-style API
    std::vector<float *> sendbuff(NUM_GPUS);
    std::vector<float *> recvbuff(NUM_GPUS);

    // Allocate GPU memory
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&sendbuff[i], BYTES));
        CHECK_CUDA(cudaMalloc(&recvbuff[i], BYTES));
    }

    // Initialize data on root device (only root's sendbuff is used in
    // broadcast)
    CHECK_CUDA(cudaSetDevice(ROOT));
    CHECK_CUDA(cudaMemcpy(sendbuff[ROOT], h_a, BYTES, cudaMemcpyHostToDevice));

    // Perform broadcast operation using MCCL
    mccl.broadcast(sendbuff, recvbuff, N, ROOT);

    // Verify results
    std::cout << "Broadcast completed. Verifying results...\n";
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        float *verify_data = new float[N];
        CHECK_CUDA(cudaMemcpy(verify_data, recvbuff[i], BYTES,
                              cudaMemcpyDeviceToHost));

        bool success = true;
        for (int j = 0; j < N; j++) {
            if (verify_data[j] != h_a[j]) {
                success = false;
                break;
            }
        }
        std::cout << "Device " << i << ": " << (success ? "PASS" : "FAIL")
                  << std::endl;
        delete[] verify_data;
    }

    // Cleanup
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(sendbuff[i]));
        CHECK_CUDA(cudaFree(recvbuff[i]));
    }
    delete[] h_a;

    return 0;
}