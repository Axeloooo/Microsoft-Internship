#include <cuda_runtime.h>

#include <iostream>

#include "checker.cuh"
#include "data.cuh"

#define N 4194304
#define BYTES (N * sizeof(float))

int main(int argc, char **argv) {
    int NUM_GPUS = 0;
    CHECK_CUDA(cudaGetDeviceCount(&NUM_GPUS));
    std::cout << "Number of GPUs available: " << NUM_GPUS << std::endl;

    int TOTAL_BYTES = BYTES * NUM_GPUS;

    // Host data for each GPU
    float **h_a = new float *[NUM_GPUS];
    for (uint i = 0; i < NUM_GPUS; i++) {
        h_a[i] = new float[N];
        InitVector<float>(h_a[i], N);
    }

    // Device arrays
    float **d_a = new float *[NUM_GPUS];
    float **d_b = new float *[NUM_GPUS];
    int *devices = new int[NUM_GPUS];
    cudaStream_t *streams = new cudaStream_t[NUM_GPUS];

    // Initialize devices array with sequential device IDs
    for (uint i = 0; i < NUM_GPUS; i++) {
        devices[i] = i;
    }

    // Allocate memory on all devices
    for (uint i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaMalloc(&d_a[i], BYTES));
        CHECK_CUDA(cudaMalloc(&d_b[i], TOTAL_BYTES));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Enable peer-to-peer access
    for (uint i = 0; i < NUM_GPUS; i++) {
        for (uint j = 0; j < NUM_GPUS; j++) {
            if (i != j) {
                CHECK_CUDA(cudaSetDevice(devices[i]));
                int can_access;
                CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, devices[i],
                                                   devices[j]));
                if (can_access) {
                    CHECK_CUDA(cudaDeviceEnablePeerAccess(devices[j], 0));
                }
            }
        }
    }

    // Initialize input data on each device
    for (uint i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaMemcpy(d_a[i], h_a[i], BYTES, cudaMemcpyHostToDevice));
    }

    // Allgather: Each GPU copies its data to all other GPUs' output buffers
    for (uint src = 0; src < NUM_GPUS; src++) {
        for (uint dst = 0; dst < NUM_GPUS; dst++) {
            CHECK_CUDA(cudaSetDevice(devices[dst]));
            size_t offset = src * N;

            if (src == dst) {
                // Local copy
                CHECK_CUDA(cudaMemcpyAsync(d_b[dst] + offset, d_a[src], BYTES,
                                           cudaMemcpyDeviceToDevice,
                                           streams[dst]));
            } else {
                // Peer-to-peer copy
                CHECK_CUDA(cudaMemcpyPeerAsync(d_b[dst] + offset, devices[dst],
                                               d_a[src], devices[src], BYTES,
                                               streams[dst]));
            }
        }
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    // Verify results
    std::cout << "Allgather completed. Verifying results...\n";
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        float *verify_data = new float[N * NUM_GPUS];
        CHECK_CUDA(cudaMemcpy(verify_data, d_b[i], TOTAL_BYTES,
                              cudaMemcpyDeviceToHost));

        bool success = true;
        for (int src = 0; src < NUM_GPUS; src++) {
            for (int j = 0; j < N; j++) {
                int idx = src * N + j;
                float expected = h_a[src][j];
                if (verify_data[idx] != expected) {
                    success = false;
                    std::cout << "Mismatch at GPU " << i << ", index " << idx
                              << ": expected " << expected << ", got "
                              << verify_data[idx] << std::endl;
                    break;
                }
            }
            if (!success) break;
        }
        std::cout << "Device " << devices[i] << ": "
                  << (success ? "PASS" : "FAIL") << std::endl;
        delete[] verify_data;
    }

    // Cleanup
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaFree(d_a[i]));
        CHECK_CUDA(cudaFree(d_b[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        delete[] h_a[i];
    }
    delete[] devices;
    delete[] d_a;
    delete[] d_b;
    delete[] streams;
    delete[] h_a;

    return 0;
}
