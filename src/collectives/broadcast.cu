#include <cuda_runtime.h>
#include <iostream>

#include "checker.cuh"
#include "data.cuh"

#define N 4194304
#define BYTES (N * sizeof(float))
#define NUM_GPUS 8
#define ROOT 0

int main(int argc, char **argv)
{
    float *h_a = new float[N];
    initVector<float>(h_a, N);

    int *devices = new int[NUM_GPUS];
    float **data = new float *[N];
    cudaStream_t *streams = new cudaStream_t[NUM_GPUS];

    for (uint i = 0; i < NUM_GPUS; i++)
    {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaMalloc(&data[i], BYTES));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Enable peer-to-peer access
    for (uint i = 0; i < NUM_GPUS; i++)
    {
        for (uint j = 0; j < NUM_GPUS; j++)
        {
            if (i != j)
            {
                CHECK_CUDA(cudaSetDevice(devices[i]));
                int can_access;
                CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, devices[i], devices[j]));
                if (can_access)
                {
                    CHECK_CUDA(cudaDeviceEnablePeerAccess(devices[j], 0));
                }
            }
        }
    }

    // Initialize data on root device
    CHECK_CUDA(cudaSetDevice(devices[ROOT]));
    CHECK_CUDA(cudaMemcpy(data[ROOT], h_a, BYTES, cudaMemcpyHostToDevice));

    // Broadcast from root to all other devices using peer-to-peer copy
    for (uint i = 0; i < NUM_GPUS; i++)
    {
        if (i != ROOT)
        {
            CHECK_CUDA(cudaSetDevice(devices[i]));
            CHECK_CUDA(cudaMemcpyPeerAsync(data[i], devices[i], data[ROOT], devices[ROOT], BYTES, streams[i]));
        }
    }

    // Synchronize all streams
    for (size_t i = 0; i < NUM_GPUS; i++)
    {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    // Verify results
    std::cout << "Broadcast completed. Verifying results...\n";
    for (uint i = 0; i < NUM_GPUS; i++)
    {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        float *verify_data = new float[N];
        CHECK_CUDA(cudaMemcpy(verify_data, data[i], BYTES, cudaMemcpyDeviceToHost));

        bool success = true;
        for (int j = 0; j < N; j++)
        {
            if (verify_data[j] != h_a[j])
            {
                success = false;
                break;
            }
        }
        std::cout << "Device " << devices[i] << ": " << (success ? "PASS" : "FAIL") << std::endl;
        delete[] verify_data;
    }

    // Cleanup
    for (uint i = 0; i < NUM_GPUS; i++)
    {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaFree(data[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    delete[] devices;
    delete[] data;
    delete[] streams;
    delete[] h_a;

    return 0;
}