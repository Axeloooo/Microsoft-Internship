#include <cuda_runtime.h>
#include <iostream>

#include "checker.cuh"
#include "data.cuh"

#define N 4194304
#define BYTES (N * sizeof(float))
#define BLOCK 256
#define GRID ((N + BLOCK - 1) / BLOCK)
#define ROOT 0

__global__ void vectorAdd(float *a, float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char **argv)
{
    int NUM_GPUS = 0;
    CHECK_CUDA(cudaGetDeviceCount(&NUM_GPUS));
    std::cout << "Number of GPUs available: " << NUM_GPUS << std::endl;

    // Host data for each GPU
    float **h_a = new float *[NUM_GPUS];
    float *h_expected = new float[N];

    // Initialize expected result to zero
    for (int i = 0; i < N; i++)
    {
        h_expected[i] = 0.0f;
    }

    for (int i = 0; i < NUM_GPUS; i++)
    {
        h_a[i] = new float[N];
        initVector<float>(h_a[i], N);

        for (int j = 0; j < N; j++)
        {
            h_expected[j] += h_a[i][j];
        }
    }

    // Device arrays
    float **d_in = new float *[NUM_GPUS];
    float **d_out = new float *[NUM_GPUS];
    float **d_tmp = new float *[NUM_GPUS];
    int *devices = new int[NUM_GPUS];
    cudaStream_t *streams = new cudaStream_t[NUM_GPUS];

    // Initialize devices array with sequential device IDs
    for (int i = 0; i < NUM_GPUS; i++)
    {
        devices[i] = i;
    }

    // Allocate memory on all devices
    for (int i = 0; i < NUM_GPUS; i++)
    {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaMalloc(&d_in[i], BYTES));
        CHECK_CUDA(cudaMalloc(&d_out[i], BYTES));
        CHECK_CUDA(cudaMalloc(&d_tmp[i], BYTES));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Enable peer-to-peer access
    for (int i = 0; i < NUM_GPUS; i++)
    {
        for (int j = 0; j < NUM_GPUS; j++)
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

    // Initialize input data on each device
    for (int i = 0; i < NUM_GPUS; i++)
    {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaMemcpy(d_in[i], h_a[i], BYTES, cudaMemcpyHostToDevice));
    }

    // Step 1: Reduce - accumulate all data to GPU 0 (root)
    CHECK_CUDA(cudaSetDevice(devices[ROOT]));
    CHECK_CUDA(cudaMemcpy(d_out[ROOT], d_in[ROOT], BYTES, cudaMemcpyDeviceToDevice));

    for (int i = 0; i < NUM_GPUS; i++)
    {
        if (i != ROOT)
        {
            CHECK_CUDA(cudaSetDevice(devices[ROOT]));
            CHECK_CUDA(cudaMemcpyPeerAsync(d_tmp[ROOT], devices[ROOT], d_in[i], devices[i], BYTES, streams[ROOT]));
            CHECK_CUDA(cudaStreamSynchronize(streams[ROOT]));
            vectorAdd<<<GRID, BLOCK, 0, streams[ROOT]>>>(d_out[ROOT], d_tmp[ROOT], d_out[ROOT], N);
            CHECK_CUDA(cudaGetLastError());
        }
    }

    // Step 2: Broadcast - copy reduced result from root to all other GPUs
    for (uint i = 0; i < NUM_GPUS; i++)
    {
        if (i != ROOT)
        {
            CHECK_CUDA(cudaSetDevice(devices[i]));
            CHECK_CUDA(cudaMemcpyPeerAsync(d_out[i], devices[i], d_out[ROOT], devices[ROOT], BYTES, streams[i]));
        }
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_GPUS; i++)
    {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    // Verify results
    std::cout << "Allreduce completed. Verifying results...\n";
    for (int i = 0; i < NUM_GPUS; i++)
    {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        float *verify_data = new float[N];
        CHECK_CUDA(cudaMemcpy(verify_data, d_out[i], BYTES, cudaMemcpyDeviceToHost));

        bool success = true;
        for (int j = 0; j < N; j++)
        {
            if (abs(verify_data[j] - h_expected[j]) > 1e-5)
            {
                success = false;
                std::cout << "Mismatch at GPU " << i << ", index " << j
                          << ": expected " << h_expected[j] << ", got " << verify_data[j] << std::endl;
                break;
            }
        }
        std::cout << "Device " << devices[i] << ": " << (success ? "PASS" : "FAIL") << std::endl;
        delete[] verify_data;
    }

    // Cleanup
    for (int i = 0; i < NUM_GPUS; i++)
    {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaFree(d_in[i]));
        CHECK_CUDA(cudaFree(d_out[i]));
        CHECK_CUDA(cudaFree(d_tmp[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        delete[] h_a[i];
    }
    delete[] devices;
    delete[] d_in;
    delete[] d_out;
    delete[] d_tmp;
    delete[] streams;
    delete[] h_a;
    delete[] h_expected;

    return 0;
}
