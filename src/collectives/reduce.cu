#include <cuda_runtime.h>
#include <iostream>

#include "checker.cuh"
#include "data.cuh"

#define N 4194304
#define BYTES (N * sizeof(float))
#define BLOCK 256
#define GRID (N + BLOCK - 1 / BLOCK)
#define ROOT_GPU 0

__global__ void vectorAdd(float *a, float *b, float *c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main(int argc, char **argv)
{
    int NUM_GPUS = 0;
    CHECK_CUDA(cudaGetDeviceCount(&NUM_GPUS));
    std::cout << "Number of GPUs available: " << NUM_GPUS << std::endl;

    // Host data for each GPU
    float **h_in = new float *[NUM_GPUS];
    float *h_out = new float[N];

    // Initialize expected result to zero
    for (int i = 0; i < N; i++)
    {
        h_out[i] = 0.0f;
    }

    for (int i = 0; i < NUM_GPUS; i++)
    {
        h_in[i] = new float[N];
        initVector<float>(h_in[i], N);

        for (int j = 0; j < N; j++)
        {
            h_out[j] += h_in[i][j];
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
        CHECK_CUDA(cudaMemcpy(d_in[i], h_in[i], BYTES, cudaMemcpyHostToDevice));
    }

    cudaSetDevice(devices[ROOT_GPU]);
    cudaMemcpy(d_out[ROOT_GPU], d_in[ROOT_GPU], BYTES, cudaMemcpyDeviceToDevice);

    // Manual reduce all non-root GPUs send their data to root GPU
    for (int i = 0; i < NUM_GPUS; i++)
    {
        if (i != ROOT_GPU)
        {
            CHECK_CUDA(cudaSetDevice(devices[ROOT_GPU]));
            CHECK_CUDA(cudaMemcpyPeerAsync(d_tmp[ROOT_GPU], devices[ROOT_GPU], d_in[i], devices[i], BYTES, streams[ROOT_GPU]));
            CHECK_CUDA(cudaStreamSynchronize(streams[ROOT_GPU]));
            vectorAdd<<<GRID, BLOCK, 0, streams[ROOT_GPU]>>>(d_out[ROOT_GPU], d_tmp[ROOT_GPU], d_out[ROOT_GPU], N);
            CHECK_CUDA(cudaGetLastError());
        }
    }

    // Synchronize root GPU stream
    CHECK_CUDA(cudaSetDevice(devices[ROOT_GPU]));
    CHECK_CUDA(cudaStreamSynchronize(streams[ROOT_GPU]));

    // Verify results - only check root GPU
    std::cout << "Reduce completed. Verifying result on root GPU " << ROOT_GPU << "...\n";

    CHECK_CUDA(cudaSetDevice(devices[ROOT_GPU]));
    float *verify_data = new float[N];
    CHECK_CUDA(cudaMemcpy(verify_data, d_out[ROOT_GPU], BYTES, cudaMemcpyDeviceToHost));

    bool success = true;
    for (int j = 0; j < N; j++)
    {
        if (abs(verify_data[j] - h_out[j]) > 1e-5)
        {
            success = false;
            std::cout << "Mismatch at index " << j
                      << ": expected " << h_out[j] << ", got " << verify_data[j] << std::endl;
            break;
        }
    }
    std::cout << "Root GPU " << ROOT_GPU << ": " << (success ? "PASS" : "FAIL") << std::endl;

    delete[] verify_data;

    // Cleanup
    for (int i = 0; i < NUM_GPUS; i++)
    {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaFree(d_in[i]));
        CHECK_CUDA(cudaFree(d_out[i]));
        CHECK_CUDA(cudaFree(d_tmp[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        delete[] h_in[i];
    }
    delete[] devices;
    delete[] d_in;
    delete[] d_out;
    delete[] d_tmp;
    delete[] streams;
    delete[] h_in;
    delete[] h_out;

    return 0;
}
