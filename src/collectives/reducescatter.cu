#include <cuda_runtime.h>

#include <cstdlib>

#include "checker.cuh"
#include "data.cuh"

#define N 1048576
#define BYTES (N * sizeof(float))
#define BLOCK 256
#define GRID (N + BLOCK - 1) / BLOCK

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main(int argc, char **argv) {
    int NUM_GPUS = 0;
    CHECK_CUDA(cudaGetDeviceCount(&NUM_GPUS));
    std::cout << "Number of GPUs available: " << NUM_GPUS << std::endl;

    if (NUM_GPUS == 0) {
        std::cerr << "No GPUs available!" << std::endl;
        return 1;
    }

    // Calculate chunk size per GPU
    int chunk_size = N / NUM_GPUS;
    int chunk_bytes = chunk_size * sizeof(float);

    // Host data for each GPU
    float **h_in = new float *[NUM_GPUS];
    float **h_out = new float *[NUM_GPUS];

    // Initialize input data and expected results
    for (int i = 0; i < NUM_GPUS; i++) {
        h_in[i] = new float[N];
        h_out[i] = new float[chunk_size];

        InitVector<float>(h_in[i], N);
    }

    // Calculate expected results after ALL input data is initialized
    for (int i = 0; i < NUM_GPUS; i++) {
        // Calculate expected result for this GPU's chunk
        for (int j = 0; j < chunk_size; j++) {
            int global_idx = i * chunk_size + j;
            h_out[i][j] = 0.0f;
            // Sum this element across all GPUs
            for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
                h_out[i][j] += h_in[gpu][global_idx];
            }
        }
    }

    // Device arrays
    float **d_in = new float *[NUM_GPUS];
    float **d_out = new float *[NUM_GPUS];
    float **d_tmp = new float *[NUM_GPUS];
    int *devices = new int[NUM_GPUS];
    cudaStream_t *streams = new cudaStream_t[NUM_GPUS];

    // Initialize devices array with sequential device IDs
    for (int i = 0; i < NUM_GPUS; i++) {
        devices[i] = i;
    }

    // Allocate memory on all devices
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaMalloc(&d_in[i], BYTES));
        CHECK_CUDA(cudaMalloc(&d_out[i], chunk_bytes));
        CHECK_CUDA(cudaMalloc(&d_tmp[i], chunk_bytes));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Enable peer-to-peer access
    for (int i = 0; i < NUM_GPUS; i++) {
        for (int j = 0; j < NUM_GPUS; j++) {
            if (i != j) {
                CHECK_CUDA(cudaSetDevice(devices[i]));
                int can_access;
                CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, devices[i],
                                                   devices[j]));
                if (can_access) {
                    CHECK_CUDA(cudaDeviceEnablePeerAccess(devices[j], 0));
                } else {
                    std::cout << "Peer access not supported between GPU "
                              << devices[i] << " and GPU " << devices[j]
                              << std::endl;
                }
            }
        }
    }

    // Initialize input data on each device
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaMemcpy(d_in[i], h_in[i], BYTES, cudaMemcpyHostToDevice));
    }

    // Manual reduce-scatter implementation
    for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
        CHECK_CUDA(cudaSetDevice(devices[gpu]));

        // Initialize this GPU's output with its own chunk
        int chunk_offset = gpu * chunk_size;
        CHECK_CUDA(cudaMemcpy(d_out[gpu], d_in[gpu] + chunk_offset, chunk_bytes,
                              cudaMemcpyDeviceToDevice));

        // Reduce chunks from other GPUs
        for (int src_gpu = 0; src_gpu < NUM_GPUS; src_gpu++) {
            if (src_gpu != gpu) {
                // Copy the chunk from source GPU to temporary buffer
                CHECK_CUDA(cudaMemcpyPeerAsync(
                    d_tmp[gpu], devices[gpu], d_in[src_gpu] + chunk_offset,
                    devices[src_gpu], chunk_bytes, streams[gpu]));
                CHECK_CUDA(cudaStreamSynchronize(streams[gpu]));

                // Add to the accumulated result
                int grid_size = (chunk_size + BLOCK - 1) / BLOCK;
                vectorAdd<<<grid_size, BLOCK, 0, streams[gpu]>>>(
                    d_out[gpu], d_tmp[gpu], d_out[gpu], chunk_size);
                CHECK_CUDA(cudaGetLastError());
            }
        }

        // Synchronize this GPU's stream
        CHECK_CUDA(cudaStreamSynchronize(streams[gpu]));
    }

    // Verify results on all GPUs
    std::cout << "Reduce-scatter completed. Verifying results...\n";

    bool all_success = true;
    for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
        CHECK_CUDA(cudaSetDevice(devices[gpu]));
        float *verify_data = new float[chunk_size];
        CHECK_CUDA(cudaMemcpy(verify_data, d_out[gpu], chunk_bytes,
                              cudaMemcpyDeviceToHost));

        bool success = true;
        for (int j = 0; j < chunk_size; j++) {
            if (abs(verify_data[j] - h_out[gpu][j]) > 1e-5) {
                success = false;
                std::cout << "GPU " << gpu << " mismatch at index " << j
                          << ": expected " << h_out[gpu][j] << ", got "
                          << verify_data[j] << std::endl;
                break;
            }
        }
        std::cout << "GPU " << gpu << ": " << (success ? "PASS" : "FAIL")
                  << std::endl;

        if (!success) all_success = false;
        delete[] verify_data;
    }

    std::cout << "Overall result: " << (all_success ? "PASS" : "FAIL")
              << std::endl;

    // Cleanup
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaFree(d_in[i]));
        CHECK_CUDA(cudaFree(d_out[i]));
        CHECK_CUDA(cudaFree(d_tmp[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        delete[] h_in[i];
        delete[] h_out[i];
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
