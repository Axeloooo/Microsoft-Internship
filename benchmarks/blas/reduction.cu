#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "blas/reduction.cuh"
#include "utils/checker.cuh"
#include "utils/data.cuh"
#include "utils/timer.cuh"

void checkResult(const int &cpuResult, const int &gpuResult) {
    if (std::fabs(cpuResult - gpuResult) > 1e-5) {
        std::cerr << "Results do not match!" << std::endl;
        std::cerr << "CPU result: " << cpuResult << ", GPU result: " << gpuResult << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "Results are correct!" << std::endl;
}

void reductionCpu(const int *h_in_data, int *h_out_data, const unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        *h_out_data += h_in_data[i];
    }
}

int main() {
    const unsigned int N = 4194304;
    const unsigned int BLOCK_SIZE = 256;

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const int numBlocks = (N + (2 * BLOCK_SIZE) - 1) / (2 * BLOCK_SIZE);

    unsigned int bytes = N * sizeof(int);

    int *h_in = new int[N];
    int h_out_cpu, h_out_gpu = 0;

    // Initialize data
    InitVector<int>(h_in, N);

    int *d_in, *d_out;

    // Allocate memory on GPU
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, h_out_gpu, sizeof(int)));

    // Perform CPU computation
    CudaTimer cpuTimer("Reduction CPU", bytes);
    cpuTimer.Start();
    reductionCpu(h_in, &h_out_cpu, N);
    cpuTimer.Stop();
    cpuTimer.Print();

    // Perfom warm up
    std::cout << "Warming up GPUs..." << std::endl;
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
        naiveReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
        interleaveAddressingReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
        sequenceAddressingReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
        addDuringLoadReduce<<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
        unrollLastWarpReduce<<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
        completelyUnrollReduce<BLOCK_SIZE><<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
        multipleElementsReduce<BLOCK_SIZE><<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Perform naive reduce using CUDA kernel
    CudaTimer naiveReduceTimer("Naive Reduce GPU", bytes);
    naiveReduceTimer.Start();
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
    naiveReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    naiveReduceTimer.Stop();
    naiveReduceTimer.Print();

    // Copy data from device to host
    CUDA_CHECK(cudaMemcpy(&h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform interleaved addressing reduce using CUDA kernel
    CudaTimer interleavedReduceTimer("Interleaved Addressing Reduce GPU", bytes);
    interleavedReduceTimer.Start();
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
    interleaveAddressingReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    interleavedReduceTimer.Stop();
    interleavedReduceTimer.Print();

    // Copy data from device to host
    CUDA_CHECK(cudaMemcpy(&h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform sequence addressing reduce using CUDA kernel
    CudaTimer sequenceReduceTimer("Sequence Addressing Reduce GPU", bytes);
    sequenceReduceTimer.Start();
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
    naiveReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    sequenceReduceTimer.Stop();
    sequenceReduceTimer.Print();

    // Copy data from device to host
    CUDA_CHECK(cudaMemcpy(&h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform add during load reduce using CUDA kernel
    CudaTimer addDuringLoadReduceTimer("Add During Load Reduce GPU", bytes);
    addDuringLoadReduceTimer.Start();
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
    addDuringLoadReduce<<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    addDuringLoadReduceTimer.Stop();
    addDuringLoadReduceTimer.Print();

    // Copy data from device to host
    CUDA_CHECK(cudaMemcpy(&h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform unroll last warp reduce using CUDA kernel
    CudaTimer unrollLastWarpReduceTimer("Unroll Last Warp Reduce GPU", bytes);
    unrollLastWarpReduceTimer.Start();
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
    unrollLastWarpReduce<<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unrollLastWarpReduceTimer.Stop();
    unrollLastWarpReduceTimer.Print();

    // Copy data from device to host
    CUDA_CHECK(cudaMemcpy(&h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform completely unrolled warp reduce using CUDA kernel
    CudaTimer completelyUnrolledWarpReduceTimer("Completely Unrolled Warp Reduce GPU", bytes);
    completelyUnrolledWarpReduceTimer.Start();
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
    completelyUnrollReduce<BLOCK_SIZE><<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    completelyUnrolledWarpReduceTimer.Stop();
    completelyUnrolledWarpReduceTimer.Print();

    // Copy data from device to host
    CUDA_CHECK(cudaMemcpy(&h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform multiple elements reduce using CUDA kernel
    CudaTimer multipleElementsReduceTimer("Multiple Elements Reduce GPU", bytes);
    multipleElementsReduceTimer.Start();
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
    multipleElementsReduce<BLOCK_SIZE><<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    multipleElementsReduceTimer.Stop();
    multipleElementsReduceTimer.Print();

    // Copy data from device to host
    CUDA_CHECK(cudaMemcpy(&h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Clean up
    delete[] h_in;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}