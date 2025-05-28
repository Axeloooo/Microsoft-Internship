#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#include "timer.cuh"
#include "checker.cuh"
#include "reduction.cuh"

#define N 4194304
#define BLOCK_SIZE 256

void initData(int *vector, int n)
{
    srand(42);
    for (int i = 0; i < n; i++)
    {
        vector[i] = rand() % 100;
    }
}

void checkResult(const int *cpuResult, const int *gpuResult)
{
    if (std::fabs(*cpuResult - *gpuResult) > 1e-5)
    {
        std::cerr << "Results do not match!" << std::endl;
        std::cerr << "CPU result: " << *cpuResult << ", GPU result: " << *gpuResult << "\n"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "Results are correct!" << "\n"
              << std::endl;
}

void reductionCpu(const int *h_in_data, int *h_out_data, unsigned int n)
{
    h_out_data[0] = 0;
    for (unsigned int i = 0; i < n; i++)
    {
        h_out_data[0] += h_in_data[i];
    }
}

int main()
{
    size_t bytes = N * sizeof(int);
    const int gridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int numBlocks = (N + (2 * BLOCK_SIZE) - 1) / (2 * BLOCK_SIZE);

    int *h_in = new int[N];
    int *h_out_cpu = new int[1];
    int *h_out_gpu = new int[1];

    initData(h_in, N);

    int *d_in, *d_out;

    // Allocate memory on GPU
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(int)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Perform CPU computation
    CudaTimer cpuTimer;
    cpuTimer.start();
    reductionCpu(h_in, h_out_cpu, N);
    cpuTimer.stop();
    cpuTimer.print(&bytes, "CPU");
    std::cout << "CPU result: " << h_out_cpu[0] << "\n"
              << std::endl;

    // Perfom warm up
    std::cout << "Warming up GPUs..." << "\n"
              << std::endl;
    for (int i = 0; i < 3; i++)
    {
        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        naiveReduce<<<gridDim, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        interleaveAddressingReduce<<<gridDim, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        sequenceAddressingReduce<<<gridDim, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        addDuringLoadReduce<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        unrollLastWarpReduce<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        completelyUnrollReduce<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        multipleElementsReduce<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Perform naive reduce GPU computation
    CudaTimer naiveReduceTimer;
    naiveReduceTimer.start();
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    naiveReduce<<<gridDim, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    naiveReduceTimer.stop();
    naiveReduceTimer.print(&bytes, "Naive Reduce GPU");

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform interleaved addressing reduce GPU computation
    CudaTimer interleavedReduceTimer;
    interleavedReduceTimer.start();
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    interleaveAddressingReduce<<<gridDim, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    interleavedReduceTimer.stop();
    interleavedReduceTimer.print(&bytes, "Interleaved Addressing Reduce GPU");

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform sequence addressing reduce GPU computation
    CudaTimer sequenceReduceTimer;
    sequenceReduceTimer.start();
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    naiveReduce<<<gridDim, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    sequenceReduceTimer.stop();
    sequenceReduceTimer.print(&bytes, "Sequence Addressing Reduce GPU");

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform add during load reduce GPU computation
    CudaTimer addDuringLoadReduceTimer;
    addDuringLoadReduceTimer.start();
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    addDuringLoadReduce<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    addDuringLoadReduceTimer.stop();
    addDuringLoadReduceTimer.print(&bytes, "Add During Load Reduce GPU");

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform unroll last warp reduce GPU computation
    CudaTimer unrollLastWarpReduceTimer;
    unrollLastWarpReduceTimer.start();
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    unrollLastWarpReduce<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    unrollLastWarpReduceTimer.stop();
    unrollLastWarpReduceTimer.print(&bytes, "Unroll Last Warp Reduce GPU");

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform completely unrolled warp reduce GPU computation
    CudaTimer completelyUnrolledWarpReduceTimer;
    completelyUnrolledWarpReduceTimer.start();
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    completelyUnrollReduce<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    completelyUnrolledWarpReduceTimer.stop();
    completelyUnrolledWarpReduceTimer.print(&bytes, "Completely Unrolled Warp Reduce GPU");

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform multiple elements reduce GPU computation
    CudaTimer multipleElementsReduceTimer;
    multipleElementsReduceTimer.start();
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    multipleElementsReduce<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    multipleElementsReduceTimer.stop();
    multipleElementsReduceTimer.print(&bytes, "Multiple Elements Reduce GPU");

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Free Memory
    delete[] h_in;
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}