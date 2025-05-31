#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#include "timer.cuh"
#include "checker.cuh"
#include "data.cuh"
#include "reduction.cuh"

#define N 4194304
#define BLOCK_SIZE 256

void checkResult(const int &cpuResult, const int &gpuResult)
{
    if (std::fabs(cpuResult - gpuResult) > 1e-5)
    {
        std::cerr << "Results do not match!" << std::endl;
        std::cerr << "CPU result: " << cpuResult << ", GPU result: " << gpuResult << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "Results are correct!" << std::endl;
}

void reductionCpu(const int *h_in_data, int *h_out_data, unsigned int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        *h_out_data += h_in_data[i];
    }
}

int main()
{
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const int numBlocks = (N + (2 * BLOCK_SIZE) - 1) / (2 * BLOCK_SIZE);

    size_t bytes = N * sizeof(int);

    int *h_in = new int[N];
    int h_out_cpu, h_out_gpu = 0;

    // Initialize data
    initVector<int>(h_in, N);

    int *d_in, *d_out;

    // Allocate memory on GPU
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(int)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, h_out_gpu, sizeof(int)));

    // Perform CPU computation
    CudaTimer cpuTimer("Reduction CPU", &bytes);
    cpuTimer.start();
    reductionCpu(h_in, &h_out_cpu, N);
    cpuTimer.stop();
    cpuTimer.print();

    // Perfom warm up
    std::cout << "Warming up GPUs..." << std::endl;
    for (int i = 0; i < 3; i++)
    {
        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        naiveReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        interleaveAddressingReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        sequenceAddressingReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        addDuringLoadReduce<<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        unrollLastWarpReduce<<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        completelyUnrollReduce<BLOCK_SIZE><<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        multipleElementsReduce<BLOCK_SIZE><<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Perform naive reduce using CUDA kernel
    CudaTimer naiveReduceTimer("Naive Reduce GPU", &bytes);
    naiveReduceTimer.start();
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    naiveReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CHECK_CUDA(cudaGetLastError());
    naiveReduceTimer.stop();
    naiveReduceTimer.print();

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(&h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform interleaved addressing reduce using CUDA kernel
    CudaTimer interleavedReduceTimer("Interleaved Addressing Reduce GPU", &bytes);
    interleavedReduceTimer.start();
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    interleaveAddressingReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CHECK_CUDA(cudaGetLastError());
    interleavedReduceTimer.stop();
    interleavedReduceTimer.print();

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(&h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform sequence addressing reduce using CUDA kernel
    CudaTimer sequenceReduceTimer("Sequence Addressing Reduce GPU", &bytes);
    sequenceReduceTimer.start();
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    naiveReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CHECK_CUDA(cudaGetLastError());
    sequenceReduceTimer.stop();
    sequenceReduceTimer.print();

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(&h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform add during load reduce using CUDA kernel
    CudaTimer addDuringLoadReduceTimer("Add During Load Reduce GPU", &bytes);
    addDuringLoadReduceTimer.start();
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    addDuringLoadReduce<<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CHECK_CUDA(cudaGetLastError());
    addDuringLoadReduceTimer.stop();
    addDuringLoadReduceTimer.print();

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(&h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform unroll last warp reduce using CUDA kernel
    CudaTimer unrollLastWarpReduceTimer("Unroll Last Warp Reduce GPU", &bytes);
    unrollLastWarpReduceTimer.start();
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    unrollLastWarpReduce<<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CHECK_CUDA(cudaGetLastError());
    unrollLastWarpReduceTimer.stop();
    unrollLastWarpReduceTimer.print();

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(&h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform completely unrolled warp reduce using CUDA kernel
    CudaTimer completelyUnrolledWarpReduceTimer("Completely Unrolled Warp Reduce GPU", &bytes);
    completelyUnrolledWarpReduceTimer.start();
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    completelyUnrollReduce<BLOCK_SIZE><<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CHECK_CUDA(cudaGetLastError());
    completelyUnrolledWarpReduceTimer.stop();
    completelyUnrolledWarpReduceTimer.print();

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(&h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Perform multiple elements reduce using CUDA kernel
    CudaTimer multipleElementsReduceTimer("Multiple Elements Reduce GPU", &bytes);
    multipleElementsReduceTimer.start();
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    multipleElementsReduce<BLOCK_SIZE><<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);
    CHECK_CUDA(cudaGetLastError());
    multipleElementsReduceTimer.stop();
    multipleElementsReduceTimer.print();

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(&h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_out_cpu, h_out_gpu);

    // Clean up
    delete[] h_in;
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}