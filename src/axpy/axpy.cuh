__global__ void axpyGpu(const float alpha, const float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    y[i] = alpha * x[i] + y[i];
}