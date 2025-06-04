__global__ void saxpy(const float alpha, const float *x, float *y, const int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride)
    {
        y[i] = alpha * x[i] + y[i];
    }
}