__global__ void saxpy(const float alpha, const float *x, float *y, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        y[i] = alpha * x[i] + y[i];
    }
}