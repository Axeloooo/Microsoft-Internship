__global__ void dotProductGpu(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        *c += a[i] * b[i];
    }
}