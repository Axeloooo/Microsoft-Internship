__global__ void sdot(const float *a, const float *b, float *c, int n)
{
    // TODO Optimize the dot product kernel for better performance.

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        atomicAdd(c, a[i] * b[i]);
    }
}