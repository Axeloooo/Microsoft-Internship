__global__ void naiveSgemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N)
    {
        float tmp = 0.0;
        for (uint i = 0; i < K; ++i)
        {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

template <const uint BLOCK_SIZE>
__global__ void coalescingSgemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    const int cRow = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const int cCol = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if (cRow < M && cCol < N)
    {
        float tmp = 0.0;
        for (uint i = 0; i < K; ++i)
        {
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
    }
}