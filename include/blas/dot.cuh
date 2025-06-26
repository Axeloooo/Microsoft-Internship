__global__ void naiveSdot(const float *a, const float *b, float *c, const int n);

template <unsigned int BLOCK_SIZE>
__device__ void warpTemplateReduce(volatile float *sharedMemory, const unsigned int tid);

template <unsigned int BLOCK_SIZE>
__global__ void optimizedSdot(const float *a, const float *b, float *c, const int n);