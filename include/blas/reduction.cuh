__global__ void naiveReduce(int *g_in_data, int *g_out_data);

__global__ void interleaveAddressingReduce(int *g_in_data, int *g_out_data);

__global__ void sequenceAddressingReduce(int *g_in_data, int *g_out_data);

__global__ void addDuringLoadReduce(int *g_in_data, int *g_out_data);

__device__ void warpReduce(volatile int *sdata, int tid);

__global__ void unrollLastWarpReduce(int *g_in_data, int *g_out_data);

template <unsigned int BLOCK_SIZE>
__device__ void warpTemplateReduce(volatile int *sdata, unsigned int tid);

template <unsigned int BLOCK_SIZE>
__global__ void completelyUnrollReduce(int *g_in_data, int *g_out_data);

template <unsigned int BLOCK_SIZE>
__global__ void multipleElementsReduce(int *g_in_data, int *g_out_data, unsigned int n);