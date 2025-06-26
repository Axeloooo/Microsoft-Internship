__global__ void NaiveGemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);

template <unsigned int BLOCK_SIZE>
__global__ void CoalescingGemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);

template <unsigned int BLOCK_SIZE>
__global__ void CacheGemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);

template <unsigned int BM, unsigned int BN, unsigned int BK, unsigned int TM>
__global__ void Tiling1DGemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);