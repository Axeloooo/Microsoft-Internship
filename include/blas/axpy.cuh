__global__ void naiveSaxpy(const float alpha, const float *x, float *y, const int n);

__global__ void vectorizedSaxpy(const float alpha, const float4 *x4, float4 *y4, const int nv);