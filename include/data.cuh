template <typename T>
void initVector(T *vector, int n)
{
    srand(42);
    for (int i = 0; i < n; i++)
    {
        vector[i] = static_cast<T>(rand()) / RAND_MAX;
    }
}

template <typename T>
void initializeMatrix(T *matrix, int rows, int cols)
{
    srand(42);
    for (int i = 0; i < rows * cols; ++i)
    {
        matrix[i] = static_cast<T>(rand()) / RAND_MAX;
    }
}

template <typename T>
void verifyVector(const T *d_y, const T *h_y, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        if (fabs(d_y[i] - h_y[i]) > 1e-5)
        {
            std::cerr << "Results do not match!" << std::endl;
            std::cerr << "Data mismatch at index " << i << ": " << d_y[i] << " != " << h_y[i] << std::endl;
            return;
        }
    }
    std::cout << "Results are correct!" << std::endl;
}

template <typename T>
void verifyMatrix(const T *cpuC, const T *gpuC, int m, int n)
{
    double diff = 0.0;
    for (int i = 0; i < m * n; i++)
    {
        diff = std::fabs(cpuC[i] - gpuC[i]);
        if (diff > 0.01)
        {
            std::cerr << "Results don't match!" << std::endl;
            return;
        }
    }
    std::cout << "Results match!" << std::endl;
}