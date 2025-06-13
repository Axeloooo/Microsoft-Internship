/**
 * @file data.cuh
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Utility functions for initializing and verifying vectors and matrices
 * @version 0.1
 * @date 2025-06-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/**
 * @brief Utility functions for initializing and verifying vectors and matrices
 *
 * @tparam T Data type for the vector or matrix (e.g., float, double)
 * @param vector Pointer to the vector data
 * @param n  Number of elements in the vector
 */
template <typename T>
void InitVector(T *vector, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        vector[i] = static_cast<T>(rand()) / RAND_MAX;
    }
}

/**
 * @brief Initialize a matrix with random values
 *
 * @tparam T Data type for the matrix (e.g., float, double)
 * @param matrix Pointer to the matrix data
 * @param rows  Number of rows in the matrix
 * @param cols  Number of columns in the matrix
 */
template <typename T>
void InitializeMatrix(T *matrix, int rows, int cols) {
    srand(42);
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<T>(rand()) / RAND_MAX;
    }
}

/**
 * @brief Verify the results of a vector operation
 *
 * @tparam T Data type for the vector (e.g., float, double)
 * @param d_y Pointer to the device output vector
 * @param h_y Pointer to the host expected output vector
 * @param n   Number of elements in the vector
 */
template <typename T>
void VerifyVector(const T *d_y, const T *h_y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (fabs(d_y[i] - h_y[i]) > 1e-5) {
            std::cerr << "Results do not match!" << std::endl;
            std::cerr << "Data mismatch at index " << i << ": " << d_y[i]
                      << " != " << h_y[i] << std::endl;
            return;
        }
    }
    std::cout << "Results are correct!" << std::endl;
}

/**
 * @brief Verify the results of a matrix operation
 *
 * @tparam T Data type for the matrix (e.g., float, double)
 * @param cpuC Pointer to the CPU output matrix
 * @param gpuC Pointer to the GPU output matrix
 * @param m Number of rows in the matrix
 * @param n Number of columns in the matrix
 */
template <typename T>
void VerifyMatrix(const T *cpuC, const T *gpuC, int m, int n) {
    double diff = 0.0;
    for (int i = 0; i < m * n; i++) {
        diff = std::fabs(cpuC[i] - gpuC[i]);
        if (diff > 0.01) {
            std::cerr << "Results don't match!" << std::endl;
            return;
        }
    }
    std::cout << "Results match!" << std::endl;
}