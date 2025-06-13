/**
 * @file data.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Utility functions for initializing and verifying vectors and matrices
 * @version 0.1
 * @date 2025-06-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "data.cuh"

/**
 * @brief Initialize a vector with random values
 *
 * @tparam T Data type for the vector or matrix (e.g., float, double)
 * @param vector Pointer to the vector data
 * @param n  Number of elements in the vector
 */
template <typename T>
void InitVector(T *vector, const std::size_t &n) {
    srand(42);
    for (std::size_t i = 0; i < n; i++) {
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
void InitializeMatrix(T *matrix, const std::size_t &rows, const std::size_t &cols) {
    srand(42);
    for (std::size_t i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<T>(rand()) / RAND_MAX;
    }
}

/**
 * @brief Verify the results of a vector operation
 *
 * @tparam T Data type for the vector (e.g., float, double)
 * @param h_vector Pointer to the device output vector
 * @param d_vector Pointer to the host expected output vector
 * @param n   Number of elements in the vector
 */
template <typename T>
void VerifyVector(const T *h_vector, const T *d_vector, const std::size_t &n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (std::fabs(d_vector[i] - h_vector[i]) > 1e-5) {
            std::cerr << "Results do not match!" << std::endl;
            std::cerr << "Data mismatch at index " << i << ": " << d_vector[i] << " != " << h_vector[i] << std::endl;
            return;
        }
    }
    std::cout << "Results are correct!" << std::endl;
}

/**
 * @brief Verify the results of a matrix operation
 *
 * @tparam T Data type for the matrix (e.g., float, double)
 * @param h_matrix Pointer to the CPU output matrix
 * @param d_matrix Pointer to the GPU output matrix
 * @param m Number of rows in the matrix
 * @param n Number of columns in the matrix
 */
template <typename T>
void VerifyMatrix(const T *h_matrix, const T *d_matrix, const std::size_t &m, const std::size_t &n) {
    double diff = 0.0;
    for (std::size_t i = 0; i < m * n; i++) {
        diff = std::fabs(d_matrix[i] - h_matrix[i]);
        if (diff > 0.01) {
            std::cerr << "Results don't match!" << std::endl;
            return;
        }
    }
    std::cout << "Results match!" << std::endl;
}