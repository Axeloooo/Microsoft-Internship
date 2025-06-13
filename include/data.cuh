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

#ifndef DATA_CUH
#define DATA_CUH

/**
 * @brief Utility functions for initializing and verifying vectors and matrices
 *
 * @tparam T Data type for the vector or matrix (e.g., float, double)
 * @param vector Pointer to the vector data
 * @param n  Number of elements in the vector
 */
template <typename T>
void InitVector(T *vector, int n);

/**
 * @brief Initialize a matrix with random values
 *
 * @tparam T Data type for the matrix (e.g., float, double)
 * @param matrix Pointer to the matrix data
 * @param rows  Number of rows in the matrix
 * @param cols  Number of columns in the matrix
 */
template <typename T>
void InitializeMatrix(T *matrix, int rows, int cols);

/**
 * @brief Verify the results of a vector operation
 *
 * @tparam T Data type for the vector (e.g., float, double)
 * @param d_y Pointer to the device output vector
 * @param h_y Pointer to the host expected output vector
 * @param n   Number of elements in the vector
 */
template <typename T>
void VerifyVector(const T *h_vector, const T *d_vector, size_t n);

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
void VerifyMatrix(const T *h_matrix, const T *d_matrix, int m, int n);

#endif