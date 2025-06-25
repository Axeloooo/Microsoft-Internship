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

#include <cstddef>
#include <string>
#include <vector>

/**
 * @brief Utility functions for initializing and verifying vectors and matrices
 *
 * @tparam T Data type for the vector or matrix (e.g., float, double)
 * @param vector Pointer to the vector data
 * @param n  Number of elements in the vector
 */
template <typename T>
void InitVector(T *vector, const std::size_t &n);

/**
 * @brief Initialize a matrix with random values
 *
 * @tparam T Data type for the matrix (e.g., float, double)
 * @param matrix Pointer to the matrix data
 * @param rows  Number of rows in the matrix
 * @param cols  Number of columns in the matrix
 */
template <typename T>
void InitializeMatrix(T *matrix, const std::size_t &rows, const std::size_t &cols);

/**
 * @brief Verify the results of a vector operation
 *
 * @tparam T Data type for the vector (e.g., float, double)
 * @param d_y Pointer to the device output vector
 * @param h_y Pointer to the host expected output vector
 * @param n   Number of elements in the vector
 */
template <typename T>
void VerifyVector(const T *h_vector, const T *d_vector, const std::size_t &n);

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
void VerifyMatrix(const T *h_matrix, const T *d_matrix, const std::size_t &m, const std::size_t &n);

/**
 * @brief Verify the results of a collective operation
 *
 * @tparam T Data type for the collective operation (e.g., float, int)
 * @param label Label for the collective operation (e.g., "AllReduce", "Broadcast")
 * @param d_buff Pointer to the device output buffer
 * @param h_buff Reference host buffer containing expected results
 * @param rank Rank of the process in the collective operation
 * @param n Number of elements in the buffer
 */
template <typename T>
void VerifyCollective(const std::string &label, const T *d_buff, const std::vector<T> &h_buff, int rank, std::size_t n);

#endif