/**
 * @file mpiUtils.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief A wrapper module for the MPI library, providing a simplified interface for common MPI operations.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <mpi.h>
#include <stdio.h>

#include <cstdlib>
#include <unordered_map>
#include <vector>

#define COMM_WORLD MpiComm(MPI_COMM_WORLD)
#define MPICHECK(cmd)                                                        \
    do {                                                                     \
        int e = cmd;                                                         \
        if (e != MPI_SUCCESS) {                                              \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// A wrapper module of the MPI library.
namespace tensorrt_llm::mpi {

// A wrapper of MPI data type. MPI_TYPE_{data_type}
enum MpiType {
    MPI_TYPE_BYTE,
    MPI_TYPE_CHAR,
    MPI_TYPE_INT,
    MPI_TYPE_INT64_T,
    MPI_TYPE_UINT32_T,
    MPI_TYPE_UNSIGNED_LONG_LONG,
};

// A wrapper of MPI_Op type.
enum MpiOp {
    MPI_OP_NULLOP,
    MPI_OP_MAX,
    MPI_OP_MIN,
    MPI_OP_SUM,
    MPI_OP_PROD,
    MPI_OP_LAND,
    MPI_OP_BAND,
    MPI_OP_LOR,
    MPI_OP_BOR,
    MPI_OP_LXOR,
    MPI_OP_BXOR,
    MPI_OP_MINLOC,
    MPI_OP_MAXLOC,
    MPI_OP_REPLACE,
};

// A wrapper of the level of MPI thread support
enum MpiThreadSupport { THREAD_SINGLE, THREAD_FUNNELED, THREAD_SERIALIZED, THREAD_MULTIPLE };

struct MpiComm {
    MPI_Comm group;
    MpiComm() {};
    MpiComm(MPI_Comm g) : group(g) {};
};

MPI_Datatype getMpiDtype(MpiType dtype);

void initialize(int* argc, char*** argv);
void initThread(int* argc, char*** argv, MpiThreadSupport required, int* provided);
void finalize();
bool isInitialized();
void barrier(MpiComm comm);
void barrier();

int getCommWorldRank();
int getCommWorldSize();

void bcast(void* buffer, size_t size, MpiType dtype, int root, MpiComm comm);
void bcast(std::vector<int64_t>& packed, int root, MpiComm comm);
void comm_split(MpiComm comm, int color, int key, MpiComm* newcomm);
void allreduce(const void* sendbuf, void* recvbuf, int count, MpiType dtype, MpiOp op, MpiComm comm);
void allgather(const void* sendbuf, void* recvbuf, int count, MpiType dtype, MpiComm comm);

}  // namespace tensorrt_llm::mpi
