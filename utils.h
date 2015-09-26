#ifndef UTILS_H
#define UTILS_H
#include <mpi.h>

inline int block_decompose(const int n, const int p, const int rank) {
    return n / p + ((rank < n % p) ? 1 : 0);
}

inline int block_decompose(const int n, MPI_Comm comm) {
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    return block_decompose(n, p, rank);
}

inline int block_decompose_by_dim(const int n, MPI_Comm comm, int dim) {
    int dims[2];
    int periods[2];
    int coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);
    return block_decompose(n, dims[dim], coords[dim]);
}

void print_matrix( const int n, const int m, double* matrix );

double l2norm( const int n, double* vector , MPI_Comm comm );

void distributed_vector_sub( const int n, double* local_x, double* local_y, double* local_z, MPI_Comm comm );

void distributed_vector_membermult( const int n, double* local_x, double* local_y, double* local_z, MPI_Comm comm );

void setup_onetomany( const int p, const int datanum, const int datasize, int* displacements, int* sendsizes );

void reorganize_onetomany( const int p, const int row, const int col, int* sendsizes, double* from, double *to );

#endif
