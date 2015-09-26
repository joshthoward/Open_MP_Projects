#ifndef MPI_JACOBI_H
#define MPI_JACOBI_H
#include <mpi.h>

void distribute_vector(const int n, double* input_vector, double** local_vector,
                       MPI_Comm comm);

void gather_vector(const int n, double* local_vector, double* output_vector,
                   MPI_Comm comm);

void distribute_matrix(const int n, double* input_matrix, double** local_matrix,
                       MPI_Comm comm);

void transpose_bcast_vector(const int n, double* col_vector, double* row_vector,
                            MPI_Comm comm);

void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x,
                                    double* local_y,MPI_Comm comm);

void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                        MPI_Comm comm, int max_iter = 100,double l2_termination = 1e-10);

void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm);

void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter = 100, double l2_termination = 1e-10);

#endif
