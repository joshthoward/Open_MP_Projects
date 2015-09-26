#ifndef JACOBI_H
#define JACOBI_H

void matrix_vector_mult(const int n, const double* A,
                        const double* x, double* y);

void matrix_vector_mult(const int n, const int m, const double* A,
                        const double* x, double* y);

void jacobi(const int n, double* A, double* b, double* x,
            int max_iter = 100, double l2_termination = 1e-10);

#endif
