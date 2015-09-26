#include "jacobi.h"
#include <vector>
#include <iostream>
#include <cmath>

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
    matrix_vector_mult( n, n, A, x, y );
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
    int i,j;
    for (i = 0 ; i < n ; i++) {
        y[i] = 0;
        for (j = 0 ; j < m ; j++) {
            y[i] = y[i] + A[i*m + j] * x[j];
        }
    }
}

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
    // Remove diagonal from matrix A
    std::vector<double> invD(n,0), R(n*n,0), temp(n,0);
    int i,j, itr = 0;
    for (i = 0 ; i < n ; i++) {
        for (j = 0 ; j < n ; j++) {
            if (i == j) {
                invD[i] = 1.0 / A[i*n + j];
            } else {
                R[i*n + j] = A[i*n + j];
            }
        }
        x[i] = 0;
    }
    
    // iteratively improve x
    double norm;
    for ( itr = 0; itr < max_iter; itr++ ){
                
        norm = 0;
        matrix_vector_mult(n, A, x, &temp[0]);
        for (i = 0 ; i < n ; i++) {
            norm += (b[i] - temp[i])*(b[i] - temp[i]);
        }
        norm = sqrt(norm);
        //std::cout << "b-Ax=" << temp[i] << ", iteration: " << itr+1 << std::endl;
                
        if (norm <= l2_termination) {
            break;
        }        
        
        // std::cout << "norm: " << norm << ", iteration: " << itr << std::endl;

        matrix_vector_mult(n, &R[0], x, &temp[0]);
        for (i = 0 ; i < n ; i++) {
            x[i] = invD[i] * (b[i] - temp[i]);
        }
    }
}
