#include "utils.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>

void print_matrix( const int n, const int m, double* matrix ){
	for ( int i = 0; i < n; i++ ){
		for ( int j = 0; j < m; j++ ){
			std::cout << matrix[i*m + j] << ' ';
		}
		std::cout << std::endl;
	}
}

double l2norm( const int n, double* local_vector, MPI_Comm comm ){
	// Stats
	int rank, row, coords[2];
	double sum;
	// Get local and cartesian ranks
	MPI_Comm_rank( comm, &rank );
	MPI_Cart_coords( comm, rank, 2, coords );

	// Construct communicator
	int remdims1[2] = { 1, 0 };
	MPI_Comm col_comm;
	MPI_Cart_sub( comm, remdims1, &col_comm );

	// Calculate L2 norm
	if ( coords[1] == 0 ){
		row = block_decompose_by_dim( n, comm, 0 );

		// Square members
		double local_sum = 0.0;
		for ( int i = 0; i < row; i++ )
			local_sum += local_vector[i] * local_vector[i];

		MPI_Allreduce( &local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, col_comm );
		sum = sqrt( sum );
	}

	MPI_Comm_free( &col_comm );

	// Construct communicator
	int remdims2[2] = { 0, 1 };
	MPI_Comm row_comm;
	MPI_Cart_sub( comm, remdims2, &row_comm );

	MPI_Bcast( &sum, 1, MPI_DOUBLE, 0, row_comm );

	MPI_Comm_free( &row_comm );

	return sum;
}

void distributed_vector_sub( const int n, double* local_x, double* local_y, double* local_z, MPI_Comm comm ){
	// Stats
	int rank, row, coords[2];
	// Get local and cartesian ranks
	MPI_Comm_rank( comm, &rank );
	MPI_Cart_coords( comm, rank, 2, coords );

	if ( coords[1] == 0 ){
		row = block_decompose_by_dim( n, comm, 0 );
		for ( int i = 0; i < row; i++ ){
			local_z[i] = local_x[i] - local_y[i];
		}
	}
	MPI_Barrier( comm );
}

void distributed_vector_membermult( const int n, double* local_x, double* local_y, double* local_z, MPI_Comm comm ){
	// Stats
	int rank, row, coords[2];
	// Get local and cartesian ranks
	MPI_Comm_rank( comm, &rank );
	MPI_Cart_coords( comm, rank, 2, coords );

	if ( coords[1] == 0 ){
		row = block_decompose_by_dim( n, comm, 0 );
		for ( int i = 0; i < row; i++ ){
			local_z[i] = local_x[i] * local_y[i];
		}
	}
	MPI_Barrier( comm );
}

void setup_onetomany( const int p, const int datanum, const int datasize, int* displacements, int* sendsizes ){
	int sofar = 0;
	for ( int i = 0; i < p; i++ ){
		int tosend = datasize * ( datanum / p + ( ( i < datanum % p ) ? 1 : 0 ) );
		displacements[i] = sofar;
		sendsizes[i] = tosend;
		sofar += tosend;
	}
}

void reorganize_onetomany( const int p, const int row, const int col, int* sendsizes, double* from, double *to ){
	int fromiter = 0;
	int toiter = 0;
	for ( int i = 0; i < p; i++ ){
		for ( int j = fromiter; j < row*col; j += col ){
			for ( int k = 0; k < sendsizes[i]/row; k++ ){
				to[toiter++] = from[j+k];
			}
		}
		fromiter += sendsizes[i]/row;
	}
}
