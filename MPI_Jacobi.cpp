#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <iostream>

void distribute_vector( const int n, double* input_vector, double** local_vector, MPI_Comm comm )
{
	int coords[2], rank;
	int row = block_decompose_by_dim( n, comm, 0 );
	*local_vector = new double[row];

	// Get local and cartesian ranks
	MPI_Comm_rank( comm, &rank );
	MPI_Cart_coords( comm, rank, 2, coords );

	// Construct communicator
	MPI_Comm col_comm;
	int remdims[2] = { 1, 0 };
	MPI_Cart_sub( comm, remdims, &col_comm );

	if ( coords[1] == 0 ){
		int size;
		MPI_Comm_size( col_comm, &size );

		std::vector<int> displacements;
		std::vector<int> sendsizes;
		if ( coords[0] == 0 )
		{
			displacements.resize( size );
			sendsizes.resize( size );
			int size;
			MPI_Comm_size( col_comm, &size );
			setup_onetomany( size, n, 1, &displacements[0], &sendsizes[0] );
		}

		// Scatter
		MPI_Scatterv( input_vector, &sendsizes[0], &displacements[0], MPI_DOUBLE, *local_vector, row, MPI_DOUBLE, 0, col_comm );
	}

	MPI_Barrier( comm );
	MPI_Comm_free( &col_comm );
}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector( const int n, double* local_vector, double* output_vector, MPI_Comm comm )
{
	int coords[2], rank;
	int row = block_decompose_by_dim( n, comm, 0 );

	// Clear result vector
	if ( coords[1] == 0 )
		for ( int i = 0; i < row; i++ )
			output_vector[i] = 0.0;

	// Get local and cartesian ranks
	MPI_Comm_rank( comm, &rank );
	MPI_Cart_coords( comm, rank, 2, coords );

	// Construct communicator
	MPI_Comm col_comm;
	int remdims[2] = { 1, 0 };
	MPI_Cart_sub( comm, remdims, &col_comm );

	if ( coords[1] == 0 ){
		int size;
		MPI_Comm_size( col_comm, &size );

		std::vector<int> displacements;
		std::vector<int> sendsizes;
		if ( coords[0] == 0 )
		{
			displacements.resize( size );
			sendsizes.resize( size );
			int size;
			MPI_Comm_size( col_comm, &size );
			setup_onetomany( size, n, 1, &displacements[0], &sendsizes[0] );
		}

		// Gather
		MPI_Gatherv( local_vector, row, MPI_DOUBLE, output_vector, &sendsizes[0], &displacements[0], MPI_DOUBLE, 0, col_comm );
		MPI_Barrier( col_comm );
	}

	MPI_Barrier( comm );
	MPI_Comm_free( &col_comm );
}

void distribute_matrix( const int n, double* input_matrix, double** local_matrix, MPI_Comm comm )
{
	// Stats
	int rank, col, row, coords[2];
	// Get local and cartesian ranks
	MPI_Comm_rank( comm, &rank );
	MPI_Cart_coords( comm, rank, 2, coords );
	row = block_decompose_by_dim( n, comm, 0 );
	col = block_decompose_by_dim( n, comm, 1 );

	// Construct communicators
	int remdims1[2] = { 1, 0 };
	MPI_Comm col_comm;
	MPI_Cart_sub( comm, remdims1, &col_comm );
	MPI_Comm row_comm;
	int remdims2[2] = { 0, 1 };
	MPI_Cart_sub( comm, remdims2, &row_comm );

	std::vector<double> temp;
	if ( coords[1] == 0 )
	{
		std::vector<int> displacements;
		std::vector<int> sendsizes;
		if ( coords[0] == 0 )
		{
			int size;
			MPI_Comm_size( col_comm, &size );
			displacements.resize( size );
			sendsizes.resize( size );
			setup_onetomany( size, n, n, &displacements[0], &sendsizes[0] );
		}

		// Scatter among rows
		temp.resize( n*row );
		MPI_Scatterv( input_matrix, &sendsizes[0], &displacements[0], MPI_DOUBLE, &temp[0], n*row, MPI_DOUBLE, 0, col_comm );
		MPI_Barrier( col_comm );
	}
	MPI_Comm_free( &col_comm );
	MPI_Barrier( comm );

	std::vector<int> displacements;
	std::vector<int> sendsizes;
	if ( coords[1] == 0 )
	{
		int size;
		MPI_Comm_size( row_comm, &size );
		displacements.resize( size );
		sendsizes.resize( size );
		setup_onetomany( size, n, row, &displacements[0], &sendsizes[0] );
		std::vector<double> temp2 = temp;
		reorganize_onetomany( size, row, n, &sendsizes[0], &temp2[0], &temp[0] );
	}

	// Scatter among columns
	*local_matrix = new double[row*col];
	MPI_Scatterv( &temp[0], &sendsizes[0], &displacements[0], MPI_DOUBLE, *local_matrix, row*col, MPI_DOUBLE, 0, row_comm );
	MPI_Comm_free( &row_comm );
}


void transpose_bcast_vector( const int n, double* col_vector, double* row_vector, MPI_Comm comm )
{
	int rank, send_rank, recv_rank, coords[2];
	MPI_Status status;

	// Get local and cartesian ranks
	MPI_Comm_rank( comm, &rank );
	MPI_Cart_coords( comm, rank, 2, coords );

	// Determine recv and send ranks
	int recvcoords[2] = { coords[0], coords[0] };
	MPI_Cart_rank( comm, recvcoords, &recv_rank );
	int sendcoords[2] = { coords[0], 0 };
	MPI_Cart_rank( comm, sendcoords, &send_rank );

	// Determine data size 
	int row = block_decompose_by_dim( n, comm, 0 );
	int col = block_decompose_by_dim( n, comm, 1 );

	// Construct communicator
	MPI_Comm col_comm;
	int remdims[2] = { 1, 0 };
	MPI_Cart_sub( comm, remdims, &col_comm );

	// Send to diagonal
	if ( coords[1] == 0 ) {
		for ( int i = 0; i < row; i++ )
			row_vector[i] = col_vector[i];
		if ( coords[0] != 0 || coords[1] != 0 )
			MPI_Send( row_vector, row, MPI_DOUBLE, recv_rank, 100, comm );
	}
	else if ( coords[0] == coords[1] ) {
		// Receive
		if ( coords[0] != 0 || coords[1] != 0 )
			MPI_Recv( row_vector, col, MPI_DOUBLE, send_rank, 100, comm, &status );
	}
	// Scatter along column
	MPI_Bcast( row_vector, col, MPI_DOUBLE, coords[1], col_comm );

	MPI_Barrier( comm );
	MPI_Comm_free( &col_comm );
}


void distributed_matrix_vector_mult( const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm )
{
    // Stats
	int rank, col, row, coords[2];
	// Get local and cartesian ranks
	MPI_Comm_rank( comm, &rank );
	MPI_Cart_coords( comm, rank, 2, coords );

	row = block_decompose_by_dim( n, comm, 0 );
	col = block_decompose_by_dim( n, comm, 1 );

	// Clear result vector
	if ( coords[1] == 0 )
		for ( int i = 0; i < row; i++ )
			local_y[i] = 0.0;

	// Distribute vector
	std::vector<double> transX( col );
	transpose_bcast_vector( n, local_x, &transX[0], comm );

	// Local multiplication
	std::vector<double> Ax( row );
	matrix_vector_mult( row, col, local_A, &transX[0], &Ax[0] );

	// Construct communicator
	int remdims[2] = { 0, 1 };
	MPI_Comm row_comm;
	MPI_Cart_sub( comm, remdims, &row_comm );

	// Reduction
	MPI_Reduce( &Ax[0], local_y, row, MPI_DOUBLE, MPI_SUM, 0, row_comm );

	MPI_Barrier( comm );
	MPI_Comm_free( &row_comm );
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi( const int n, double* local_A, double* local_b, double* local_x,
	MPI_Comm comm, int max_iter, double l2_termination )
{
	// Broadcast info
	MPI_Bcast( &l2_termination, 1, MPI_DOUBLE, 0, comm );
	MPI_Bcast( &max_iter, 1, MPI_INT, 0, comm );
	MPI_Barrier( comm );

	// Stats
	int rank, col, row, coords[2];
	// Get local and cartesian ranks
	MPI_Comm_rank( comm, &rank );
	MPI_Cart_coords( comm, rank, 2, coords );

	// R
	row = block_decompose_by_dim( n, comm, 0 );
	col = block_decompose_by_dim( n, comm, 1 );
	std::vector<double> invD;
	std::vector<double> R( row*col, 0.0 );
	for ( int i = 0; i < row; i++ ){
		for ( int j = 0; j < col; j++ ){
			R[i*col + j] = local_A[i*col + j];
		}
	}

	// Send diagonal to first column
	if ( coords[0] == coords[1] ){
		invD.resize( row, 0.0 );
		for ( int i = 0; i < row; i++ ){
			invD[i] = 1.0 / local_A[i*col + i];
			R[i*col + i] = 0.0;
		}
		int targetrank;
		int targetcoords[2] = { coords[0], 0 };
		MPI_Cart_rank( comm, targetcoords, &targetrank );
		if ( coords[0] != 0 )
			MPI_Send( &invD[0], row, MPI_DOUBLE, targetrank, 1, comm );
	}
	else if ( coords[1] == 0 ){
		invD.resize( row, 0.0 );
		MPI_Status status;
		int sourcerank;
		int srccoords[2] = { coords[0], coords[0] };
		MPI_Cart_rank( comm, srccoords, &sourcerank );
		MPI_Recv( &invD[0], row, MPI_DOUBLE, sourcerank, 1, comm, &status );
	}

	// Initialize local_x
	if ( coords[1] == 0 ){
		for ( int i = 0; i < row; i++ ){
			local_x[i] = 0.0;
		}
	}
	MPI_Barrier( comm );

	// Allocate data structures
	std::vector<double> Ax;
	if ( coords[1] == 0 )
		Ax.resize( row, 0.0 );
	std::vector<double> Axb;
	if ( coords[1] == 0 )
		Axb.resize( row, 0.0 );
	std::vector<double> Rx;
	if ( coords[1] == 0 )
		Rx.resize( row, 0.0 );
	std::vector<double> bRx;
	if ( coords[1] == 0 )
		bRx.resize( row, 0.0 );

	// Iteration
	for ( int i = 0; i < max_iter; i++ ){
		// Calculate Ax - b
		distributed_matrix_vector_mult( n, local_A, local_x, &Ax[0], comm );
		distributed_vector_sub( n, &Ax[0], local_b, &Axb[0], comm );

		// Check for termination
		if ( l2norm( n, &Axb[0], comm ) <= l2_termination )
			break;

		// Rx
		distributed_matrix_vector_mult( n, &R[0], local_x, &Rx[0], comm );

		// b - Rx
		distributed_vector_sub( n, local_b, &Rx[0], &bRx[0], comm );

		// D^-1 * (b - Rx)
		distributed_vector_membermult( n, &invD[0], &bRx[0], local_x, comm );

		MPI_Barrier( comm );
	}
}

// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_b = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);

    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}
