#include"baseline_code_hls.h"

void matmultvec(DTYPE *res, int res_dim,
				DTYPE *mat, int mat_noRows, int mat_noColumns, 
				DTYPE *vec, int vec_dim) 
{
#pragma HLS INLINE
	LOOP_matmultvec1: for (int i=0; i<mat_noRows; i++) {
		res[i] = 0;
		LOOP_matmultvec2: for (int j=0; j<mat_noColumns; j++) { 
			res[i] += mat[i*mat_noColumns + j] * vec[i];
		}
	}
}

void addvecs(DTYPE *res, int res_dim,
			 DTYPE *a, int a_dim,
			 DTYPE *b, int b_dim) 
{
#pragma HLS INLINE
	LOOP_addvec: for (int i=0; i<res_dim; i++) {
#pragma HLS UNROLL
		res[i] = a[i] + b[i];
	}
}


void subvecs(DTYPE *res, int res_dim,
			 DTYPE *a, int a_dim,
			 DTYPE *b, int b_dim) 
{
#pragma HLS INLINE
	LOOP_subvec: for (int i=0; i<res_dim; i++) {
#pragma HLS UNROLL
		res[i] = a[i] - b[i];
	}
}



void amultbtrans(DTYPE *res, int res_noRows, int res_noColumns,
				 DTYPE *a, int a_noRows, int a_noColumns, 
				 DTYPE *b, int b_noRows, int b_noColumns) 
{
#pragma HLS INLINE
	LOOP_amultbtrans1: for (int i=0; i<res_noRows; i++) {
		LOOP_amultbtrans2: for (int j=0; j<res_noColumns; j++) {
			res[i*res_noColumns + j] = 0;
			LOOP_amultbtrans3: for (int k=0; k<b_noColumns; k++) {
				res[i*res_noColumns + j] += a[i*a_noColumns + k] * b[j*b_noRows + k];
			}
		}
	}
}



void amultb(DTYPE *res, int res_noRows, int res_noColumns,
			DTYPE *a, int a_noRows, int a_noColumns, 
			DTYPE *b, int b_noRows, int b_noColumns) 
{
#pragma HLS INLINE
	LOOP_amultb1: for (int i=0; i<res_noRows; i++) {
		LOOP_amultb2: for (int j=0; j<res_noColumns; j++) {
			res[i*res_noColumns + j] = 0;
			LOOP_amultb3: for (int k=0; k<b_noRows; k++) {
				res[i*res_noColumns + j] += a[i*a_noColumns + k] * b[k*b_noColumns + j];
			}
		}
	}

}



void addmats(DTYPE *res, int res_noRows, int res_noColumns,
			DTYPE *a, int a_noRows, int a_noColumns, 
			DTYPE *b, int b_noRows, int b_noColumns) 
{
#pragma HLS INLINE
	LOOP_addmat1: for (int i=0; i<res_noRows; i++) {
#pragma HLS UNROLL
		LOOP_addmat2: for (int j=0; j<res_noColumns; j++) {
#pragma HLS UNROLL
			res[i*res_noColumns + j] = a[i*a_noColumns + j] + b[i*b_noColumns + j];
		}
	}	
}



void submats(DTYPE *res, int res_noRows, int res_noColumns,
			DTYPE *a, int a_noRows, int a_noColumns, 
			DTYPE *b, int b_noRows, int b_noColumns) 
{
#pragma HLS INLINE
	LOOP_submat1: for (int i=0; i<res_noRows; i++) {
#pragma HLS UNROLL
		LOOP_submat2: for (int j=0; j<res_noColumns; j++) {
#pragma HLS UNROLL
			res[i*res_noColumns + j] = a[i*a_noColumns + j] - b[i*b_noColumns + j];
		}
	}
}


#define dimensions    MEASUREMENT_DIM

void matrix_initialization(DTYPE tempmat[dimensions][2*dimensions], DTYPE *a)
{
#pragma HLS INLINE
	int i, j;
	LOOP_copy1: for (i = 0; i < dimensions; i++) {
		LOOP_copy2: for (j = 0; j < 2 * dimensions; j++) {
#pragma HLS UNROLL
			if (j < dimensions)
				tempmat[i][j] = a[i*dimensions + j];
            if (j == (i + dimensions))
                tempmat[i][j] = 1;
        }
    }
}

/*
 * Note: Inlining and dataflow must be checked - Not sure which is good here. Prefer dataflow and no-inline - reduces latency
 * by a couple of cycles since normalization_copyback latency is much less than elimination_loops latencies. But does it
 * overcome the function overhead? Not sure
 */
void normalization_copyback(DTYPE tempmat[dimensions][2*dimensions], DTYPE *res)
{
	DTYPE inversediagonal_elements[dimensions];
#pragma HLS ARRAY_PARTITION variable=inversediagonal_elements complete dim=0
#pragma HLS ARRAY_PARTITION variable=tempmat complete dim=0
#pragma HLS DATAFLOW

	int m;
    LOOP_inversediagonal: for(m = 0; m < dimensions; m++)
    {
#pragma HLS PIPELINE II=1
    	inversediagonal_elements[m] = 1/tempmat[m][m];
    }

    int n,p;
    LOOP_normalization1: for (n = 0; n < dimensions; n++) {
#pragma HLS PIPELINE II=1
    	LOOP_normalization2: for (p = 0; p < dimensions; p++) {
#pragma HLS UNROLL
    		res[n*dimensions + p] = tempmat[n][p + dimensions] * inversediagonal_elements[n];
        }
    }
}


void inversemat(DTYPE *res,
				DTYPE *a)
{
#pragma HLS INLINE
	int i,j;
	DTYPE tempmat[dimensions][2*dimensions] = {0};
	#pragma HLS ARRAY_PARTITION variable=tempmat complete dim=0
	matrix_initialization(tempmat, a);

	int x, y, z;
	float temp;

    LOOP_elimination1: for (x = 0; x < dimensions; x++) {
    	LOOP_elimination2: for (y = 0; y < dimensions; y++) {
#pragma HLS DEPENDENCE variable=tempmat type=intra dependent=true
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=7
			if (x != y) {
				if (tempmat[x][x]==0)
				{
					LOOP_sumrows1: for (int i = 0; i < dimensions; i++){
						LOOP_sumrows2: for(int j = 0; j < 2*dimensions; j++){
							if(i > x+1)
								tempmat[x][j] = tempmat[x][j] + tempmat[i][j];
						}
					}
				}
                temp = tempmat[y][x] / tempmat[x][x];
                LOOP_elimination3: for (z = 0; z < 2 * dimensions; z++) {
					#pragma HLS UNROLL
					tempmat[y][z] = tempmat[y][z] - tempmat[x][z] * temp;
                }
            }
        }
    }

    normalization_copyback(tempmat, res);
}

#define xk_dim       STATE_DIM
#define uk_dim       INPUT_DIM
#define F_noRows     STATE_DIM
#define F_noColumns  STATE_DIM
#define B_noRows     STATE_DIM
#define B_noColumns  INPUT_DIM
#define Pk_noRows    STATE_DIM
#define Pk_noColumns STATE_DIM
#define Q_noRows     STATE_DIM
#define Q_noColumns  STATE_DIM
#define yk_dim       MEASUREMENT_DIM
#define zk_dim       MEASUREMENT_DIM
#define H_noRows     MEASUREMENT_DIM
#define H_noColumns  STATE_DIM
#define Kk_noRows    STATE_DIM
#define Kk_noColumns MEASUREMENT_DIM
#define R_noRows     MEASUREMENT_DIM
#define R_noColumns  MEASUREMENT_DIM


#define M_dim xk_dim
#define N_dim xk_dim
void statePredictor(DTYPE *xk, DTYPE *uk, DTYPE *F,	DTYPE *B)
{
	DTYPE M[M_dim], N[N_dim];
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=M complete dim=0
#pragma HLS ARRAY_PARTITION variable=N complete dim=0

	matmultvec(M, M_dim, 
			   F, F_noRows, F_noColumns,
			   xk, xk_dim);
	
	matmultvec(N, N_dim, 
			   B, B_noRows, B_noColumns,
			   uk, uk_dim);
	
	addvecs(xk, xk_dim,
			M, M_dim, 
			N, N_dim);		
}



#define L1_noRows    Pk_noRows
#define L1_noColumns F_noRows
#define L2_noRows    F_noRows
#define L2_noColumns L1_noColumns
void covariancePredictor(DTYPE *Pk, DTYPE *F, DTYPE *Q)
{
#pragma HLS INLINE
	DTYPE L1[L1_noRows * L1_noColumns], L2[L2_noRows * L2_noColumns];
#pragma HLS ARRAY_PARTITION variable=L1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=L2 complete dim=0
	
	amultbtrans(L1, L1_noRows, L1_noColumns,
				Pk, Pk_noRows, Pk_noColumns,
				F,  F_noRows,  F_noColumns);
	
	amultb(L2, L2_noRows, L2_noColumns,
		   F,  F_noRows,  F_noColumns,
		   L1, L1_noRows, L1_noColumns);
	
	addmats(Pk, Pk_noRows, Pk_noColumns,
			L2, L2_noRows, L2_noColumns,
		    Q,  Q_noRows,  Q_noColumns);		
}


#define E_dim yk_dim
void measurementResidual(DTYPE *zk, DTYPE *H, DTYPE *xk, DTYPE *yk)
{
#pragma HLS INLINE
	DTYPE E[E_dim];
#pragma HLS ARRAY_PARTITION variable=E complete dim=0

	matmultvec(E,  E_dim,
			   H,  H_noRows, H_noColumns, 
			   xk, xk_dim);

	subvecs(yk, yk_dim, 
	        zk, zk_dim,
			E,  E_dim);
}


#define A_noRows      Pk_noRows
#define A_noColumns   H_noRows
#define C1_noRows     H_noRows
#define C1_noColumns  A_noColumns

void kalmangainCalculator(DTYPE *Pk, DTYPE *H, 	DTYPE *R, DTYPE *Kk)
{
#pragma HLS INLINE
	DTYPE A[A_noRows * A_noColumns], C1[C1_noRows * C1_noColumns];
#pragma HLS ARRAY_PARTITION variable=A complete dim=0
#pragma HLS ARRAY_PARTITION variable=C1 complete dim=0

	amultbtrans(A,  A_noRows,  A_noColumns,
	            Pk, Pk_noRows, Pk_noColumns,
				H,  H_noRows,  H_noColumns);

	amultb(C1, C1_noRows, C1_noColumns,
	       H,  H_noRows,  H_noColumns,
		   A,  A_noRows,  A_noColumns);

	addmats(C1, C1_noRows, C1_noColumns, 
	        R,  R_noRows,  R_noColumns,
			C1, C1_noRows, C1_noColumns);
	
	inversemat(C1,
	           C1);

	amultb(Kk, Kk_noRows, Kk_noColumns,
	       A,  A_noRows,  A_noColumns,
	       C1, C1_noRows, C1_noColumns);		
}


#define temp_dim xk_dim
void stateUpdate(DTYPE *xk, DTYPE *Kk, DTYPE *yk)
{
#pragma HLS INLINE
	DTYPE temp[temp_dim];
#pragma HLS ARRAY_PARTITION variable=temp complete dim=0
	
	matmultvec(temp, temp_dim,
			   Kk,   Kk_noRows, Kk_noColumns,
			   yk,   yk_dim);

	addvecs(xk,   xk_dim,
	 		xk,   xk_dim,
			temp, temp_dim);
}

#define temp1_noRows    Kk_noRows
#define temp1_noColumns H_noColumns
#define temp2_noRows    temp1_noRows
#define temp2_noColumns Pk_noColumns
void covarianceUpdate(DTYPE *Kk, DTYPE *H, DTYPE *Pk)
{
#pragma HLS INLINE
	DTYPE temp1[temp1_noRows * temp1_noColumns], temp2[temp2_noRows * temp2_noColumns];

#pragma HLS ARRAY_PARTITION variable=temp1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=temp2 complete dim=0


	amultb(temp1, temp1_noRows, temp1_noColumns, 
		   Kk,    Kk_noRows,    Kk_noColumns,
		   H,     H_noRows,     H_noColumns);

	amultb(temp2, temp2_noRows, temp2_noColumns, 
		   temp1, temp1_noRows, temp1_noColumns, 
		   Pk,     Pk_noRows,   Pk_noColumns);

	submats(Pk,     Pk_noRows,    Pk_noColumns,
	        Pk,     Pk_noRows,    Pk_noColumns, 
			temp2,  temp2_noRows, temp2_noColumns);
}


void kalmanIterate(DTYPE *xk, DTYPE *uk, DTYPE *F, DTYPE *B, DTYPE *Pk, DTYPE *Q, DTYPE *yk, DTYPE *zk, DTYPE *H, DTYPE *Kk, DTYPE *R)
{

	statePredictor(xk,	uk,	F, B);
	covariancePredictor(Pk, F, Q);
    measurementResidual(zk, H, xk, yk);
    kalmangainCalculator(Pk, H, R, Kk);
	stateUpdate(xk, Kk, yk);
	covarianceUpdate(Kk, H, Pk);
}

