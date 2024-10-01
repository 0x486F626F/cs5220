const char* dgemm_desc = "My awesome dgemm 512.";
#include <immintrin.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define KERNEL_SIZE 8

void square_dgemm(const int N, double * AA, double * BB, double * CC)
{
    int bi, bj, bk, i, j, k;
    int x, y, z, xx, yy, zz;
    __declspec(align(64)) double tmp[KERNEL_SIZE];
    __declspec(align(64)) __m512d a[KERNEL_SIZE];

#ifdef CP_BLOCK
    int M = (N+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE;
#else 
    int M = (N+KERNEL_SIZE-1)/KERNEL_SIZE * KERNEL_SIZE;//BLOCK_SIZE*BLOCK_SIZE;
#endif

    double * A = AA;
    if (M != N && (size_t)AA % 64 != 0) {
        A = (double*) _mm_malloc(M*M*sizeof(double), 64);
        memset(A, 0, M*M*sizeof(double));
        for (i = 0; i < N; i ++) 
            memcpy(A+i*M, AA+i*N, sizeof(double)*N);
    }
    double * B = BB;
    if (M != N && (size_t)BB % 64 != 0) {
        B = (double*) _mm_malloc(M*M*sizeof(double), 64);
        memset(B, 0, M*M*sizeof(double));
        for (i = 0; i < N; i ++) 
            memcpy(B+i*M, BB+i*N, sizeof(double)*N);
    }
    double * C = CC;
    if (M != N && (size_t)AA % 64 != 0) {
        C = (double*) _mm_malloc(M*M*sizeof(double), 64);
        memset(C, 0, M*M*sizeof(double));
    }

    #pragma vector aligned
    for (bk = 0; bk < M; bk += BLOCK_SIZE)  
        for (bi = 0; bi < M; bi += BLOCK_SIZE)
            for (bj = 0; bj < M; bj += BLOCK_SIZE) 
            {
#ifdef CP_BLOCK
                for (z = 0; z < BLOCK_SIZE; z += KERNEL_SIZE) 
                    for (x = 0; x < BLOCK_SIZE; x += KERNEL_SIZE)
                        for (y = 0; y < BLOCK_SIZE; y += KERNEL_SIZE)

#else
                int X = (bi+BLOCK_SIZE > M ? M-bi : BLOCK_SIZE);
                int Y = (bj+BLOCK_SIZE > M ? M-bj : BLOCK_SIZE);
                int Z = (bk+BLOCK_SIZE > M ? M-bk : BLOCK_SIZE);
                for (x = 0; x < X; x += KERNEL_SIZE)
                    for (z = 0; z < Z; z += KERNEL_SIZE) 
                        for (y = 0; y < Y; y += KERNEL_SIZE)
#endif
                        {
                            int base_a = (bk+z)*M+bj+y;
                            int base_b = (bi+x)*M+bk+z;
                            int base_c = (bi+x)*M+bj+y;

                            for (k = 0; k < KERNEL_SIZE; k++)
                                a[k] = _mm512_load_pd(A+base_a+k*M);

                            for (i = 0; i < KERNEL_SIZE; i++) {
                                __m512d c = _mm512_setzero_pd();
                                for (k = 0; k < KERNEL_SIZE; k++)
                                    c = _mm512_fmadd_pd(_mm512_set1_pd(B[base_b+i*M+k]), a[k], c);
                                _mm512_store_pd(tmp, c);
                                for (j = 0; j < KERNEL_SIZE; j ++) 
                                    C[base_c+i*M+j] += tmp[j];
                            }
                        }
            }


    if (A != AA) _mm_free(A);
    if (B != BB) _mm_free(B);
    if (C != CC) {
        for (i = 0; i < N; i ++) 
            memcpy(CC+i*N, C+i*M, sizeof(double)*N);
        _mm_free(C);
    }
}
