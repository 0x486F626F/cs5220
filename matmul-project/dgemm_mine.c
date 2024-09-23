const char* dgemm_desc = "My awesome dgemm.";
#include <immintrin.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

void avx_kernel_dgemm(const double *A, const double *B, double *C)
{
    __m256d c0 = _mm256_setzero_pd();
    __m256d c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd();
    __m256d c3 = _mm256_setzero_pd();

    for (int i = 0; i < 4; i ++) {
        __m256d a0 = _mm256_set1_pd(A[i + 4 * 0]);
        __m256d a1 = _mm256_set1_pd(A[i + 4 * 1]);
        __m256d a2 = _mm256_set1_pd(A[i + 4 * 2]);
        __m256d a3 = _mm256_set1_pd(A[i + 4 * 3]);

        __m256d b = _mm256_loadu_pd(B + i * 4);

        c0 = _mm256_fmadd_pd(a0, b, c0);
        c1 = _mm256_fmadd_pd(a1, b, c1);
        c2 = _mm256_fmadd_pd(a2, b, c2);
        c3 = _mm256_fmadd_pd(a3, b, c3);
    }

    _mm256_storeu_pd(C, c0);
    _mm256_storeu_pd(C+4, c1);
    _mm256_storeu_pd(C+8, c2);
    _mm256_storeu_pd(C+12, c3);
}

void kernel_dgemm(const int N, const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void square_dgemm_main(const int M, const double *A, const double *B, double *C)
{
    const int N = 4;
    int i_block, j_block, k_block;
    int i, j, k;

    // Allocate memory for temporary blocks
    double BA[N * N];
    double BB[N * N];
    double BC[N * N];

    // Block matrix multiplication
    for (i_block = 0; i_block < M; i_block += N) {
        for (j_block = 0; j_block < M; j_block += N) {
            for (k_block = 0; k_block < M; k_block += N) {
                for (i = 0; i < N * N; i++) {
                    BA[i] = 0.0;
                    BB[i] = 0.0;
                    BC[i] = 0.0;
                }

                for (i = 0; i < N && i_block+i < M; i++) {
                    int len = k_block + N < M? N : M-k_block;
                    memcpy(BB + i*N, B + (i_block+i)*M+k_block, len * sizeof(double));
                }

                for (k = 0; k < N && k_block+k < M; k++) {
                    int len = j_block + N < M? N : M-j_block;
                    memcpy(BA + k*N, A + (k_block+k)*M+j_block, len * sizeof(double));
                }

                //kernel_dgemm(N, BB, BA, BC);
                avx_kernel_dgemm(BB, BA, BC);

                // Accumulate the result into C
                for (i = 0; i < N; i++) {
                    for (j = 0; j < N; j++) {
                        if (i_block + i < M && j_block + j < M) {
                            C[(i_block + i) * M + (j_block + j)] += BC[i * N + j];
                        }
                    }
                }
            }
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int N = 4;
    int i_block, j_block, k_block;
    int i, j, k;

    // Allocate memory for temporary blocks
    double BB[N * N];
    //double BC[N * N];

    double tmp[4];
    //__m256d a[4];
    // Block matrix multiplication
    for (i_block = 0; i_block < M; i_block += N) {
        for (j_block = 0; j_block < M; j_block += N) {
            for (k_block = 0; k_block < M; k_block += N) {

                //__m256d b[16];
                __m256d c[4] = {
                    _mm256_setzero_pd(),
                    _mm256_setzero_pd(),
                    _mm256_setzero_pd(),
                    _mm256_setzero_pd(),
                };

                for (i = 0; i < N * N; i++) BB[i] = 0.0;
                for (i = 0; i < N && i_block+i < M; i++) {
                    int len = k_block + N < M? N : M-k_block;
                    memcpy(BB + i*N, B + (i_block+i)*M+k_block, len * sizeof(double));
                    /*
                       for (k = 0; k < N && k_block+k < M; k++) 
                       b[i*N+k] = _mm256_set1_pd(B[(i_block+i)*M+k_block+k]);
                       */
                }

                __m256d a;
                for (k = 0; k < N && k_block+k < M; k++) {
                    int len = j_block + N < M? N : M-j_block;
                    int base = (k_block+k)*M+j_block;
                    if (len == N) {
                        a = _mm256_loadu_pd(A + base);
                    } else {
                        tmp[0] = A[base];
                        tmp[1] = len > 1 ? A[base+1] : 0;
                        tmp[2] = len > 2 ? A[base+2] : 0;
                        a = _mm256_set_pd(0, tmp[2], tmp[1], tmp[0]);
                    }

                    for (i = 0; i < N && i_block+i < M; i++) {
                        __m256d b = _mm256_set1_pd(BB[i * N + k]);
                        c[i] = _mm256_fmadd_pd(b, a, c[i]);
                        //c[i] = _mm256_fmadd_pd(b[i*N+k], a[k], c[i]);
                    }
                }

                for (i = 0; i < N && i_block+i<M; i++) {
                    _mm256_storeu_pd(tmp, c[i]);
                    for (j = 0; j < N; j++) 
                        if (j_block + j < M) 
                            C[(i_block + i) * M + (j_block + j)] += tmp[j];
                }

            }
        }
    }
}


