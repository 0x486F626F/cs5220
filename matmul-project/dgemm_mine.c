const char* dgemm_desc = "My awesome dgemm 512.";
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

void square_dgemm_avx256(const int M, const double *A, const double *B, double *C)
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

void square_dgemm_avx512(const int x, const int y, const int z, const int N,
        const int M, const double *A, const double *B, double *C)
{
    int i_block, j_block, k_block;
    int i, j, k;

    // Allocate memory for temporary blocks
    //double BB[N * N];

    double tmp[8];
    // Block matrix multiplication
    for (i_block = x; i_block < N; i_block += 8) {
        for (j_block = y; j_block < N; j_block += 8) {
            for (k_block = z; k_block < N; k_block += 8) {

                __m512d b[64];
                __m512d c[8] = {
                    _mm512_setzero_pd(),
                    _mm512_setzero_pd(),
                    _mm512_setzero_pd(),
                    _mm512_setzero_pd(),
                    _mm512_setzero_pd(),
                    _mm512_setzero_pd(),
                    _mm512_setzero_pd(),
                    _mm512_setzero_pd(),
                };

                //for (i = 0; i < N * N; i++) BB[i] = 0.0;
                for (i = 0; i < 8 && i_block+i < M; i++) {
                    int len = k_block + 8 < M? 8 : M-k_block;
                    //memcpy(BB + i*N, B + (i_block+i)*M+k_block, len * sizeof(double));
                    for (k = 0; k < 8 && k_block+k < M; k++) 
                        b[i*8+k] = _mm512_set1_pd(B[(i_block+i)*M+k_block+k]);
                }

                __m512d a;
                for (k = 0; k < 8 && k_block+k < M; k++) {
                    int len = j_block + 8 < M? 8 : M-j_block;
                    int base = (k_block+k)*M+j_block;
                    if (len == 8) {
                        a = _mm512_loadu_pd(A + base);
                    } else {
                        tmp[0] = A[base];
                        tmp[1] = len > 1 ? A[base+1] : 0;
                        tmp[2] = len > 2 ? A[base+2] : 0;
                        tmp[3] = len > 3 ? A[base+3] : 0;
                        tmp[4] = len > 4 ? A[base+4] : 0;
                        tmp[5] = len > 5 ? A[base+5] : 0;
                        tmp[6] = len > 6 ? A[base+6] : 0;
                        a = _mm512_set_pd(0, tmp[6], tmp[5], tmp[4], 
                                tmp[3], tmp[2], tmp[1], tmp[0]);
                    }

                    for (i = 0; i < 8 && i_block+i < M; i++) {
                        //__m512d b = _mm512_set1_pd(BB[i * N + k]);
                        //c[i] = _mm512_fmadd_pd(b, a, c[i]);
                        c[i] = _mm512_fmadd_pd(b[i*8+k], a, c[i]);
                    }
                }

                for (i = 0; i < 8 && i_block+i<M; i++) {
                    _mm512_storeu_pd(tmp, c[i]);
                    for (j = 0; j < 8; j++) 
                        if (j_block + j < M) 
                            C[(i_block + i) * M + (j_block + j)] += tmp[j];
                }

            }
        }
    }
}


const int BLOCK_SIZE = 24;
const int KERNEL_SIZE = 8;

void kernel_matmul(const int M, const double *A, const double *B, double *C,
        const int x, const int y, const int z,
        const int X, const int Y, const int Z) {
    int i, j, k;

    for (i = x; i < x+KERNEL_SIZE && i < X; i ++)
        for (j = y; j < y+KERNEL_SIZE && j < Y; j ++)
            for (k = z; k < z+KERNEL_SIZE && k < Z; k ++)
                C[i*M+j] += A[i*M+k] * B[k*M+j];
}

void kernel_matmul_avx512(const int M, const double *A, const double *B, double *C,
        const int x, const int y, const int z,
        const int X, const int Y, const int Z) {
    int i, j, k;

    __m512d a[KERNEL_SIZE * KERNEL_SIZE];
    __m512d b;
    __m512d c[KERNEL_SIZE];
    double tmp[KERNEL_SIZE];
    for (i = 0; i < KERNEL_SIZE; i ++) c[i] = _mm512_setzero_pd();

    for (i = 0; i < KERNEL_SIZE && x+i < X; i++) {
        //int len = z+KERNEL_SIZE < Z? KERNEL_SIZE : Z-z;
        for (k = 0; k < KERNEL_SIZE && z+k < Z; k ++)
            a[i*KERNEL_SIZE+k] = _mm512_set1_pd(A[(x+i)*M+z+k]);
    }

    for (k = 0; k < KERNEL_SIZE && z+k < Z; k++) {
        int len = y+KERNEL_SIZE < Y? KERNEL_SIZE : Y-y;
        int base = (z+k)*Z+y;
        if (len == 8) {
            b = _mm512_loadu_pd(A + base);
        } else {
            tmp[0] = A[base];
            tmp[1] = len > 1 ? A[base+1] : 0;
            tmp[2] = len > 2 ? A[base+2] : 0;
            tmp[3] = len > 3 ? A[base+3] : 0;
            tmp[4] = len > 4 ? A[base+4] : 0;
            tmp[5] = len > 5 ? A[base+5] : 0;
            tmp[6] = len > 6 ? A[base+6] : 0;
            b = _mm512_set_pd(0, tmp[6], tmp[5], tmp[4], 
                    tmp[3], tmp[2], tmp[1], tmp[0]);

        }

        for (i = 0; i < 8 && x+i<X; i++) {
            c[i] = _mm512_fmadd_pd(a[i*KERNEL_SIZE+k], b, c[i]);
        }
    }

    for (i = 0; i < KERNEL_SIZE && x+i<X; i ++) {
        _mm512_storeu_pd(tmp, c[i]);
        for (j = 0; j < 8; j ++) 
            if (y+j < Y) C[(x+i)*X + y+j] += tmp[j];
    }
}

// block to N by N kernel
void block_matmul(const int M, const double *A, const double *B, double *C,
        const int X, const int Y, const int Z)
{
    int x, y, z, xx, yy, zz;
    for (x = 0; x < X; x++) 
        for (y = 0; y < Y; y++) {
            double cij = C[x*M+y];
            for (z = 0; z < Z; z++) 
                //C[i*M+j] += A[i*M+k] * B[k*M+j];
                cij += A[x*M+z] * B[z*M+y];
            C[x*M+y] = cij;
        }
    /*
    for (x = 0; x < X; x += KERNEL_SIZE)
        for (y = 0; y < Y; y += KERNEL_SIZE)
            for (z = 0; z < Z; z += KERNEL_SIZE) {
                for (xx = x; xx < x+KERNEL_SIZE && xx < X; xx ++)
                    for (yy = y; yy < y+KERNEL_SIZE && yy < Y; yy ++)
                        for (zz = z; zz < z+KERNEL_SIZE && zz < Z; zz ++)
                            C[xx*M+yy] += A[xx*M+zz] * B[zz*M+yy];
            }
    */
}

void square_dgemm(const int M, 
        const double * restrict A, 
        const double * restrict B, 
        double * restrict C)
{
    int i, j, k;
    for (i = 0; i < M; i += BLOCK_SIZE)
        for (j = 0; j < M; j += BLOCK_SIZE)
            for (k = 0; k < M; k += BLOCK_SIZE) { 
                int x = (i+BLOCK_SIZE > M ? M-i : BLOCK_SIZE);
                int y = (j+BLOCK_SIZE > M ? M-j : BLOCK_SIZE);
                int z = (k+BLOCK_SIZE > M ? M-k : BLOCK_SIZE);
                //block_matmul(M, A, B, C, i, j, k);
                block_matmul(M, B+i*M+k, A+k*M+j, C+i*M+j, x, y, z);
            }
}
