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


const int BLOCK_SIZE = 16;
const int KERNEL_SIZE = 8;

void kernel_matmul(const int M, const double *A, const double *B, double *C,
        const int x, const int y, const int z,
        const int X, const int Y, const int Z) {
    int i, j, k;

    for (i = 0; i < KERNEL_SIZE && x+i < X; i ++)
        for (j = 0; j < KERNEL_SIZE && y+j < Y; j ++ ) {
            double c = C[(i+x)*M+y+j];
            for (k = 0; k < KERNEL_SIZE && z+k < Z; k ++)
                c += A[(i+x)*M+k+z] * B[(k+z)*M+j+y];
            C[(i+x)*M+j+y] = c;
        }
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
        for (k = 0; k < KERNEL_SIZE && z+k < Z; k ++)
            a[i*KERNEL_SIZE+k] = _mm512_set1_pd(A[(x+i)*M+z+k]);
    }

    for (k = 0; k < KERNEL_SIZE && z+k < Z; k++) {
        int len = y+KERNEL_SIZE < Y? KERNEL_SIZE : Y-y;
        int base = (z+k)*M+y;
        if (len == 8) {
            b = _mm512_loadu_pd(B + base);
        } else {
            tmp[0] = B[base];
            tmp[1] = len > 1 ? B[base+1] : 0;
            tmp[2] = len > 2 ? B[base+2] : 0;
            tmp[3] = len > 3 ? B[base+3] : 0;
            tmp[4] = len > 4 ? B[base+4] : 0;
            tmp[5] = len > 5 ? B[base+5] : 0;
            tmp[6] = len > 6 ? B[base+6] : 0;
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
            if (y+j < Y) C[(x+i)*M + y+j] += tmp[j];
    }
}

void basic_matmul(const int M, const double * restrict A, const double * restrict B, 
        double * restrict C, const int X, const int Y, const int Z)
{
    // basic matmul
    int x, y, z;
    for (x = 0; x < X; x++) 
        for (y = 0; y < Y; y++)
            for (z = 0; z < Z; z++) 
                C[x*M+y] += A[x*M+z] * B[z*M+y];
}

void square_dgemm(const int M, 
        const double * restrict A, 
        const double * restrict B, 
        double * restrict C)
{
    int bi, bj, bk;
    int x, y, z, xx, yy, zz;
    for (bi = 0; bi < M; bi += BLOCK_SIZE)
        for (bj = 0; bj < M; bj += BLOCK_SIZE)
            for (bk = 0; bk < M; bk += BLOCK_SIZE) { 
                int X = (bi+BLOCK_SIZE > M ? M-bi : BLOCK_SIZE);
                int Y = (bj+BLOCK_SIZE > M ? M-bj : BLOCK_SIZE);
                int Z = (bk+BLOCK_SIZE > M ? M-bk : BLOCK_SIZE);
                //block_matmul(M, B+bi*M+bk, A+bk*M+bj, C+bi*M+bj, X, Y, Z);
                for (x = 0; x < X; x += KERNEL_SIZE)
                    for (y = 0; y < Y; y += KERNEL_SIZE)
                        for (z = 0; z < Z; z += KERNEL_SIZE) {
                            int KX = (x+KERNEL_SIZE) > X ? X-x : KERNEL_SIZE;
                            int KY = (y+KERNEL_SIZE) > Y ? Y-y : KERNEL_SIZE;
                            int KZ = (z+KERNEL_SIZE) > Z ? Z-z : KERNEL_SIZE;
                            //basic_matmul(M, B+(bi+x)*M+bk+z, A+(bk+z)*M+bj+y, C+(bi+x)*M+bj+y, KX, KY, KZ);
                            kernel_matmul_avx512(M, B+(bi+x)*M+bk+z, A+(bk+z)*M+bj+y, C+(bi+x)*M+bj+y, 0, 0, 0, KX, KY, KZ);
                            /*
                            for (xx = 0; xx < KX; xx++) 
                                for (yy = 0; yy < KY; yy++) 
                                    for (zz = 0; zz < KZ; zz++)
                                        C[(bi+x+xx)*M+y+yy+bj] += B[(bi+xx+x)*M+z+zz+bk] * A[(bk+zz+z)*M+y+yy+bj];
                            */
                        }

            }
}
