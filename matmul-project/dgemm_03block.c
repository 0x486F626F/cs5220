const char* dgemm_desc = "My awesome dgemm 512.";
#include <immintrin.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

#define KERNEL_SIZE 8
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 160
#endif

void square_dgemm(const int M, const double * restrict A, 
        const double * restrict B, double * restrict C)
{
    //printf("%d\n", BLOCK_SIZE);
    int bi, bj, bk, i, j, k;
    int x, y, z, xx, yy, zz;
    double tmp[KERNEL_SIZE];
    __m512d a[KERNEL_SIZE];
    for (bk = 0; bk < M; bk += BLOCK_SIZE) 
        for (bi = 0; bi < M; bi += BLOCK_SIZE)
            for (bj = 0; bj < M; bj += BLOCK_SIZE)
            { 
                int X = (bi+BLOCK_SIZE > M ? M-bi : BLOCK_SIZE);
                int Y = (bj+BLOCK_SIZE > M ? M-bj : BLOCK_SIZE);
                int Z = (bk+BLOCK_SIZE > M ? M-bk : BLOCK_SIZE);
                for (z = 0; z < Z; z += KERNEL_SIZE) 
                    for (x = 0; x < X; x += KERNEL_SIZE)
                        for (y = 0; y < Y; y += KERNEL_SIZE)
                        {
                            int KX = (x+KERNEL_SIZE) > X ? X-x : KERNEL_SIZE;
                            int KY = (y+KERNEL_SIZE) > Y ? Y-y : KERNEL_SIZE;
                            int KZ = (z+KERNEL_SIZE) > Z ? Z-z : KERNEL_SIZE;

                            int base_a = (bk+z)*M+bj+y;
                            int base_b = (bi+x)*M+bk+z;
                            int base_c = (bi+x)*M+bj+y;

                            for (k = 0; k < KZ; k++)
                                a[k] = _mm512_load_pd(A+base_a+k*M);

                            for (i = 0; i < KX; i++) {
                                __m512d c = _mm512_setzero_pd();
                                for (k = 0; k < KZ; k++)
                                    c = _mm512_fmadd_pd(_mm512_set1_pd(B[base_b+i*M+k]), a[k], c);
                                _mm512_store_pd(tmp, c);
                                for (j = 0; j < KY; j ++) 
                                    C[base_c+i*M+j] += tmp[j];
                            }

                        }

            }
}
