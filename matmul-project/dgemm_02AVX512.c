const char* dgemm_desc = "Exp 02 AVX-512";
#include <immintrin.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

#define KERNEL_SIZE 8

#ifndef KERNEL_FACTOR
#define KERNEL_FACTOR 1
#endif

void square_dgemm(const int M, const double * restrict A, 
        const double * restrict B, double * restrict C)
{
    int i, j, k;
    int x, y, z, xx, yy, zz;
    double tmp[KERNEL_SIZE];
    __m512d a[KERNEL_SIZE*KERNEL_FACTOR];
#ifdef IJK
    for (x = 0; x < M; x += KERNEL_SIZE)
        for (y = 0; y < M; y += KERNEL_SIZE)
            for (z = 0; z < M; z += KERNEL_SIZE*KERNEL_FACTOR) 
#elif IKJ
    for (x = 0; x < M; x += KERNEL_SIZE)
        for (z = 0; z < M; z += KERNEL_SIZE*KERNEL_FACTOR) 
            for (y = 0; y < M; y += KERNEL_SIZE)
#else
    for (z = 0; z < M; z += KERNEL_SIZE*KERNEL_FACTOR) 
        for (x = 0; x < M; x += KERNEL_SIZE)
            for (y = 0; y < M; y += KERNEL_SIZE)
#endif
            {
                int KX = (x+KERNEL_SIZE) > M ? M-x : KERNEL_SIZE;
                int KY = (y+KERNEL_SIZE) > M ? M-y : KERNEL_SIZE;
                int KZ = (z+KERNEL_SIZE*KERNEL_FACTOR) > M ? M-z : KERNEL_SIZE*KERNEL_FACTOR;

                int base_a = z*M+y;
                int base_b = x*M+z;
                int base_c = x*M+y;

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
