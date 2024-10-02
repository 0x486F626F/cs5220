const char* dgemm_desc = "My awesome dgemm 512.";
#include <immintrin.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

//const int BLOCK_SIZE = 128;
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 144
#endif

#define KERNEL_SIZE 8

void square_dgemm(const int N, 
        double * AA, 
        double * BB, 
        double * CC)
{
    int bi, bj, bk, i, j, k;
    int x, y, z, xx, yy, zz;
    __declspec(align(64)) double tmp[KERNEL_SIZE];
    __m512d a[KERNEL_SIZE];

    int M = (N+KERNEL_SIZE-1)/KERNEL_SIZE * KERNEL_SIZE;//BLOCK_SIZE*BLOCK_SIZE;
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
    for (bi = 0; bi < M; bi += BLOCK_SIZE)
        for (bk = 0; bk < M; bk += BLOCK_SIZE)  
            for (bj = 0; bj < M; bj += BLOCK_SIZE) 
            {
                int X = (bi+BLOCK_SIZE > M ? M-bi : BLOCK_SIZE);
                int Y = (bj+BLOCK_SIZE > M ? M-bj : BLOCK_SIZE);
                int Z = (bk+BLOCK_SIZE > M ? M-bk : BLOCK_SIZE);
                for (x = 0; x < X; x += KERNEL_SIZE)
                    for (z = 0; z < Z; z += KERNEL_SIZE) 
                        for (y = 0; y < Y; y += KERNEL_SIZE)
                        {
                            int base_a = (bk+z)*M+bj+y;

                            __m512d vec0 = _mm512_load_pd(A+base_a+M*0);
                            __m512d vec1 = _mm512_load_pd(A+base_a+M*1);
                            __m512d vec2 = _mm512_load_pd(A+base_a+M*2);
                            __m512d vec3 = _mm512_load_pd(A+base_a+M*3);
                            __m512d vec4 = _mm512_load_pd(A+base_a+M*4);
                            __m512d vec5 = _mm512_load_pd(A+base_a+M*5);
                            __m512d vec6 = _mm512_load_pd(A+base_a+M*6);
                            __m512d vec7 = _mm512_load_pd(A+base_a+M*7);
#ifdef UNROLL2
                            for (i = 0; i < 8; i += 2) {
#else
                            for (i = 0; i < 8; i += 1) {
#endif
                                int base_b = (bi+x+i)*M+bk+z;
                                __m512d vec8 = _mm512_set1_pd(B[base_b+0]);
                                __m512d vec9 = _mm512_set1_pd(B[base_b+1]);
                                __m512d vec10 = _mm512_set1_pd(B[base_b+2]);
                                __m512d vec11 = _mm512_set1_pd(B[base_b+3]);
                                __m512d vec12 = _mm512_set1_pd(B[base_b+4]);
                                __m512d vec13 = _mm512_set1_pd(B[base_b+5]);
                                __m512d vec14 = _mm512_set1_pd(B[base_b+6]);
                                __m512d vec15 = _mm512_set1_pd(B[base_b+7]);

#ifdef UNROLL2
                                base_b += M;
                                __m512d vec16 = _mm512_set1_pd(B[base_b+0]);
                                __m512d vec17 = _mm512_set1_pd(B[base_b+1]);
                                __m512d vec18 = _mm512_set1_pd(B[base_b+2]);
                                __m512d vec19 = _mm512_set1_pd(B[base_b+3]);
                                __m512d vec20 = _mm512_set1_pd(B[base_b+4]);
                                __m512d vec21 = _mm512_set1_pd(B[base_b+5]);
                                __m512d vec22 = _mm512_set1_pd(B[base_b+6]);
                                __m512d vec23 = _mm512_set1_pd(B[base_b+7]);
#endif

                                vec8 = _mm512_mul_pd(vec0, vec8);
                                vec9 = _mm512_mul_pd(vec1, vec9);
                                vec10 = _mm512_mul_pd(vec2, vec10);
                                vec11 = _mm512_mul_pd(vec3, vec11);
                                vec12 = _mm512_mul_pd(vec4, vec12);
                                vec13 = _mm512_mul_pd(vec5, vec13);
                                vec14 = _mm512_mul_pd(vec6, vec14);
                                vec15 = _mm512_mul_pd(vec7, vec15);

                                vec8 = _mm512_add_pd(vec8, vec9);
                                vec10 = _mm512_add_pd(vec10, vec11);
                                vec12 = _mm512_add_pd(vec12, vec13);
                                vec14 = _mm512_add_pd(vec14, vec15);

                                vec8 = _mm512_add_pd(vec8, vec10);
                                vec12 = _mm512_add_pd(vec12, vec14);

                                vec8 = _mm512_add_pd(vec8, vec12);

#ifdef UNROLL2
                                base_b += M;
                                vec16 = _mm512_mul_pd(vec0, vec16);
                                vec17 = _mm512_mul_pd(vec1, vec17);
                                vec18 = _mm512_mul_pd(vec2, vec18);
                                vec19 = _mm512_mul_pd(vec3, vec19);
                                vec20 = _mm512_mul_pd(vec4, vec20);
                                vec21 = _mm512_mul_pd(vec5, vec21);
                                vec22 = _mm512_mul_pd(vec6, vec22);
                                vec23 = _mm512_mul_pd(vec7, vec23);

                                vec16 = _mm512_add_pd(vec16, vec17);
                                vec18 = _mm512_add_pd(vec18, vec19);
                                vec20 = _mm512_add_pd(vec20, vec21);
                                vec22 = _mm512_add_pd(vec22, vec23);

                                vec16 = _mm512_add_pd(vec16, vec18);
                                vec20 = _mm512_add_pd(vec20, vec22);

                                vec16 = _mm512_add_pd(vec16, vec20);
#endif

                                int base_c = (bi+x+i)*M+bj+y;
                                vec8 = _mm512_add_pd(vec8, _mm512_load_pd(C+base_c));
                                _mm512_store_pd(C+base_c, vec8);
#ifdef UNROLL2
                                base_c += M;
                                vec16 = _mm512_add_pd(vec16, _mm512_load_pd(C+base_c));
                                _mm512_store_pd(C+base_c, vec16);
#endif
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
