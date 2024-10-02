const char* dgemm_desc = "Basic, three-loop dgemm.";

void square_dgemm_ijk(const int M, const double * restrict A, 
		  const double * restrict B, double * restrict C)
{
    int i, j, k;
    for (i = 0; i < M; ++i) 
        for (j = 0; j < M; ++j) 
        {
            double c = C[i*M+j];
            for (k = 0; k < M; ++k)
                c += B[i*M+k] * A[k*M+j];
            C[i*M+j] = c;
        }
}

void square_dgemm_ikj(const int M, const double * restrict A, 
		  const double * restrict B, double * restrict C)
{
    int i, j, k;
    for (i = 0; i < M; ++i) 
        for (k = 0; k < M; ++k)
            for (j = 0; j < M; ++j) 
                C[i*M+j] += B[i*M+k] * A[k*M+j];
}

void square_dgemm_kij(const int M, const double * restrict A, 
		  const double * restrict B, double * restrict C)
{
    int i, j, k;
    for (k = 0; k < M; ++k)
        for (i = 0; i < M; ++i) 
            for (j = 0; j < M; ++j) 
                C[i*M+j] += B[i*M+k] * A[k*M+j];
}

void square_dgemm_jik(const int M, const double * restrict A, 
        const double * restrict B, double * restrict C)
{
    int i, j, k;
    for (j = 0; j < M; ++j) 
        for (i = 0; i < M; ++i) 
        {
            double c = C[i*M+j];
            for (k = 0; k < M; ++k)
                c += B[i*M+k] * A[k*M+j];
            C[i*M+j] = c;
        }
}

void square_dgemm_jki(const int M, const double * restrict A, 
        const double * restrict B, double * restrict C)
{
    int i, j, k;
    for (j = 0; j < M; ++j) 
        for (k = 0; k < M; ++k)
            for (i = 0; i < M; ++i) 
                C[i*M+j] += B[i*M+k] * A[k*M+j];
}

void square_dgemm_kji(const int M, const double * restrict A, 
        const double * restrict B, double * restrict C)
{
    int i, j, k;
    for (k = 0; k < M; ++k)
        for (j = 0; j < M; ++j) 
            for (i = 0; i < M; ++i) 
                C[i*M+j] += B[i*M+k] * A[k*M+j];
}


void square_dgemm(const int M, const double * restrict A, 
        const double * restrict B, double * restrict C)
{
#ifdef IJK
    square_dgemm_ijk(M, A, B, C);
#elif IKJ
    square_dgemm_ikj(M, A, B, C);
#elif KIJ
    square_dgemm_kij(M, A, B, C);
#elif JIK
    square_dgemm_jik(M, A, B, C);
#elif JKI
    square_dgemm_jki(M, A, B, C);
#elif KJI
    square_dgemm_kji(M, A, B, C);
#endif
}
