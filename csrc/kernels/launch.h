#pragma once
#include "../gemm.h"
int hx_launch(nanogemm_kernel_t k, nanogemm_tile_t t, int M, int N, int K, float alpha,
              const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc,
              cudaStream_t s);
