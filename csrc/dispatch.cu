#include <climits>
#include "gemm.h"
#include "kernels/common.cuh"
#include "kernels/launch.h"

namespace {
bool bad_args(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
              const float* C, int ldc) {
  if (M < 0 || N < 0 || K < 0) return true;
  if (M == 0 || N == 0) return false;
  if (ldc < N || !C) return true;
  // Kernels index with int: the last element of each matrix must have an offset < 2^31.
  auto too_big = [](long long rows, long long ld, long long cols) {
    return (rows - 1) * ld + cols > static_cast<long long>(INT_MAX);
  };
  if (too_big(M, ldc, N)) return true;
  if (K == 0) return false;
  return !A || !B || lda < K || ldb < N || too_big(M, lda, K) || too_big(K, ldb, N);
}
}  // namespace

extern "C" int nanogemm_sgemm(nanogemm_kernel_t k, nanogemm_tile_t t, int M, int N, int K,
                            float alpha, const float* A, int lda, const float* B, int ldb,
                            float beta, float* C, int ldc, cudaStream_t s) {
  if (k < NANOGEMM_K01 || k > NANOGEMM_K05) return NANOGEMM_EINVAL;
  if (bad_args(M, N, K, A, lda, B, ldb, C, ldc)) return NANOGEMM_EINVAL;
  if (M == 0 || N == 0) return NANOGEMM_OK;
  if (K == 0) {
    dim3 b(32, 8), g((N + 31) / 32, (M + 7) / 8);
    scale_c<<<g, b, 0, s>>>(M, N, beta, C, ldc);
    return static_cast<int>(cudaGetLastError());
  }
  int r = hx_launch(k, t, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
  if (r != NANOGEMM_OK) return r;
  return static_cast<int>(cudaGetLastError());
}
