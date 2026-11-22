#include <cstdio>
#include <cuda_runtime.h>
#include "../csrc/gemm.h"
static int fails = 0;
#define EXPECT(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
int main() {
  float *A, *B, *C;
  cudaMalloc(&A, 64 * 4); cudaMalloc(&B, 64 * 4); cudaMalloc(&C, 64 * 4);
  nanogemm_tile_t d = {0, 0, 0, 0, 0, 0};
  for (int k = NANOGEMM_K01; k <= NANOGEMM_K05; ++k) {
    auto kk = static_cast<nanogemm_kernel_t>(k);
    EXPECT(nanogemm_sgemm(kk, d, -1, 4, 4, 1, A, 4, B, 4, 0, C, 4, 0) == NANOGEMM_EINVAL);
    EXPECT(nanogemm_sgemm(kk, d, 4, 4, 4, 1, A, 3, B, 4, 0, C, 4, 0) == NANOGEMM_EINVAL);   // lda < K
    EXPECT(nanogemm_sgemm(kk, d, 4, 4, 4, 1, A, 4, B, 3, 0, C, 4, 0) == NANOGEMM_EINVAL);   // ldb < N
    EXPECT(nanogemm_sgemm(kk, d, 4, 4, 4, 1, A, 4, B, 4, 0, C, 3, 0) == NANOGEMM_EINVAL);   // ldc < N
    EXPECT(nanogemm_sgemm(kk, d, 4, 4, 4, 1, nullptr, 4, B, 4, 0, C, 4, 0) == NANOGEMM_EINVAL);
    EXPECT(nanogemm_sgemm(kk, d, 0, 4, 4, 1, nullptr, 4, nullptr, 4, 0, nullptr, 4, 0) == NANOGEMM_OK);
  }
  // int_offset_overflow_rejected: (M-1)*ldc + N >= 2^31 must be EINVAL before any launch.
  EXPECT(nanogemm_sgemm(NANOGEMM_K05, d, 70000, 40000, 4, 1, A, 4, B, 40000, 0, C, 40000, 0) == NANOGEMM_EINVAL);
  // B: (K-1)*ldb + N = 69999*40000 + 4 > INT_MAX while lda >= K, ldb >= N, ldc >= N hold.
  EXPECT(nanogemm_sgemm(NANOGEMM_K05, d, 4, 4, 70000, 1, A, 70000, B, 40000, 0, C, 4, 0) == NANOGEMM_EINVAL);
  // Just under the limit is accepted by the check (not launched here: tiny N makes it cheap).
  EXPECT(nanogemm_sgemm(NANOGEMM_K01, d, 2, 4, 4, 1, A, 4, B, 4, 0, C, 4, 0) == NANOGEMM_OK);
  // K = 0: C = beta * C
  float h[4] = {1, 2, 3, 4};
  cudaMemcpy(C, h, 16, cudaMemcpyHostToDevice);
  EXPECT(nanogemm_sgemm(NANOGEMM_K01, d, 2, 2, 0, 1, nullptr, 0, nullptr, 2, 2.f, C, 2, 0) == NANOGEMM_OK);
  cudaMemcpy(h, C, 16, cudaMemcpyDeviceToHost);
  EXPECT(h[0] == 2 && h[3] == 8);
  EXPECT(nanogemm_sgemm(NANOGEMM_K04, nanogemm_tile_t{96, 96, 8, 8, 8, 0}, 4, 4, 4, 1, A, 4, B, 4, 0, C, 4, 0) == NANOGEMM_ETILE);
  // unsupported_tile_rejected: a K05 tile whose shared memory is near the opt-in limit is
  // accepted or rejected according to the limit read at run time; unknown tiles are always -2.
  {
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    const int bytes = 3 * (128 * 32 + 32 * 128) * 4;  // 98,304 B
    const int want = bytes > optin ? NANOGEMM_ETILE : NANOGEMM_OK;
    const int got = nanogemm_sgemm(NANOGEMM_K05, nanogemm_tile_t{128, 128, 32, 4, 4, 3}, 4, 4, 4, 1, A, 4,
                                 B, 4, 0, C, 4, 0);
    cudaDeviceSynchronize();
    std::printf("unsupported_tile_rejected: tile 128x128x32x4x4x3 needs %d B, opt-in %d B, rc=%d\n",
                bytes, optin, got);
    EXPECT(got == want);
    EXPECT(nanogemm_sgemm(NANOGEMM_K05, nanogemm_tile_t{96, 96, 8, 8, 8, 2}, 4, 4, 4, 1, A, 4, B, 4, 0, C, 4, 0) == NANOGEMM_ETILE);
    EXPECT(nanogemm_sgemm(NANOGEMM_K05, nanogemm_tile_t{128, 128, 8, 8, 8, 4}, 4, 4, 4, 1, A, 4, B, 4, 0, C, 4, 0) == NANOGEMM_ETILE);
    EXPECT(nanogemm_sgemm(NANOGEMM_K04, nanogemm_tile_t{128, 128, 8, 8, 8, 2}, 4, 4, 4, 1, A, 4, B, 4, 0, C, 4, 0) == NANOGEMM_ETILE);
  }
  std::printf(fails ? "check_args: %d failures\n" : "check_args: ok\n", fails);
  return fails != 0;
}
