#include "launch.h"

#define HX_LAUNCH_ARGS                                                                    \
  int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, \
      float beta, float *C, int ldc, cudaStream_t s
void launch_k01(HX_LAUNCH_ARGS);
void launch_k02(HX_LAUNCH_ARGS);
void launch_k03(HX_LAUNCH_ARGS);
int launch_k04(nanogemm_tile_t t, HX_LAUNCH_ARGS);
int list_k04(nanogemm_tile_t* out, int cap);
int launch_k05(nanogemm_tile_t t, HX_LAUNCH_ARGS);
int list_k05(nanogemm_tile_t* out, int cap);

namespace {
bool is_default(const nanogemm_tile_t& t) {
  return t.bm == 0 && t.bn == 0 && t.bk == 0 && t.tm == 0 && t.tn == 0 && t.stages == 0;
}
}  // namespace

int hx_launch(nanogemm_kernel_t k, nanogemm_tile_t t, int M, int N, int K, float alpha,
              const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc,
              cudaStream_t s) {
  switch (k) {
    case NANOGEMM_K01:
      if (!is_default(t)) return NANOGEMM_ETILE;
      launch_k01(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
      return NANOGEMM_OK;
    case NANOGEMM_K02:
      if (!is_default(t)) return NANOGEMM_ETILE;
      launch_k02(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
      return NANOGEMM_OK;
    case NANOGEMM_K03:
      if (!is_default(t)) return NANOGEMM_ETILE;
      launch_k03(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
      return NANOGEMM_OK;
    case NANOGEMM_K04:
      return launch_k04(t, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
    case NANOGEMM_K05:
      return launch_k05(t, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
    default:
      return NANOGEMM_ETILE;
  }
}

extern "C" nanogemm_tile_t nanogemm_default_tile(nanogemm_kernel_t k, int M, int N, int) {
  if (k == NANOGEMM_K04) return nanogemm_tile_t{64, 128, 16, 8, 4, 0};  // 4096^3 sweep winner (R10)
  if (k == NANOGEMM_K05) {
    const long long big_blocks = (long long)((M + 127) / 128) * ((N + 255) / 256);
    return big_blocks >= 256 ? nanogemm_tile_t{128, 256, 16, 16, 8, 3}   // R15: best at 4096^3/8192^3
                             : nanogemm_tile_t{128, 128, 32, 8, 4, 2};  // R10: better when few blocks
  }
  return nanogemm_tile_t{0, 0, 0, 0, 0, 0};
}

extern "C" int nanogemm_list_tiles(nanogemm_kernel_t k, nanogemm_tile_t* out, int cap) {
  if (!out || cap <= 0) return 0;
  if (k == NANOGEMM_K04) return list_k04(out, cap);
  if (k == NANOGEMM_K05) return list_k05(out, cap);
  return 0;
}
