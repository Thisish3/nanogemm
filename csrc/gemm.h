#pragma once
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { NANOGEMM_K01 = 1, NANOGEMM_K02, NANOGEMM_K03, NANOGEMM_K04, NANOGEMM_K05 } nanogemm_kernel_t;
typedef struct { int bm, bn, bk, tm, tn, stages; } nanogemm_tile_t;  /* all 0 = default */
enum { NANOGEMM_OK = 0, NANOGEMM_EINVAL = -1, NANOGEMM_ETILE = -2 };
int nanogemm_sgemm(nanogemm_kernel_t k, nanogemm_tile_t t, int M, int N, int K, float alpha,
                 const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc,
                 cudaStream_t stream);
/* tile used when the caller passes an all-zero tile (k01-k03: all zero). k05 picks by shape:
   128x256x16x16x8 3-stage when ceil(M/128)*ceil(N/256) >= 256 blocks, else 128x128x32x8x4 2-stage
   (rule measured in docs/results/2026-11-10_heur_final.csv, LEDGER R16). */
nanogemm_tile_t nanogemm_default_tile(nanogemm_kernel_t k, int M, int N, int K);
/* fills up to cap supported tiles for k04/k05 on the current device; returns count */
int nanogemm_list_tiles(nanogemm_kernel_t k, nanogemm_tile_t* out, int cap);
#ifdef __cplusplus
}
#endif
