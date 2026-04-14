# NPUKernelBench Level 1 — A5 Evaluation Results

> Updated: 2026-04-14. Hardware: Ascend950PR (A5). CANN 9.0.T501.
> Evaluation: `utils/verification_ascendc.py` + `utils/performance.py`.
> All performance numbers independently verified (never trust worker-reported numbers).

## Summary Table

| # | Problem | Type | Mode | Precision | Best | Worst | Mean | ≥0.6x | ≥1.0x | Status |
|:---:|---------|:----:|:----:|:---------:|:----:|:-----:|:----:|:-----:|:-----:|:------:|
| 1 | GELU | Elem | SIMD | **PASS** 50/50 | 1.84x | 0.52x | **1.35x** ✅ | 49/50 | 46/50 | Verified |
| 2 | SwiGLU | Elem+Chunk | SIMD | **PASS** 50/50 | 12.74x | 0.69x | **1.75x** ✅ | 50/50 | 45/50 | Verified |
| 3 | Add | Elem+Bcast | SIMD | **PASS** 50/50 | 5.71x | 0.44x | **0.99x** ✅ | 42/50 | 20/50 | Verified |
| 4 | Abs | Elem | SIMD | **PASS** 50/50 | 1.41x | 1.00x | **1.22x** ✅ | 50/50 | 49/50 | Verified |
| 5 | Cumsum | Scan | SIMD | **PASS** 51/51 | 1.19x | 0.02x | **0.48x** ⚠️ | 16/51 | 8/51 | Verified ⚠️ |
| 6 | Histc | Histogram | SIMD | **PASS** 15/15 | 0.41x | 0.14x | **0.27x** ⚠️ | 0/15 | 0/15 | Verified ⚠️ |
| 7 | Sum | Reduction | SIMD | **PASS** 44/44 | 1.57x | 0.19x | **0.55x** ⚠️ | 11/44 | 7/44 | Verified ⚠️ |
| 8 | Sort | Sort | SIMD | **PASS** 31/31 | 1.19x | 0.01x | **0.32x** ⚠️ | 6/31 | 1/31 | Verified ⚠️ |
| 9 | TopK | Selection | SIMD | **PASS** 29/29 | 1.40x | 0.00x | **0.31x** ⚠️ | 6/29 | 2/29 | Verified ⚠️ |
| 10 | LayerNorm | Norm | SIMD | **PASS** 60/60 | 16.80x | 0.09x | **1.08x** ✅ | 27/60 | 9/60 | Verified |
| 11 | GroupNorm | Norm | SIMD | **PASS** 73/73 | 1.28x | 0.06x | **0.65x** ✅ | 37/73 | 16/73 | Verified |
| 12 | Permute | DataMove | SIMT | **PASS** 149/149 | 1.66x | 0.01x | **0.50x** ⚠️ | 77/149 | 7/149 | Verified ⚠️ |
| 13 | Cat | DataMove | SIMD | **PASS** 51/51 | 1.74x | 0.33x | **1.24x** ✅ | 48/51 | 35/51 | Verified |
| 14 | Split | DataMove | SIMD | **PASS** 57/57 | 0.65x | 0.01x | **0.29x** ⚠️ | 1/57 | 0/57 | Verified ⚠️ |
| 15 | Pad | DataMove | SIMT+SIMD | **PASS** 51/51 | 2.05x | 0.02x | **0.58x** ⚠️ | 23/51 | 12/51 | Verified ⚠️ |
| 16 | Repeat | DataMove | SIMD | **PASS** 49/49 | 19.97x | 0.05x | **1.39x** ✅ | 36/49 | 25/49 | Verified |
| 17 | AdamW | Optimizer | SIMD | **PASS** 18/18 | 10.40x | 3.87x | **5.65x** ✅ | 18/18 | 18/18 | Verified |
| 18 | Index | Indexing | SIMD | **PASS** 41/41 | 3.35x | 0.01x | **0.58x** ⚠️ | 14/41 | 4/41 | Verified ⚠️ |
| 19 | IndexPut | Indexing | SIMT | ❌ **FAIL** 32/46 | 2.25x | 0.50x | **1.39x** ✅ | 45/46 | 40/46 | ❌ Precision FAIL |
| 20 | Gather | Indexing | SIMT | **PASS** 47/47 | 21.34x | 0.11x | **2.20x** ✅ | 39/47 | 30/47 | Verified |
| 21 | Scatter | Indexing | SIMT | **PASS** 47/47 | 283.00x | 0.43x | **39.56x** ✅ | 46/47 | 45/47 | Verified |
| 22 | Nonzero | Selection | SIMT | **PASS** | — | — | — | — | — | Pending |
| 23 | RepeatInterleave | DataMove | SIMD | **PASS** 75/75 | 21.24x | 0.12x | **1.28x** ✅ | 61/75 | 35/75 | Verified |
| 24 | EmbeddingDenseBackward | Sparse | SIMT | ❌ **FAIL** 27/30 | 0.41x | 0.02x | **0.13x** ⚠️ | 0/30 | 0/30 | ❌ Precision FAIL |
| 25 | NLLLoss | Loss | SIMT | **PASS** 50/50 | 6.51x | 0.04x | **0.81x** ✅ | 29/50 | 8/50 | Verified |
| 26 | AvgPool3d | Pooling | SIMT | **PASS** 72/72 | 22.96x | 0.37x | **1.85x** ✅ | 62/72 | 35/72 | Verified |
| 27 | MaxPool3d | Pooling | SIMT | **PASS** 50/50 | 1.20x | 0.06x | **0.68x** ✅ | 31/50 | 6/50 | Verified |
| 28 | Interpolate | Resize | SIMT | **PASS** 73/73 | 19.83x | 0.05x | **1.72x** ✅ | 22/73 | 11/73 | Verified |
| 29 | DynamicQuant | Quant | SIMD | **PASS** 42/42 | 7.62x | 0.03x | **0.60x** ✅ | 3/42 | 3/42 | Verified |
| 30 | NMS | Detection | SIMD | **PASS** 31/31 | 4.67x | 0.11x | **1.19x** ✅ | 15/31 | 12/31 | Verified |
| 31 | IOU | Detection | SIMD | ❌ **FAIL** 22/30 | 10.22x | 0.27x | **2.42x** ✅ | 25/30 | 21/30 | ❌ Precision FAIL |

