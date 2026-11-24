# NN-08 Metal Optimization Experiment: Single-Row Right-Transposed FP32 Projection

## Verdict and Recommendation

**Outcome:** Negative result under the predeclared NN-08 acceptance gates. **No production kernel or dispatch changes are shipped**; `internal/device/metal_backend.m` remains unchanged from `v0.11.2` (`8db36638be5b0166d676ae1322d89dd095823cb3`).

The primary candidate (`r2s4`, a SIMD-cooperative `M=1` right-transposed FP32 kernel) achieved **-11.47%** paired median warmed sustained decode latency (`90.933 ms` → `80.917 ms`, winning **6 of 6** alternating fresh-process pairs) and **-11.91%** GPU command-interval latency (`89.35 ms` → `78.71 ms`) with zero regressions on prefill, concurrency, residency, or greedy token IDs (including close decision ID `95551` at position 11), and **lower** aggregate error (p50, p95, RMSE, and max absolute error) against CPU than `v0.11.2`.

However, under the predeclared full-row mixed tolerance `1e-4 + 1e-4*abs(CPU)` across all 4,232,448 unique logits in 32-step sustained decode, `r2s4` produced **4 unique near-zero logit violations** (`max_fraction = 1.08730` vs `0.96079` for `v0.11.2`). Four alternative parallel-K accumulation orders — including IEEE-754 `fma`, 32-way scalar reduction, 16-wide NEON-order `simd_shuffle_xor`, and **48-bit FP64-equivalent compensated summation (`two_prod` + `two_sum`)** — all still exceeded `1.0` (`max_fraction = 1.06825–1.13657`). Conversely, bit-exact sequential `0..K-1` kernels (`f4seq`, `tile`) preserved `0` violations (`max_fraction = 0.96079`) but missed the ≥10% decode speedup gate (`-4.3%` and `+114.6%`).

Per the NN-08 negative-result rule:
- The production `v0.11.2` Metal fast path is preserved unmodified.
- Expanded upstream FP64/tail/special-value tests (`gpu/matmul_metal_test.go`) and isolated resident projection benchmarks (`gpu/matmul_metal_benchmark_test.go`) are retained.
- **GATE-D / LL-05 recommendation:** Retain immutable release `v0.11.2` (`8db36638be5b0166d676ae1322d89dd095823cb3`) for LL-05 unless the owner explicitly revisits the full-row near-zero logit tolerance policy against CPU or prioritizes investigating the rank-2 `RMSNorm` stage cost (`18.27 ms` / 33 dispatches per token) first.

---

## Hypothesis and Tested Designs

In `v0.11.2`, `Scope.MatMulRightTranspose` (`variant == 2` in `internal/device/metal_backend.m`) dispatches scalar `nn_matmul` with a `16×16` threadgroup (`groupHeight = ceil(M/16)`). At `M = 1`, only 16 of 256 threads per threadgroup are active and adjacent lanes stride by `K` floats (`8–32 KiB`) across `right[col * K + inner]`.

We evaluated a narrowly scoped `M = 1` right-transposed kernel (`nn_matmul_row_right_transpose`, compiled lazily alongside `nn_matmul` with `fastMathEnabled = NO`, preserving `M > 1`, standard `MatMul`, ANN `Sequential`, and zero added device memory) across seven kernel variants:

| Variant | Dispatch / Geometry | Accumulation Order |
| --- | --- | --- |
| Control (`v0.11.2`) | `16×16` TG, scalar `nn_matmul` | Sequential `0..K-1` |
| `r2s4` (primary candidate) | `128×1` TG (4 SIMD groups), 2 rows/SIMD, `float4` loads + scalar tail | 128-way `float4` + `simd_sum` |
| `r1s4` / `r4s4` | `128×1` TG, 1 or 4 rows/SIMD, `float4` loads | 128-way `float4` + `simd_sum` |
| `dot` | `128×1` TG, 2 rows/SIMD, `float4` loads, scalar lane accumulator | 32-way + `simd_sum` |
| `comp` | `128×1` TG, 2 rows/SIMD, `float4` loads, `two_prod` + `two_sum` | 48-bit compensated + `simd_sum` |
| `q4` | `128×1` TG, 4 lanes/row (`4×float4` = 16 floats), `simd_shuffle_xor` | 16-wide NEON-like |
| `f4seq` | `128×1` TG, 1 thread/col, `float4` loads | Bit-exact sequential `0..K-1` |
| `tile` | `32×1` TG, `32×33` threadgroup shared tile | Bit-exact sequential `0..K-1` |

---

## Hardware, Environment, and Isolated Harness

- **Machine:** MacBook Air `Mac15,13`, Apple M3 (4P+4E CPU, 10-core GPU), 24 GiB unified memory, macOS `26.5.2 (25F84)`, Darwin `arm64`, Go `1.27.3` (`GOWORK=off`, `CGO_ENABLED=1`, `GOARM64=v8.0`, `GOMAXPROCS=8`).
- **Pinned Llama 3.2 1B snapshot:** `meta-llama/Llama-3.2-1B` revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`; verified manifest SHA-256 `3404dfda97c69cf3a9a3cf15dc4db7017cb049da07462c52f2e3004f6e12ab64`, frozen reference SHA-256 `3a66ddc5cbdef45e397e47a6bd992676ab3d35234a8b389908f721b05466d361`, checkpoint SHA-256 `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a`.
- **Isolated consumer harness:** All end-to-end binaries and raw logs were built and stored strictly inside gitignored `.benchmarks/gpu-nn-08/2026-11-24-m3/`. The control binary (`metal-control`) and CPU reference binary (`cpu`) were built from an unmodified read-only copy of Llama `main` (`63d5a247...`) pinning immutable `github.com/itsmontoya/neuralnetwork@v0.11.2`. The candidate binary (`metal-candidate-r2s4`) was built from a separate gitignored copy with a local `replace` pointing to the NN-08 working tree. **The real Llama repository and its `go.mod` were never modified.**

---

## Isolated Resident Projection Benchmarks

Measured via `Benchmark_MatMulRightTransposeMetal` (`-benchtime=10x -count=5`, median of 5 runs; includes per-op scope submit/wait):

| Shape | Control `v0.11.2` median (ms) | Candidate `r2s4` median (ms) | Delta | Requested Device Bytes |
| --- | ---: | ---: | ---: | ---: |
| `FFN_GateUp_M1_K2048_N8192` | 1.461 | 1.256 | -14.0% | 67,149,824 |
| `FFN_Down_M1_K8192_N2048` | 1.689 | 1.258 | -25.5% | 67,149,824 |
| `Attn_QO_M1_K2048_N2048` | 0.607 | 0.582 | -4.1% | 16,793,600 |
| `Attn_KV_M1_K2048_N512` | 0.376 | 0.248 | -34.0% | 4,204,544 |
| `Classifier_M1_K2048_N128256` | 12.354 | 11.751 | -4.9% | 1,051,194,368 |
| `FFN_GateUp_M6_K2048_N8192` (guard) | 1.645 | 1.522 | -7.5% | 67,354,624 |
| `FFN_GateUp_M27_K2048_N8192` (guard) | 4.286 | 4.922 | +14.8% (noise; 4.30–6.40) | 68,214,784 |
| `FFN_GateUp_M28_K2048_N8192` (guard) | 4.371 | 4.282 | -2.0% | 68,255,744 |
| `FFN_GateUp_M29_K2048_N8192` (guard) | 4.611 | 4.535 | -1.6% | 68,296,704 |
| `OddTail_M1_K257_N33` | 0.245 | 0.164 | -33.1% | 35,084 |

---

## End-to-End Paired Llama 3.2 1B Results (Control `v0.11.2` vs Candidate `r2s4`)

Six alternating fresh-process pairs per workload (each backend first 3 times, rotated workload order, 2 s cooldown, macOS `/usr/bin/time -l` peak RSS bound < 16 GiB):

| Metric | Control `v0.11.2` median (range) | Candidate `r2s4` median (range) | Paired delta (range) | Wins |
| --- | ---: | ---: | ---: | ---: |
| Sustained warm decode, ms/call | 90.933 (89.323–93.622) | **80.917** (79.554–81.752) | **-11.47%** (-14.90% to -8.48%) | **6/6** |
| Sustained warm decode, tok/s | 10.938 (10.611–11.109) | **12.313** (12.090–12.653) | **+12.84%** (+8.82% to +19.24%) | **6/6** |
| Short first inference TTFT, ms | 142.330 (123.215–148.062) | 137.814 (127.754–146.627) | +0.10% (-7.93% to +3.68%) | 3/6 |
| Short launch to first token, s | 2.231 (2.202–2.267) | 2.247 (2.166–2.348) | +1.08% (-2.64% to +4.81%) | 2/6 |
| Short warm prefill, ms | 99.936 (98.311–100.337) | 99.186 (98.995–101.061) | -0.55% (-1.34% to +0.87%) | 4/6 |
| Short warm decode, ms/call | 88.668 (87.921–90.418) | **77.719** (77.319–79.755) | **-12.07%** (-12.77% to -11.31%) | **6/6** |
| Short warm 8 predictions, ms | 721.226 (715.398–733.874) | **645.862** (640.417–659.918) | **-10.25%** (-11.10% to -9.07%) | **6/6** |
| 27-token warm prefill, ms | 300.120 (295.321–301.990) | 300.130 (295.375–302.532) | -0.27% (-1.25% to +2.44%) | 4/6 |
| 28-token warm prefill, ms | 298.443 (297.271–300.861) | 299.381 (295.688–301.630) | +0.25% (-1.72% to +1.15%) | 3/6 |
| 29-token warm prefill, ms | 320.695 (317.640–322.651) | 320.531 (317.728–321.939) | -0.34% (-1.08% to +1.35%) | 4/6 |
| Shared warm aggregate, tok/s | 14.608 (14.513–14.725) | **15.252** (15.080–15.402) | **+3.89%** (+3.14% to +5.93%) | **6/6** |
| Independent warm aggregate, tok/s | 14.370 (14.186–14.798) | **15.145** (14.968–15.387) | **+4.89%** (+3.67% to +7.29%) | **6/6** |
| Short / Sustained peak RSS, GiB | 6.596 / 6.596 | 6.596 / 6.596 | 0.00% / 0.00% | — |
| Short / Sustained post-close RSS, GiB | 5.618 / 5.618 | 5.617 / 5.618 | 0.00% / 0.00% | — |
| Shared peak / post-close RSS, GiB | 6.598 / 5.619 | 6.598 / 5.619 | 0.00% / 0.00% | — |
| Independent peak / post-close RSS, GiB | 8.589 / 1.191 | 8.672 / 1.386 | +3.00% / +6.16% (noisy: -62% to +191%) | — |

### Native Metal Diagnostics (3 repeats × 32 warm sustained decode commands = 96 commands)

| Metric | Control `v0.11.2` | Candidate `r2s4` | Delta |
| --- | ---: | ---: | ---: |
| Command-only GPU interval, ms | 89.35 (85.44–94.28) | **78.71** (75.84–86.97) | **-11.91% (-10.64 ms)** |
| Command-only host wait, ms | 90.04 (86.22–95.05) | **79.41** (76.54–87.27) | **-11.81% (-10.63 ms)** |
| Stage-sampled GPU interval, ms | 92.00 (88.08–100.33) | **81.24** (77.42–97.09) | **-11.70% (-10.76 ms)** |
| FFN gate/up (32 disp), ms | 26.465 (25.723–30.235) | **24.679** (23.711–27.017) | -6.7% |
| FFN down (16 disp), ms | 14.793 (14.413–15.723) | **12.284** (11.910–13.416) | -17.0% |
| Classifier (1 disp), ms | 12.482 (11.965–14.146) | **11.945** (11.547–13.525) | -4.3% |
| Attention Q/O (32 disp), ms | 7.738 (7.590–8.541) | **6.320** (6.068–7.028) | -18.3% |
| Attention K/V (32 disp), ms | 5.686 (5.598–6.291) | **1.595** (1.523–2.205) | -71.9% |
| **Total projections (113 disp), ms** | **67.164** | **56.823** | **-15.40% (-10.341 ms)** |
| RMSNorm (33 disp), ms | 18.272 (18.025–18.657) | 18.292 (18.066–18.746) | +0.1% (unchanged) |
| Compute dispatches / Blits | 242 / 17 | 242 / 17 | exact match |
| Live requested device HWM, B | 4,949,910,528 | 4,949,910,528 | 0 added bytes |
| Driver `currentAllocatedSize` HWM, B | 4,950,605,824 | 4,950,605,824 | 0 added bytes |

---

## Numerical Verification and Root Cause of Gate Failure

All 96 Metal processes reproduced deterministic outputs across repeats, matched every greedy ID (including position 11 ID `95551`: margin `0.04331207` for `r2s4`, `0.04331017` for `v0.11.2`, `0.04334068` frozen), and passed the frozen reference bound `1e-3 + 1e-3*abs(reference)` (`max_frozen_frac = 0.02166` for `r2s4` vs `0.02076` for `v0.11.2`). On `short`, `27`, `28`, `29`, and `recovery`, `r2s4` had **0** full-row violations (`max_fraction = 0.302–0.616`).

On `sustained` (`33 × 128,256 = 4,232,448` unique positions, also reproduced in `shared` and `independent`), comparing every returned FP32 logit against the CPU reference under `1e-4 + 1e-4*abs(CPU)`:

| Variant | Sustained warm decode (ms) | p50 abs vs CPU | p95 abs vs CPU | RMSE vs CPU | Max abs vs CPU | Max tol frac | Unique violations (`>1.0`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control `v0.11.2` | 90.93 | 1.132e-5 | 4.244e-5 | 2.002e-5 | 1.214e-4 | **0.96079** | **0** |
| `r2s4` (128-way `float4` + `simd_sum`) | **80.92** (-11.47%) | **1.073e-5** | **3.943e-5** | **1.885e-5** | **1.194e-4** | 1.08730 | 4 |
| `dot` (32-way scalar + `simd_sum`) | ~81 | 1.073e-5 | 3.910e-5 | 1.875e-5 | 1.225e-4 | 1.13657 | 3 |
| `comp` (48-bit `two_prod` + `two_sum`) | 83.14 (-8.6%) | **1.049e-5** | 4.196e-5 | 1.947e-5 | 1.254e-4 | 1.06825 | 5 |
| `q4` (16-wide NEON-order) | 86.72 (-4.6%) | 1.073e-5 | 4.199e-5 | 1.962e-5 | 1.264e-4 | 1.08846 | 3 |
| `f4seq` (1D 128-thread, sequential) | 87.01 (-4.3%) | 1.132e-5 | 4.244e-5 | 2.002e-5 | 1.214e-4 | **0.96079** | **0** |
| `tile` (`32×33` shared tile, sequential) | 195.10 (+114.6%) | 1.132e-5 | 4.244e-5 | 2.002e-5 | 1.214e-4 | **0.96079** | **0** |

The 4 unique violations for `r2s4` occurred on near-zero logits (`|CPU| < 0.06`, where the mixed tolerance floor is `~1.00e-4`):
- Row 14, ID `16058`: CPU `+0.03449892`, control `+0.03457368` (`frac 0.72266`), candidate `+0.03461140` (`frac 1.08730`).
- Row 14, ID `7514`: CPU `-0.03844665`, control `-0.03835321` (`frac 0.89978`), candidate `-0.03833827` (`frac 1.04364`).
- Row 27, ID `95746`: CPU `+0.00794956`, control `+0.00785542` (`frac 0.93394`), candidate `+0.00784642` (`frac 1.02330`).
- Row 27, ID `99366`: CPU `-0.05659370`, control `-0.05665306` (`frac 0.56183`), candidate `-0.05670026` (`frac 1.00851`).

Because even 48-bit FP64-equivalent compensated dot products (`comp`) exhibit `max_fraction = 1.06825` at these same near-zero positions, the `1e-4` absolute floor leaves less than `4%` headroom above `v0.11.2`'s `0.96079` baseline for any non-sequential FP32 matmul reduction order when combined with the remaining unchanged Metal ops across 16 layers.

---

## Gate-by-Gate Summary

| Gate | Requirement | `r2s4` Result | Verdict |
| --- | --- | --- | --- |
| Sustained decode speedup | ≥10% lower median, ≥5/6 pairs, GPU + wall | -11.47% wall (6/6), -11.91% GPU | PASS |
| Cold TTFT & 27/28/29 prefill | Within +5% of baseline | +0.10% TTFT; -0.27% / +0.25% / -0.34% prefill | PASS |
| Shared & independent throughput | Within -5% of baseline | +3.89% shared (6/6), +4.89% independent (6/6) | PASS |
| Scheduling & residency | 1 cmd/wait/download, 242 disp, 17 blits, 0 added bytes | Exact match (`4,943,388,672` model, `4,949,910,528` HWM) | PASS |
| Memory (RSS) | Peak & post-close within +5% | 0.00% on short/sustained/shared; +3.00% peak / +6.16% noisy post-close on independent | MARGINAL / NOISE |
| Greedy IDs & frozen reference | Exact IDs (incl. `95551`), frozen ≤ `1e-3 + 1e-3*|ref|` | Exact match, `max_frozen_frac = 0.02166` | PASS |
| Full-row FP32 vs CPU | Every row ≤ `1e-4 + 1e-4*|CPU|` (`0` violations) | 4 unique near-zero violations (`max_fraction = 1.08730`) | **FAIL** |
| **Overall NN-08 decision** | **All gates must hold** | **Negative result: preserve `v0.11.2`, retain tests/benchmarks/docs** | **RETAIN `v0.11.2`** |
