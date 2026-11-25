# ADR: Phase 3 Generic GPU Primitives for Resident ONNX Execution

**Status** · Proposed for review in GPU roadmap NN-13.

**Inspected source** · `main` at `32f83b827469b384e3a2546dc9ef4a5482e2c479` (`v0.12.1`).

## Context

Phase 1 (`v0.11.x`) and Phase 2 (`v0.12.1`) established explicit Metal execution
in `github.com/itsmontoya/neuralnetwork/gpu` for Llama-style decoder workloads
(`Float32`/`Float16`/`BFloat16`/`Q8_0`/`Q4_0` weights, rank-2 projections,
`RMSNorm`, `GatherRows`, `RotatePairs`, `SiLU`/`SwiGLU`, `KVCache`, and
`CausalAttention`).

Integrating `v0.12.1` into `github.com/itsmontoya/onnx` (`ONNX-01`) showed
strong resident throughput on supported subgraphs (`ResidentFFN` 1.53×–3.61×
over CPU), but exposed a small set of missing generic primitives that forced
standard ONNX MLPs, residual blocks, and Transformers to break `gpu.Scope`
execution and round-trip through CPU:

- Contiguous metadata-only shape transforms (`Reshape`, `Flatten`, `Squeeze`,
  `Unsqueeze`).
- Standard activations (`ReLU`, `Sigmoid`, `Tanh`) and `Softmax` / `SoftmaxAxis`.
- Elementwise `Sub`, `Div`, `Scale`, and multidirectional broadcast binary ops.
- `LayerNorm` with optional bias and reduced-precision affine parameters.
- General contiguous `Transpose`, rank-2 `MatMulLeftTranspose`, and 3D+
  broadcast-batched `MatMul` / `BatchedMatMulRightTranspose`.

Phase 3 (`NN-13`–`NN-17`) implements these generic primitives inside
`neuralnetwork/gpu` while preserving 100% backward compatibility with `v0.12.1`
and `github.com/itsmontoya/llama`.

## Decision

### 1. Zero-Copy `Tensor.View` & Storage-Level Lease/Alias Tracking (`NN-13`)

#### Public API

- `func (t *Tensor) View(shape []int64) (*Tensor, error)`
  - Returns a zero-copy contiguous row-major `*Tensor` sharing the underlying
    device allocation with a new immutable `shape`.
  - Supported for non-quantized dtypes (`Float32`, `Float16`, `BFloat16`).
  - Rejects quantized tensors (`Q8_0`, `Q4_0`), empty/rank-0 shapes, non-positive
    dimensions, overflow/buffer-limit shapes, or mismatched element counts with
    `ErrInvalid`.
  - Rejects `nil` or closed source tensors with `ErrClosed`.
  - Clones the caller's `shape` slice; `Shape()` returns an independent copy,
    and `Shape()`, `DType()`, and `Bytes()` remain readable after `Close()`.
  - Permitted while the underlying storage holds an active `Scope` lease or host
    transfer so callers can chain `MatMul → View → Add` inside one recording
    without `ErrBusy`.

#### Internal Storage & Handle Split

Split the previous monolithic `tensorState` into two internal structs:

- `tensorStorage`: owns the shared device allocation and synchronization state
  (`device`, `dtype`, `count`, `bytes`, `buffer`, `readers`, `writer`,
  `hostAccess`, `initialized`, `liveHandles`).
- `tensorState`: owns per-handle metadata (`storage *tensorStorage`, immutable
  `shape []int64`, `closed bool`).

#### Lease, Alias, Initialization, and Lifetime Invariants

- **Storage-keyed leases:** `scopeState.leases` is keyed by `*tensorStorage`
  rather than `*tensorState`.
- **Alias rejection:** `acquireLocked` compares `source.state.storage == destination.state.storage`.
  Two distinct views of the same allocation cannot be passed as source and
  destination to any `Scope` operation (`ErrInvalid`), nor can competing scopes
  write through sibling views (`ErrBusy`). Multiple read inputs referencing
  sibling views within the same scope are allowed and share one read lease.
- **Shared initialization:** Uploading through or completing a `Scope` write to
  any handle sets `storage.initialized = true` for all sibling views. Within the
  same `Scope`, an earlier write through one view satisfies read initialization
  for sibling views (`storage.writer == scope`).
- **Host transfer exclusion:** Host transfers (`Upload*`, `Download*`) check and
  acquire `storage.hostAccess` against `storage.readers`, `storage.writer`, and
  `storage.hostAccess`, rejecting concurrent transfers or transfers against a
  scope-leased sibling view with `ErrBusy`.
- **Independent handle `Close()` and deferred release:** `Tensor.Close()` is
  idempotent, marks only that handle `closed`, and decrements
  `storage.liveHandles` once. `Device` counts one child per live
  `tensorStorage` (in addition to open scopes). The native `MTLBuffer` is
  released and `device.children` decremented only when `storage.liveHandles == 0`,
  `!storage.hostAccess`, `storage.readers == 0`, and `storage.writer == nil`.
  Closing a parent tensor leaves open sibling views valid for subsequent host
  transfers and scopes; `Device.Close()` returns `ErrBusy` until all tensors,
  views, transfers, and scopes finish and close.

### 2. Standard Activations & Parallel SIMD `Softmax` (`NN-14`)

Planned `Scope` methods:

- `Scope.ReLU(input, destination *Tensor) error`
- `Scope.Sigmoid(input, destination *Tensor) error`
- `Scope.Tanh(input, destination *Tensor) error`
- `Scope.Softmax(input, destination *Tensor) error`
- `Scope.SoftmaxAxis(input, destination *Tensor, axis int) error`

Contracts:

- **Validation:** `Float32` operands with identical shapes, `count <= math.MaxUint32`,
  same device, and destination not aliasing `input` (including via `Tensor.View`).
  `Softmax` normalizes along the last axis (`len(shape)-1`); `SoftmaxAxis`
  normalizes along `axis` with `0 <= axis < len(shape)` (`ErrInvalid` otherwise).
- **Numerics:** Compiled lazily via `pthread_once` with `[options setFastMathEnabled:NO]`.
  - `ReLU(x)`: `isnan(x) ? x : max(x, 0.0f)` (propagates `NaN`).
  - `Sigmoid(x)`: stable split branch (`x >= 0 ? 1/(1+exp(-x)) : exp(x)/(1+exp(x))`;
    `+Inf → 1`, `-Inf → 0`, `NaN → NaN`).
  - `Tanh(x)`: saturates cleanly to `±1` for large finite `|x|` (`±Inf → ±1`,
    `NaN → NaN`, no `Inf/Inf` overflow).
  - `Softmax` / `SoftmaxAxis`: subtracts row maximum before `exp` in FP32.
    Contiguous last-axis (`inner == 1`) uses 1 threadgroup per row with
    `simd_max`/`simd_sum`, disjoint `shared[34]` threadgroup broadcast slots
    (avoiding WAR races), `float4` loads/stores when `axisSize % 4 == 0`, and
    scalar fallback. General axes (`inner > 1`) use a strided `[outer, axisSize, inner]`
    path. `axisSize == 1` yields `1.0` for finite inputs.

### 3. Elementwise Arithmetic & Multidirectional Broadcast Binary Ops (`NN-15`)

Planned `Scope` methods:

- `Scope.Sub(left, right, destination *Tensor) error`
- `Scope.Div(left, right, destination *Tensor) error`
- `Scope.Scale(input, destination *Tensor, scale float32) error`
- `Scope.AddBroadcast(left, right, destination *Tensor) error`
- `Scope.SubBroadcast(left, right, destination *Tensor) error`
- `Scope.MulBroadcast(left, right, destination *Tensor) error`
- `Scope.DivBroadcast(left, right, destination *Tensor) error`

Contracts:

- **Backward compatibility:** Existing `Scope.Add` and `Scope.Mul` continue to
  require identical shapes and reject mismatched shapes with `ErrInvalid`.
- **Same-shape ops:** `Sub`, `Div`, and `Scale` follow the same `Float32`,
  identical-shape, `count <= math.MaxUint32`, and non-aliasing destination rules
  as `Add` and `SiLU`.
- **Broadcast ops:** Follow ONNX/NumPy right-aligned broadcasting for rank 1..6.
  Reject incompatible dimensions or rank > 6 with `ErrInvalid`. `destination.Shape()`
  must equal the exact broadcast shape; all operands must be `Float32` on the
  same device with `count <= math.MaxUint32`; destination must not alias either
  input (including via `Tensor.View`).
- **Kernel paths (`fastMathEnabled = NO`):** Fast path 1 for identical shapes
  (contiguous elementwise), fast path 2 for last-axis bias (`[..., N] op [1,...,1,N]`),
  fast path 3 for scalar `[1]` broadcast, and a coalesced general rank-6 stride
  path. IEEE-754 FP32 arithmetic (`1/0 → +Inf`, `-1/0 → -Inf`, `0/0 → NaN`,
  `NaN` propagation).

### 4. Parallel SIMD `LayerNorm` (`NN-16`)

Planned `Scope` method:

- `Scope.LayerNorm(input, weight, bias, destination *Tensor, epsilon float32) error`

Contracts:

- **Validation:** Normalizes along the last axis `width = shape[len(shape)-1]`
  for any rank ≥ 1 `Float32` `input` and matching `Float32` `destination`
  (`count <= math.MaxUint32`). `weight` is required (rank-1 `[width]`,
  `Float32`, `Float16`, or `BFloat16`). `bias` is optional (`nil` omits bias
  addition; when non-nil, rank-1 `[width]` with `Float32`, `Float16`, or
  `BFloat16`, independently of `weight` dtype). `epsilon` must be finite and
  positive (`epsilon > 0 && epsilon <= math.MaxFloat32`). Quantized weights or
  biases, non-`Float32` input/destination, or destination aliasing `input`,
  `weight`, or `bias` (including via `Tensor.View`) return `ErrInvalid`.
- **Numerics & kernel (`fastMathEnabled = NO`):** Matches ONNX opset-17
  `LayerNormalization` using a numerically stable two-pass FP32 reduction
  (Pass 1: mean `μ = sum(x)/W`; Pass 2: variance `σ² = sum((x - μ)²)/W`, avoiding
  catastrophic cancellation on large-offset rows; Pass 3: normalize, scale, and
  optional bias). Uses 1 threadgroup per row with `simd_sum` and disjoint
  `shared[34]` broadcast slots, `float4` fast path when `width % 4 == 0`, and
  exact scalar fallback.

### 5. General `Transpose`, `MatMulLeftTranspose`, & Batched 3D+ MatMul (`NN-17`)

Planned `Scope` methods:

- `Scope.Transpose(input, destination *Tensor, perm []int) error`
- `Scope.MatMulLeftTranspose(left, right, destination *Tensor) error`
- `Scope.BatchedMatMul(left, right, destination *Tensor) error`
- `Scope.BatchedMatMulRightTranspose(left, right, destination *Tensor) error`

Contracts:

- **`Transpose`:** Permutes axes of any rank 1..6 non-quantized tensor
  (`Float32`, `Float16`, `BFloat16`, matching input/destination dtype). Validates
  `len(perm) == len(input.Shape())`, valid permutation of `0..rank-1`,
  `destination.Shape()[i] == input.Shape()[perm[i]]`, and non-aliasing
  destination (including via `Tensor.View`). Identity permutation may fast-path
  to blit `encodeCopy`; non-identity transposes use 2-byte (`uint16`) and 4-byte
  (`uint32`) Metal kernels.
- **`MatMulLeftTranspose`:** Computes rank-2 `destination[M, N] = left[K, M]^T × right[K, N]`
  for `Float32` `left`/`destination` and `Float32`, `Float16`, or `BFloat16`
  `right` (quantized dtypes rejected with `ErrInvalid`), accumulating in FP32
  with `fastMathEnabled = NO`. Existing `MatMul` and `MatMulRightTranspose`
  remain bit-for-bit untouched.
- **`BatchedMatMul` & `BatchedMatMulRightTranspose`:** Compute
  `left[..., M, K] × right[..., K, N] → destination[..., M, N]` and
  `left[..., M, K] × right[..., N, K]^T → destination[..., M, N]` for rank 2..6
  tensors with right-aligned batch broadcasting (up to 4 batch dimensions).
  Support `Float32` `left`/`destination` and `Float32`, `Float16`, or `BFloat16`
  `right` (quantized dtypes rejected with `ErrInvalid`), FP32 accumulation,
  `fastMathEnabled = NO`, and a 3D Metal grid `(N, M, totalBatches)` using
  precomputed batch strides.

### 6. Backward Compatibility, Portability, and Numerics

- All existing `v0.12.1` `gpu` APIs and their validation, error, and numerical
  contracts remain unchanged.
- Default CPU, `purego`, `CGO_ENABLED=0`, non-Darwin (`GOOS=linux GOARCH=amd64`),
  and ANN Metal (`Sequential`) paths remain compilable, tested, and behaviorally
  unchanged.
- Every new Metal shader compiles lazily via `pthread_once` with
  `[options setFastMathEnabled:NO]` and is verified against independent FP64
  references across normal values, odd/tail shapes, large magnitudes,
  cancellation, `NaN`, and `±Inf`.

## Consequences

- `NN-13` establishes storage-level lease, lifetime, and alias safety before any
  Phase 3 compute kernels are added, so `NN-14`–`NN-17` inherit view alias
  rejection and deferred buffer release automatically.
- `Tensor.View` introduces zero new native buffer allocations and zero new
  Metal shaders.
- Completing `NN-13`–`NN-17` inside `neuralnetwork` allows `ONNX-02` to pin one
  immutable release (`v0.13.0`) and execute full ONNX MLPs, residual networks,
  and Transformer encoder blocks in a single resident `gpu.Scope`.
