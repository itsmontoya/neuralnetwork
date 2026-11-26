# ADR: Block-Quantized Metal Weight Tensors (`Q8_0` / `Q4_0` / `Q4_1`)

**Status** · Proposed for review in GPU roadmap NN-12; extended with `GatherRows` in NN-19, tiled `simdgroup_float8x8` GEMM in NN-22, and asymmetric `Q4_1` in NN-25.

**Inspected source** · `main` at `3c40a385497af418d81446a5c942e068e36515b1` (updated on `gpu-nn-25`).

## Context

NN-09 through NN-11 delivered `Float16` and `BFloat16` resident weights,
parallel SIMD `RMSNorm`, tied `GatherRows`, and optional `Float16` `KVCache`
storage. At 16-bit weights, Llama 3.2 1B still reads `2.471 GB` per decode
token across 113 right-transposed projections (~25 ms DRAM bandwidth floor on
Apple M3's 100 GB/s unified memory).

Weight-only block quantization (`Q8_0`, `Q4_0`, and `Q4_1`) cuts projection
weight storage and DRAM traffic by **~3.76×** (`Q8_0`, `1.0625` bytes/param),
**~7.11×** (`Q4_0`, `0.5625` bytes/param), and **6.40×** (`Q4_1`, `0.625`
bytes/param) relative to FP32 while keeping activations and dot-product
accumulation in FP32. Both target consumers benefit:

- `llama` can load or host-quantize linear weights into `Q8_0`/`Q4_0`/`Q4_1`
  for dramatically lower memory footprint and higher `M=1` decode throughput.
- `onnx` can map block-quantized linear operators (such as `MatMulNBits`) onto
  the same uniform `Scope.MatMulRightTranspose` API.

## Decision

### 1. Extending `gpu.DType` with `Q8_0`, `Q4_0`, and `Q4_1`

Extend `gpu.DType` in [`gpu/dtype.go`](../gpu/dtype.go) with explicit
block-quantized discriminants:

| Constant | String | `Bytes()` | `BlockSize()` | `BlockBytes()` | `IsQuantized()` | Compression vs FP32 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `Float32` | `"float32"` | `4` | `1` | `4` | `false` | `1.00×` |
| `Float16` | `"float16"` | `2` | `1` | `2` | `false` | `2.00×` |
| `BFloat16` | `"bfloat16"` | `2` | `1` | `2` | `false` | `2.00×` |
| `Q8_0` | `"q8_0"` | `0` | `32` | `34` | `true` | `3.76×` (`128/34`) |
| `Q4_0` | `"q4_0"` | `0` | `32` | `18` | `true` | `7.11×` (`128/18`) |
| `Q4_1` | `"q4_1"` | `0` | `32` | `20` | `true` | `6.40×` (`128/20`) |

`DType.Bytes()` returns `0` for block-quantized types because they have no
integer byte width per element. `DType.BlockSize()` and `DType.BlockBytes()`
report exact block geometry for all valid dtypes (`1` element per block for
scalar dtypes; `32` elements per block for `Q8_0`, `Q4_0`, and `Q4_1`).

### 2. Exact Block Layouts (GGUF-Compatible)

Every block encodes `QuantBlockSize = 32` contiguous elements along the inner
(last) dimension `K` of a rank-2 row-major `[N, K]` weight matrix.

#### `Q8_0` (`QuantBlockBytesQ8_0 = 34` bytes per 32 elements)

| Byte offset | Type | Description |
| --- | --- | --- |
| `0..1` | little-endian `uint16` | IEEE-754 binary16 scale `d` (`Float16`) |
| `2..33` | `32 × int8` | Signed 8-bit quantized weights `q[0..31]` |

- **Dequantization:** for `i` in `0..31`:
  `value[i] = Float16ToFloat32(d) * float32(int8(q[i]))`.
- **Host quantization (`QuantizeQ8_0`):** for each 32-element block:
  - If any element is `NaN`, encode `d = Float32ToFloat16(NaN)` (`0x7e00`) and
    `q[i] = 0`.
  - Otherwise let `maxAbs = max_i |x[i]|`. If `maxAbs == 0`, encode `d = 0x0000`
    and `q[i] = 0`.
  - Otherwise let `dBits = Float32ToFloat16(maxAbs / 127.0)` and
    `d = Float16ToFloat32(dBits)`. If `d == 0` or `!isFinite(d)`, store `dBits`
    and `q[i] = 0`; otherwise `inv = 1.0 / d` and
    `q[i] = int8(clamp(math.RoundToEven(float64(x[i] * inv)), -127, 127))`.

#### `Q4_0` (`QuantBlockBytesQ4_0 = 18` bytes per 32 elements)

| Byte offset | Type | Description |
| --- | --- | --- |
| `0..1` | little-endian `uint16` | IEEE-754 binary16 scale `d` (`Float16`) |
| `2..17` | `16 × uint8` | Packed 4-bit unsigned nibbles `qs[0..15]`, offset by `-8` |

- **Exact nibble ordering (GGUF split-half layout):**
  - Low nibble of `qs[j]` (`qs[j] & 0x0F`) stores element `j` (`0..15`):
    `q[j] = int(qs[j] & 0x0F) - 8` (range `-8..+7`).
  - High nibble of `qs[j]` (`qs[j] >> 4`) stores element `j + 16` (`16..31`):
    `q[j + 16] = int(qs[j] >> 4) - 8` (range `-8..+7`).
- **Dequantization:** for `j` in `0..15`:
  - `value[j] = Float16ToFloat32(d) * float32(int(qs[j] & 0x0F) - 8)`
  - `value[j + 16] = Float16ToFloat32(d) * float32(int(qs[j] >> 4) - 8)`
- **Host quantization (`QuantizeQ4_0`):** for each 32-element block:
  - If any element is `NaN`, encode `d = Float32ToFloat16(NaN)` (`0x7e00`) and
    `qs[j] = 0x88` (all zero offsets).
  - Otherwise find the element `maxVal` with largest magnitude `|maxVal|`. If
    `maxVal == 0`, encode `d = 0x0000` and `qs[j] = 0x88`.
  - Otherwise let `dBits = Float32ToFloat16(maxVal / -8.0)` and
    `d = Float16ToFloat32(dBits)`. If `d == 0` or `!isFinite(d)`, store `dBits`
    and `qs[j] = 0x88`; otherwise `inv = 1.0 / d` and for each element `i`:
    `u[i] = uint8(clamp(math.RoundToEven(float64(x[i] * inv)) + 8, 0, 15))`,
    packing `qs[j] = u[j] | (u[j+16] << 4)` for `j = 0..15`.

#### `Q4_1` (`QuantBlockBytesQ4_1 = 20` bytes per 32 elements)

| Byte offset | Type | Description |
| --- | --- | --- |
| `0..1` | little-endian `uint16` | IEEE-754 binary16 scale `d` (`Float16`) |
| `2..3` | little-endian `uint16` | IEEE-754 binary16 minimum `m` (`Float16`) |
| `4..19` | `16 × uint8` | Packed 4-bit unsigned nibbles `qs[0..15]` (range `0..15`) |

- **Exact nibble ordering (GGUF split-half layout):**
  - Low nibble of `qs[j]` (`qs[j] & 0x0F`) stores element `j` (`0..15`), unsigned `0..15`.
  - High nibble of `qs[j]` (`qs[j] >> 4`) stores element `j + 16` (`16..31`), unsigned `0..15`.
- **Dequantization:** for `j` in `0..15`:
  - `value[j] = Float16ToFloat32(d) * float32(qs[j] & 0x0F) + Float16ToFloat32(m)`
  - `value[j + 16] = Float16ToFloat32(d) * float32(qs[j] >> 4) + Float16ToFloat32(m)`
- **Host quantization (`QuantizeQ4_1`):** for each 32-element block:
  - If any element is `NaN`, encode `d = Float32ToFloat16(NaN)` (`0x7e00`),
    `m = Float32ToFloat16(NaN)` (`0x7e00`), and `qs[j] = 0`.
  - Otherwise find `minVal = min_i x[i]` and `maxVal = max_i x[i]`, encode
    `mBits = Float32ToFloat16(minVal)`, `m = Float16ToFloat32(mBits)`.
  - If `maxVal == minVal`, encode `d = 0x0000`, `m = mBits`, and `qs[j] = 0`.
  - Otherwise let `dBits = Float32ToFloat16(float32((float64(maxVal) - float64(minVal)) / 15.0))`
    and `d = Float16ToFloat32(dBits)`. If `d == 0` or `!isFinite(d)` or `!isFinite(m)`,
    store `dBits`, `mBits`, and `qs[j] = 0`; otherwise `inv = 1.0 / d` and for
    each element `i`:
    `u[i] = uint8(clamp(math.RoundToEven(float64((x[i] - m) * inv)), 0, 15))`,
    packing `qs[j] = u[j] | (u[j+16] << 4)` for `j = 0..15`.

#### Alignment Note for Metal Kernels

Both `34` (`Q8_0`) and `18` (`Q4_0`) are multiples of `2`, **not** `4`. Odd
blocks begin at 2-byte-aligned addresses. Metal kernels treat weight buffers as
`const device uchar *`, read 2-byte `half` scales at guaranteed even offsets,
and read `int8`/`uint8` payloads via 2-byte `ushort` loads without misaligned
4-byte or 16-byte casts. `Q4_1` blocks are `20` bytes (a multiple of `4`), so
both 2-byte `half` headers (`d` at `+0`, `m` at `+2`) and 4-byte `uint` payload
chunks (`+4, +8, +12, +16`) are naturally 4-byte aligned.

### 3. Shape, Alignment, and Transfer Contract

- **Tensor construction (`Device.NewTensorWithDType(shape, dtype)`):**
  - Quantized dtypes (`Q8_0`, `Q4_0`, `Q4_1`) require rank-2 shape `[N, K]` with
    `N > 0`, `K > 0`, and `K % 32 == 0`. Invalid rank, non-positive dimensions,
    misaligned `K`, `uint64` overflow, or byte size exceeding
    `min(math.MaxInt, maxBufferLength)` return `ErrInvalid`.
  - Total byte size is `N * (K / 32) * dtype.BlockBytes()`.
- **Host transfers on `Tensor`:**
  - `UploadBytes(values []byte) error` and
    `DownloadBytesInto(destination []byte) error` transfer exact raw byte slices
    (`len == Tensor.Bytes()`) for any valid `DType`, allowing direct zero-repack
    upload of pre-quantized GGUF buffers.
  - `UploadFloat32As(values []float32) error` and
    `DownloadFloat32Into(destination []float32) error` support `Q8_0`, `Q4_0`,
    and `Q4_1` by invoking the deterministic Go quantize/dequantize helpers.
  - `Upload`/`DownloadInto` remain `Float32`-only; `UploadBits`/`DownloadBitsInto`
    remain `Float16`/`BFloat16`-only.
- **Deterministic Go helpers (`gpu/quantize.go`):**
  - `QuantizeQ8_0`, `DequantizeQ8_0`, `QuantizeQ4_0`, `DequantizeQ4_0`,
    `QuantizeQ4_1`, `DequantizeQ4_1`.

### 4. Operation DType Rules

| Operation | `Q8_0` / `Q4_0` / `Q4_1` Support | Notes |
| --- | --- | --- |
| `Scope.MatMulRightTranspose` | **Supported** (`Float32 [M,K] × Q8_0/Q4_0/Q4_1 [N,K]^T → Float32 [M,N]`) | Fused dequantize-on-the-fly with FP32 accumulation (`fastMathEnabled = NO`): `M=1` SIMD GEMV, `M>=8 && N>=8` `simdgroup_float8x8` tiled GEMM (`BK=32`), and scalar fallback for `2 <= M < 8` or `N < 8`. Rejects quantized dtypes as `left` or `destination` and verifies `K % 32 == 0`. |
| `Scope.GatherRows` | **Supported** (`Q* → Float32` dequantizing gather; `Q* → Q*` matching-dtype blit) | Enables single-allocation quantized tied embedding/classifier tables. Rejects cross-quantized or quantized-into-16-bit conversions with `ErrInvalid`. |
| `Scope.Copy` | **Supported** (matching dtype & shape) | Raw byte Metal blit (`Q8_0 → Q8_0`, `Q4_0 → Q4_0`, `Q4_1 → Q4_1`). |
| `Scope.MatMul` (non-transposed) | **Rejected (`ErrInvalid`)** | Blocks pack along the inner axis `K`; non-transposed `[K, N]` would quantize across output columns. |
| `Tensor.View`, `Fill`, `Add`, `Mul`, `Sub`, `Div`, `Scale`, `*Broadcast`, `SiLU`, `SwiGLU`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax*`, `RMSNorm`, `LayerNorm`, `Transpose`, `MatMulLeftTranspose`, `BatchedMatMul*`, `RotatePairs`, `KVCache`, `AppendKV`, `CausalAttention` | **Rejected (`ErrInvalid`)** | Activations, norms, views, and KV cache remain floating-point. |

## Consequences

- Zero breaking changes to existing `Float32`, `Float16`, `BFloat16`, `Q8_0`, or
  `Q4_0` APIs, kernels, ANN Metal (`Sequential`), default CPU, `purego`,
  `CGO_ENABLED=0`, or non-Darwin builds.
- `Scope.MatMulRightTranspose` stays completely uniform across all six weight
  dtypes (`Float32`, `Float16`, `BFloat16`, `Q8_0`, `Q4_0`, `Q4_1`).
- GGUF-compatible `Q8_0`, `Q4_0`, and `Q4_1` layouts allow direct raw byte
  upload via `UploadBytes` as well as host-side quantization via `UploadFloat32As`.
