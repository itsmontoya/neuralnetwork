# ADR: Reduced-Precision Metal Tensors (`Float16` / `BFloat16`)

**Status** · Proposed for review in GPU roadmap NN-09.

**Inspected source** · `main` at `33cae5825c1b6724b0e784eb6293010cbc41d26c`
(`v0.11.3`).

## Context

`github.com/itsmontoya/neuralnetwork/gpu` currently allocates and transfers
contiguous row-major FP32 (`4` bytes/element) tensors only. Profiling Llama 3.2
1B on Apple M3 (`100 GB/s` unified memory) established that single-token (`M=1`)
FP32 Metal decode is bandwidth-bound (`4.943 GB` of weights read per token across
113 projections). Meanwhile, both target consumers require native 16-bit storage:

- `llama` loads native BF16 Hugging Face checkpoints and benefits from FP16/BF16
  weights and FP16 KV caches.
- `onnx` executes graphs with explicit `FLOAT`, `FLOAT16`, and `BFLOAT16`
  tensors.

NN-09 establishes the `gpu.DType` contract, multi-dtype tensor storage, checked
byte sizing, host↔device transfers, deterministic Go conversions, device-to-device
copy, and dtype validation on existing FP32 operations. Compute kernels for
reduced precision are deferred to NN-10 (projections) and NN-11 (RMSNorm,
GatherRows, KV cache, attention, elementwise).

## Decision

### 1. `gpu.DType` Enumeration and Byte Widths

Introduce `DType` (`uint8`) in [`gpu/dtype.go`](../gpu/dtype.go) with explicit
non-zero valid discriminants so the zero value is invalid and never silently
defaults to FP32:

| Constant | String | Element width | Representation |
| --- | --- | --- | --- |
| `Float32` | `"float32"` | `4` bytes | IEEE-754 binary32 (`float32`) |
| `Float16` | `"float16"` | `2` bytes | IEEE-754 binary16 (`uint16`) |
| `BFloat16` | `"bfloat16"` | `2` bytes | Top 16 bits of IEEE-754 binary32 (`uint16`) |

`DType` exposes `Valid() bool`, `Bytes() uint64` (`0` when invalid), and
`String() string` (`"invalid"` when invalid).

All tensors remain contiguous row-major allocations on the creating `Device`.
Checked byte sizing computes `elements * dtype.Bytes()` while guarding against
signed `int64` shape errors, `uint64` overflow, host `math.MaxInt` byte limits,
and the Metal device's `maxBufferLength`.

### 2. Exact `Float16` and `BFloat16` Bit Representations & Rounding

Host and device memory store `Float16` and `BFloat16` elements as contiguous
little-endian `uint16` bit patterns:

- **`Float16` (IEEE-754 binary16):** 1 sign bit, 5 exponent bits (bias `15`),
  10 significand bits.
  - Max finite: `0x7bff` (`65504.0`). Min positive normal: `0x0400` ($2^{-14}$).
    Min positive subnormal: `0x0001` ($2^{-24}$).
  - `Float32ToFloat16(float32) uint16` uses **round-to-nearest-even (RNE)**.
    Values with magnitude $\ge 65520$ (halfway between `65504` and `65536`)
    overflow to signed infinity (`0x7c00` / `0xfc00`). Magnitudes below $2^{-25}$
    underflow to signed zero (`0x0000` / `0x8000`); exact $2^{-25}$ ties to even
    (`±0`), while values strictly above $2^{-25}$ round up to `0x0001` / `0x8001`.
    Subnormal rounding that carries out of bit 9 promotes seamlessly to the
    smallest normal (`0x0400` / `0x8400`). Signed zero and ±Inf are preserved.
    NaNs preserve sign and top 10 payload bits with the quiet NaN bit (`0x0200`)
    forced so no signaling NaN collapses to ±Inf.
  - `Float16ToFloat32(uint16) float32` is exact for all finite values (normals,
    subnormals, signed zero) and ±Inf, and promotes NaNs to quiet binary32 NaNs
    preserving sign and payload bits.

- **`BFloat16`:** 1 sign bit, 8 exponent bits (bias `127`), 7 significand bits.
  - Max finite: `0x7f7f` ($\approx 3.3895314 \times 10^{38}$). Min positive
    normal: `0x0080` ($2^{-126}$). Min positive subnormal: `0x0001` ($2^{-133}$).
  - `Float32ToBFloat16(float32) uint16` uses **round-to-nearest-even (RNE)** on
    the dropped lower 16 bits (`(u + 0x7fff + ((u >> 16) & 1)) >> 16` for all
    non-NaN values). Overflow from max finite rounds to ±Inf (`0x7f80` /
    `0xff80`). Signed zero, ±Inf, and subnormals follow IEEE-754 RNE. NaNs
    preserve sign and top 7 payload bits with the quiet NaN bit (`0x0040`) forced
    so no NaN collapses to ±Inf.
  - `BFloat16ToFloat32(uint16) float32` is exact (`math.Float32frombits(uint32(bits) << 16)`).

### 3. Backward-Compatible Public API Surface

All existing FP32 APIs keep their exact signatures and semantics:

- `Device.NewFloat32(shape []int64) (*Tensor, error)` — unchanged (delegates to
  `NewTensorWithDType(shape, Float32)`).
- `Tensor.Upload([]float32) error` and `Tensor.DownloadInto([]float32) error` —
  unchanged for `Float32` tensors; return `ErrInvalid` on non-`Float32` tensors.
- `Tensor.Shape() []int64` and `Tensor.Close() error` — unchanged.

New minimal typed APIs:

- `Device.NewTensorWithDType(shape []int64, dtype DType) (*Tensor, error)` —
  allocates an uninitialized contiguous row-major tensor of the requested
  `DType`.
- `Tensor.DType() DType` — returns the immutable `DType` (including after
  `Close`; returns invalid `0` on a nil handle).
- `Tensor.Bytes() uint64` — returns the exact byte length (`elements * dtype.Bytes()`),
  including after `Close`.
- `Tensor.UploadBits(values []uint16) error` and
  `Tensor.DownloadBitsInto(destination []uint16) error` — zero-conversion raw
  16-bit transfers for `Float16` and `BFloat16` tensors. Returns `ErrInvalid`
  for `Float32` tensors or length mismatches.
- `Tensor.UploadFloat32As(values []float32) error` and
  `Tensor.DownloadFloat32Into(destination []float32) error` — host-side
  RNE conversion and transfer for any valid `DType` (`Float32` transfers
  directly without intermediate allocation).

### 4. `internal/device` Storage and Transfer Layer

`internal/device` reuses the existing `MTLBuffer` allocation (`MTLResourceStorageModeShared`),
scope resource retention, and `ResourceSnapshot` accounting:

- `Runtime.NewByteBuffer(bytes uint64) (*Buffer, error)` allocates a raw
  byte-sized `MTLBuffer` (`Buffer.Bytes()` returns `bytes`; `Buffer.Count()` is
  `0` so FP32-only `Scope.Encode*` methods cannot accidentally accept a raw
  buffer).
- `Buffer.UploadBytes([]byte)`, `Buffer.DownloadBytes([]byte)`,
  `Buffer.UploadUint16([]uint16)`, and `Buffer.DownloadUint16([]uint16)` copy
  exact buffer lengths via `nn_metal_buffer_upload_bytes` /
  `nn_metal_buffer_download_bytes` (`memcpy` into/out of `[MTLBuffer contents]`).
- `Scope.EncodeCopy(source, destination *Buffer)` validates matching byte
  lengths (`source.bytes == destination.bytes`) and blits `source.bytes`,
  supporting any `DType` including odd 16-bit element counts.
- Existing `Runtime.NewBuffer(count)`, `Buffer.Upload([]float32)`,
  `Buffer.Download([]float32)`, and all ANN Metal paths remain unchanged.

### 5. Operation DType Rules (NN-09 vs Planned NN-10 / NN-11)

In **NN-09**:

- `Scope.Copy(source, destination)` supports `Float32`, `Float16`, and
  `BFloat16` when `source.DType() == destination.DType()` and shapes match;
  mismatched dtypes return `ErrInvalid`.
- All existing compute and KV operations (`Fill`, `GatherRows`, `Add`, `Mul`,
  `SiLU`, `SwiGLU`, `MatMul`, `MatMulRightTranspose`, `RMSNorm`, `RotatePairs`,
  `AppendKV`, `CausalAttention`) explicitly reject non-`Float32` operands with
  `ErrInvalid`.

Planned mixed-precision rules for **NN-10** and **NN-11** (all with FP32
accumulation and `[options setFastMathEnabled:NO]`):

| Operation | NN-10 / NN-11 Planned DType Combinations |
| --- | --- |
| `MatMul`, `MatMulRightTranspose` (NN-10) | `Float32 × Float32 → Float32` (existing, bit-exact preserved); `Float32 × Float16 → Float32`; `Float32 × BFloat16 → Float32` (mixed activation × reduced-precision weight). |
| `RMSNorm` (NN-11) | `Float32` input/output with `Float32`, `Float16`, or `BFloat16` weight; FP32 variance accumulation. |
| `GatherRows` (NN-11) | Matching dtype (`Float32 → Float32`, `Float16 → Float16`, `BFloat16 → BFloat16`) and dequantizing gather (`Float16/BFloat16 → Float32`) for tied embeddings. |
| `KVCache` / `AppendKV` / `CausalAttention` (NN-11) | `Float32` (existing) and optional `Float16` KV storage with FP32 query, score, softmax, and output accumulation. |
| `Add`, `Mul`, `SiLU`, `SwiGLU`, `RotatePairs` (NN-11) | Matching dtype where needed by `llama`/`onnx`, without unrequested kernel matrix bloat. |

## Apple M3 Metal Toolchain & SDK Probe

Probed on this host (`macOS 26.5.2`, Darwin `25.5.0` arm64, macOS SDK `14.4`,
`Apple M3`, `MTLGPUFamilyApple9 = true`, `MTLGPUFamilyMetal3 = true`,
`maxBufferLength = 14,302,248,960` bytes) via runtime
`[MTLDevice newLibraryWithSource:options:error:]` with
`[options setFastMathEnabled:NO]`:

| MSL Construct | Compile Options | Result |
| --- | --- | --- |
| `half`, `half4` | Default language version, `fastMathEnabled = NO` | **PASS** |
| `bfloat`, `bfloat4` | Default language version, `fastMathEnabled = NO` | **PASS** |
| `bfloat`, `bfloat4` | `MTLLanguageVersion3_1`, `fastMathEnabled = NO` | **PASS** |
| `ushort`, `ushort4` bitcast (`as_type<float>(uint(x) << 16)`) | Default language version, `fastMathEnabled = NO` | **PASS** |

Note: `xcrun metal` offline compiler is not installed on this Command Line Tools
host, so runtime `newLibraryWithSource` remains the shader compilation path.
Native MSL `half`/`half4` and `bfloat`/`bfloat4` are both available for NN-10
kernels on Apple M3, with `ushort4` bitcast available as a portable fallback if
older Metal language versions are ever targeted.

## Consequences

- Zero breaking changes to existing FP32 `gpu` callers, ANN Metal (`Sequential`),
  default CPU, `purego`, `CGO_ENABLED=0`, or Linux builds.
- Exact 50% device memory reduction for `Float16` and `BFloat16` weight tensors
  is immediately verifiable via `Tensor.Bytes()` and `internal/device` resource
  snapshots.
- NN-10 can add FP16/BF16 projection kernels directly on top of `Tensor.DType()`
  and `internal/device.Buffer` without changing tensor lifecycle, leasing, or
  host transfer code.
