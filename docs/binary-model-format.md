# Binary Model Format

The binary model format is intended for production artifacts where compact
weights, bounded decoding, and corruption detection matter. JSON remains the
inspectable compatibility and debugging format.

## API

```go
checksum, err := network.SaveBinary(writer)
loaded, verifiedChecksum, err := model.LoadSequentialBinary(reader)
```

Both operations stream through `io.Writer` or `io.Reader`; they do not buffer
the complete encoded artifact. The returned `ModelChecksum` formats as lowercase
hexadecimal with its `String` method so a sidecar manifest can reference it.

## Version 1 contract

The byte layout is:

```text
8 bytes   magic: "NNSEQBIN"
2 bytes   format version, unsigned little-endian: 1
variable  model payload
32 bytes  SHA-256 of the exact model payload
```

Payload integers are canonical unsigned varints. Counts and dimensions are
non-negative. Configuration and parameter `float32` values use IEEE 754 binary32
bits in little-endian byte order. Strings use a varint byte length followed by
UTF-8 bytes. Matrices use a presence byte, row and column varints, then
row-major raw `float32` values.

Layers and their fields are written in model order and a fixed version-specific
field order. The payload preserves every layer configuration, trainable
parameter, running mean, and running variance supported by JSON version 1. Like
JSON, it does not contain optimizer state, gradients, transient forward state,
training history, callbacks, schedules, or application preprocessing metadata.

Encoding is deterministic for the same model state. The SHA-256 checksum covers
the architecture and evaluation parameters, but not the magic, version, or
checksum footer itself.

## Compatibility and limits

Binary format versions are independent of the existing JSON document version.
Readers accept only binary version 1 and reject unknown versions; an incompatible
layout change requires a new binary version. Adding a layer or field therefore
does not silently reinterpret an older binary contract. JSON `Save` and
`LoadSequential` remain unchanged.

Version 1 applies decoder limits before proportional allocation:

* At most 4,096 layers.
* At most 4,096 configured activation widths.
* At most 256 bytes per encoded string.
* At most 134,217,728 cumulative matrix values (512 MiB of raw `float32` data).

Matrix bodies are consumed in 32 KiB chunks and value storage grows only as
bytes arrive. Truncated, corrupt, trailing, oversized, malformed, and
unknown-version artifacts return contextual errors without constructing a
model. A checksum is returned only after the complete artifact is verified.

## Representative benchmark

Measured on an Apple M3, macOS 26.5.2, arm64, Go 1.26.5. The model is the
three-block OCR candidate graph documented in [CNN CPU Performance](cnn-cpu-performance.md).
Each load timing reports three iterations after setup.

```sh
go test ./model -run '^$' -bench '^Benchmark_SequentialArtifactLoad$' -benchtime=3x -count=1 -benchmem
```

| Format | Artifact bytes | Load ns/op | B/op | allocs/op |
| --- | ---: | ---: | ---: | ---: |
| JSON | 96,092,889 | 671,322,403 | 399,310,605 | 254 |
| Binary | 16,118,002 | 16,696,208 | 137,048,730 | 531 |

The binary artifact is 83.2% smaller and loads about 40.2x faster in this local
measurement. Allocation count is higher because the streaming decoder appends
value chunks, while total allocated bytes fall by 65.7%.
