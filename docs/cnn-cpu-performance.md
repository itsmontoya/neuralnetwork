# CNN CPU Performance

The representative benchmarks exercise the three convolution shapes planned for
the fixed-format OCR model. The end-to-end candidate uses three
`Conv2D -> ReLU -> MaxPool2D` blocks followed by `Flatten -> Dense(163)`.

The checked-in numbers are local development evidence, not a deployment budget.
Run the same benchmarks on the deployment-class Linux host before choosing a
production latency target.

## Environment

* Hardware: Apple M3
* Operating system: macOS 26.5.2
* Architecture: arm64
* Go: 1.26.5

## Commands

```sh
go test ./layer -run '^$' -bench 'Benchmark_Conv2D(Forward|Backward)$' -benchtime=3x -count=1 -benchmem
go test ./model -run '^$' -bench 'Benchmark_SequentialOCRCandidate(Forward|TrainBatch)$' -benchtime=3x -count=1 -benchmem
```

CPU profiles used the second convolution block because it contains enough work
to produce stable samples while isolating the convolution implementation:

```sh
go test ./layer -run '^$' -bench '^Benchmark_Conv2DForward/OCRBatch8Block2$' -benchtime=5x -cpuprofile=forward.pprof
go test ./layer -run '^$' -bench '^Benchmark_Conv2DBackward/OCRBatch8Block2$' -benchtime=3x -cpuprofile=backward.pprof
```

## Baseline

The baseline is commit `ab03a6a`, before the convolution kernel optimization.
Each timing below reports three benchmark iterations after an untimed warm-up.

| Benchmark | ns/op | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| Conv2D forward `[1,1,96,256] -> 16` | 6,419,972 | 0 | 0 |
| Conv2D forward `[8,1,96,256] -> 16` | 50,432,694 | 0 | 0 |
| Conv2D forward `[8,16,48,128] -> 32` | 390,133,070 | 0 | 0 |
| Conv2D forward `[8,32,24,64] -> 64` | 384,261,083 | 0 | 0 |
| Conv2D backward `[1,1,96,256] -> 16` | 6,478,806 | 0 | 0 |
| Conv2D backward `[8,1,96,256] -> 16` | 50,978,986 | 0 | 0 |
| Conv2D backward `[8,16,48,128] -> 32` | 660,547,055 | 0 | 0 |
| Conv2D backward `[8,32,24,64] -> 64` | 429,830,181 | 0 | 0 |
| Candidate forward, batch 1 | 107,113,430 | 0 | 0 |
| Candidate forward, batch 8 | 860,183,472 | 0 | 0 |
| Candidate training, batch 1 | 236,639,736 | 0 | 0 |
| Candidate training, batch 8 | 1,824,499,167 | 16 | 0 |

Forward profiling attributed 96.88% of samples to `Conv2D.forwardInto`.
Backward profiling attributed 78.91% to `Conv2D.backwardInto`; the benchmark's
untimed setup and runtime profiling samples accounted for the remainder. These
profiles identify the scalar convolution loops as the optimization target.

## Optimized results

Commit `92ca88b` reorders the scalar loops to reuse each kernel weight while
walking contiguous spatial rows. It adds no goroutines, worker pool, platform
dependency, or steady-state allocation.

| Benchmark | ns/op | B/op | allocs/op | Baseline / optimized |
| --- | ---: | ---: | ---: | ---: |
| Conv2D forward `[1,1,96,256] -> 16` | 4,791,069 | 0 | 0 | 1.34x |
| Conv2D forward `[8,1,96,256] -> 16` | 26,821,556 | 0 | 0 | 1.88x |
| Conv2D forward `[8,16,48,128] -> 32` | 203,305,070 | 0 | 0 | 1.92x |
| Conv2D forward `[8,32,24,64] -> 64` | 200,605,472 | 0 | 0 | 1.92x |
| Conv2D backward `[1,1,96,256] -> 16` | 6,812,431 | 0 | 0 | 0.95x |
| Conv2D backward `[8,1,96,256] -> 16` | 53,871,597 | 0 | 0 | 0.95x |
| Conv2D backward `[8,16,48,128] -> 32` | 503,219,528 | 0 | 0 | 1.31x |
| Conv2D backward `[8,32,24,64] -> 64` | 382,134,792 | 0 | 0 | 1.12x |
| Candidate forward, batch 1 | 58,426,305 | 0 | 0 | 1.83x |
| Candidate forward, batch 8 | 468,810,722 | 0 | 0 | 1.83x |
| Candidate training, batch 1 | 180,084,806 | 0 | 0 | 1.31x |
| Candidate training, batch 8 | 1,518,114,278 | 0 | 0 | 1.20x |

The small single-input-channel backward cases regress by about 5%, while the
multi-channel training kernels improve by 1.12x to 1.31x. The complete candidate
graph improves more materially because its forward passes dominate this CPU
workload.

Batch-one candidate inference is 58.43 ms per image. Batch-eight inference is
468.81 ms per batch, or 17.06 images per second. Batch-one training is 180.08 ms
per image; batch-eight training is 1.52 seconds per batch, or 5.27 images per
second.

The loop order retains `float32` operations and the forward accumulation order
for each output. Backward accumulation order changes for overlapping input
gradients; convolution value tests and finite-difference gradient tests use the
repository's existing `float32` tolerances. Default, race, and `purego` test
configurations pass. Deployment-class Linux measurements remain an external
release gate.
