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
go test ./layer -run '^$' -bench 'Benchmark_Conv2D(Forward|Backward)$' -benchtime=1x -count=1 -benchmem
go test ./model -run '^$' -bench 'Benchmark_SequentialOCRCandidate(Forward|TrainBatch)$' -benchtime=1x -count=1 -benchmem
```

CPU profiles used the second convolution block because it contains enough work
to produce stable samples while isolating the convolution implementation:

```sh
go test ./layer -run '^$' -bench '^Benchmark_Conv2DForward/OCRBatch8Block2$' -benchtime=5x -cpuprofile=forward.pprof
go test ./layer -run '^$' -bench '^Benchmark_Conv2DBackward/OCRBatch8Block2$' -benchtime=3x -cpuprofile=backward.pprof
```

## Baseline

The baseline is commit `6346baf`, before the convolution kernel optimization.
Each timing below is a single benchmark iteration after an untimed warm-up.

| Benchmark | ns/op | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| Conv2D forward `[1,1,96,256] -> 16` | 8,143,459 | 0 | 0 |
| Conv2D forward `[8,1,96,256] -> 16` | 47,350,166 | 0 | 0 |
| Conv2D forward `[8,16,48,128] -> 32` | 377,349,500 | 0 | 0 |
| Conv2D forward `[8,32,24,64] -> 64` | 372,444,666 | 0 | 0 |
| Conv2D backward `[1,1,96,256] -> 16` | 6,268,334 | 0 | 0 |
| Conv2D backward `[8,1,96,256] -> 16` | 50,029,041 | 0 | 0 |
| Conv2D backward `[8,16,48,128] -> 32` | 602,270,083 | 0 | 0 |
| Conv2D backward `[8,32,24,64] -> 64` | 459,474,875 | 0 | 0 |
| Candidate forward, batch 1 | 103,407,459 | 0 | 0 |
| Candidate forward, batch 8 | 852,622,458 | 0 | 0 |
| Candidate training, batch 1 | 229,722,417 | 0 | 0 |
| Candidate training, batch 8 | 1,811,335,792 | 0 | 0 |

Forward profiling attributed 96.88% of samples to `Conv2D.forwardInto`.
Backward profiling attributed 78.91% to `Conv2D.backwardInto`; the benchmark's
untimed setup and runtime profiling samples accounted for the remainder. These
profiles identify the scalar convolution loops as the optimization target.
