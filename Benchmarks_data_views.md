# Data View Copy-Pressure Benchmarks

Captured on July 29, 2026. These are before measurements of the existing
copying APIs and fit internals. No production view implementation was present.
Matching after cases will be added without replacing these results.

## Environment

| Field | Value |
| --- | --- |
| Hardware | MacBook Air (Mac15,13), Apple M3, 8 cores, 24 GB |
| OS | macOS 26.5.2 (Darwin 25.5.0, build 25F84) |
| Architecture | arm64 |
| Go version | go1.26.5 |
| CGO | enabled |
| Power | AC power, low-power mode disabled |
| GOMAXPROCS | 1 |
| Source baseline | `8a7b3dc44be11a6236c76304e6edf90d8355d05b` |

The source baseline was clean before the benchmark-only files were added. The
measurements were taken from the resulting benchmark-only worktree; production
code was unchanged.

## Commands

The accepted commands were run exactly as follows:

```sh
GOMAXPROCS=1 go test ./data -run '^$' \
  -bench '^Benchmark_(DataViews|SequenceDataViews)' \
  -benchmem -count=10 -benchtime=500ms

GOMAXPROCS=1 go test ./model -run '^$' \
  -bench '^Benchmark_(FitDataViews|FitSequenceDataViews)' \
  -benchmem -benchtime=3x
```

The data table reports the median of ten results. The model command produces
one result from exactly three iterations, so the model table records that
result directly rather than describing it as a multi-run median.

The tables below come from a second run after `go clean -cache`. Its measured
byte and allocation results matched the first run. Data timing medians differed
by at most 3.31%, and model timings differed by at most 3.44%, confirming that
the commands reproduce the recorded baseline on this platform.

## Shapes and accounting

| Board | Inputs | Targets | Lengths | Batches |
| --- | --- | --- | --- | --- |
| Ordinary | `[4096, 256]` | `[4096, 16]` | n/a | `256` |
| Ordinary partial | `[4100, 256]` | `[4100, 16]` | n/a | `256`, final `4` |
| Wide sequence | `[512, 4096]` | `[512, 8]` | `512` values, `steps=128` | `256`; or `192`, `192`, `128` |
| Wide sequence validation | `[256, 4096]` | `[256, 8]` | `256` values, `steps=128` | whole board |

Matrix bytes are `elementCount*4`. Length bytes are
`lengthCount*(strconv.IntSize/8)`, which is eight bytes per length on this
platform. `logical-B-copied/op` is calculated copy traffic between distinct
data-owned or fit-scratch storage. It excludes indexes, matrix headers,
allocator metadata, layer caches, predictions, gradients, and parameter work.
`B/op` and `allocs/op` are measured by the Go benchmark harness.

Constructor results are compatibility controls because borrowed constructors
are not part of the accepted design. Selected-row results reuse caller-owned
destinations: their zero `B/op` is expected, but the reported logical copy
volume remains nonzero.

## Data baseline medians

| Benchmark | ns/op | B/op | allocs/op | Logical B read | Logical B copied |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Ordinary/Constructor/Ordinary4096x256_4096x16` | 142,975 | 4,456,560 | 5 | 4,456,448 | 4,456,448 |
| `Ordinary/WholeDatasetCopiedAccess/Ordinary4096x256_4096x16` | 140,041 | 4,456,544 | 4 | 4,456,448 | 4,456,448 |
| `Ordinary/WholeBatchCopiedAccess/Ordinary256x256_256x16` | 10,610 | 278,624 | 4 | 278,528 | 278,528 |
| `Ordinary/OrderedBatches/Ordinary4096_Batch256` | 166,059 | 4,491,136 | 82 | 4,456,448 | 4,456,448 |
| `Ordinary/OrderedPartialBatches/Ordinary4100_Batch256` | 167,400 | 4,503,808 | 87 | 4,460,800 | 4,460,800 |
| `Ordinary/ShuffledBatches/Ordinary4096_Batch256` | 199,229 | 4,491,136 | 82 | 4,456,448 | 4,456,448 |
| `Ordinary/OrderedSplit/Ordinary4096_75_25` | 156,408 | 4,489,440 | 11 | 4,456,448 | 4,456,448 |
| `Ordinary/ShuffledSplit/Ordinary4096_75_25` | 186,564 | 4,489,440 | 11 | 4,456,448 | 4,456,448 |
| `Ordinary/SelectedRows/Contiguous256` | 6,425 | 0 | 0 | 278,528 | 278,528 |
| `Ordinary/SelectedRows/ArbitraryRepeated256` | 6,657 | 0 | 0 | 278,528 | 278,528 |
| `Sequence/Constructor/Sequence512x4096_512x8_Lengths512` | 257,419 | 8,409,240 | 7 | 8,409,088 | 8,409,088 |
| `Sequence/WholeDatasetCopiedAccess/Sequence512x4096_512x8_Lengths512` | 251,936 | 8,409,216 | 6 | 8,409,088 | 8,409,088 |
| `Sequence/WholeBatchCopiedAccess/Sequence256x4096_256x8_Lengths256` | 125,316 | 4,204,672 | 6 | 4,204,544 | 4,204,544 |
| `Sequence/BatchLengthCopy/Lengths256` | 470 | 2,080 | 2 | 2,048 | 2,048 |
| `Sequence/OrderedBatches/Sequence512_Batch256` | 420,964 | 8,413,504 | 16 | 8,409,088 | 8,409,088 |
| `Sequence/OrderedPartialBatches/Sequence512_Batch192` | 357,280 | 8,413,664 | 23 | 8,409,088 | 8,409,088 |
| `Sequence/ShuffledBatches/Sequence512_Batch256` | 416,555 | 8,413,504 | 16 | 8,409,088 | 8,409,088 |
| `Sequence/OrderedSplit/Sequence512_75_25` | 430,514 | 8,413,488 | 15 | 8,409,088 | 8,409,088 |
| `Sequence/ShuffledSplit/Sequence512_75_25` | 420,908 | 8,413,488 | 15 | 8,409,088 | 8,409,088 |
| `Sequence/SelectedRows/Contiguous256` | 89,456 | 0 | 0 | 4,204,544 | 4,204,544 |
| `Sequence/SelectedRows/ArbitraryRepeated256` | 95,223 | 0 | 0 | 4,204,544 | 4,204,544 |

## Model baseline

The ordinary graph is `Dense(256, 32) -> Linear -> Dense(32, 16)`. The
length-aware graph is `SimpleRNN([128, 32], 8) -> GatherLastValid ->
Dense(8, 8)`. Both use deterministic finite fixtures, mean-squared error, SGD,
and caller-seeded initializers and shuffle sources. Cold cases construct
operation-owned fit scratch. Warm cases perform an untimed warm-up and reuse
that scratch.

| Benchmark | ns/op | B/op | allocs/op | Logical B read | Logical B copied |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Ordinary/FullTrainingEvaluation/Cold/Ordinary4096x256_4096x16` | 22,250,500 | 4,456,544 | 4 | 4,456,448 | 4,456,448 |
| `Ordinary/FullTrainingEvaluation/Warm/Ordinary4096x256_4096x16` | 21,843,264 | 0 | 0 | 4,456,448 | 4,456,448 |
| `Ordinary/FullValidationEvaluation/Cold/Ordinary4096x256_4096x16` | 22,021,319 | 4,456,544 | 4 | 4,456,448 | 4,456,448 |
| `Ordinary/FullValidationEvaluation/Warm/Ordinary4096x256_4096x16` | 22,088,347 | 0 | 0 | 4,456,448 | 4,456,448 |
| `Ordinary/CompleteEpoch/Ordered/Cold/Train4096_Validation4096_Batch256` | 111,512,264 | 9,224,528 | 14 | 13,369,344 | 13,369,344 |
| `Ordinary/CompleteEpoch/Ordered/Warm/Train4096_Validation4096_Batch256` | 110,346,750 | 16 | 0 | 13,369,344 | 13,369,344 |
| `Ordinary/CompleteEpoch/Shuffled/Cold/Train4096_Validation4096_Batch256` | 111,610,931 | 9,224,544 | 14 | 13,369,344 | 13,369,344 |
| `Ordinary/CompleteEpoch/Shuffled/Warm/Train4096_Validation4096_Batch256` | 110,691,667 | 0 | 0 | 13,369,344 | 13,369,344 |
| `Sequence/FullTrainingEvaluation/Cold/Sequence512x4096_512x8_Lengths512` | 18,723,514 | 8,409,184 | 5 | 8,409,088 | 8,409,088 |
| `Sequence/FullTrainingEvaluation/Warm/Sequence512x4096_512x8_Lengths512` | 18,678,528 | 0 | 0 | 8,409,088 | 8,409,088 |
| `Sequence/FullValidationEvaluation/Cold/Sequence256x4096_256x8_Lengths256` | 9,282,458 | 4,204,640 | 5 | 4,204,544 | 4,204,544 |
| `Sequence/FullValidationEvaluation/Warm/Sequence256x4096_256x8_Lengths256` | 9,245,208 | 0 | 0 | 4,204,544 | 4,204,544 |
| `Sequence/CompleteEpoch/Ordered/Cold/Train512_Validation256_Batch192` | 70,167,361 | 17,872,816 | 21 | 21,022,720 | 21,022,720 |
| `Sequence/CompleteEpoch/Ordered/Warm/Train512_Validation256_Batch192` | 69,505,889 | 0 | 0 | 21,022,720 | 21,022,720 |
| `Sequence/CompleteEpoch/Shuffled/Cold/Train512_Validation256_Batch256` | 70,120,875 | 16,822,608 | 17 | 21,022,720 | 21,022,720 |
| `Sequence/CompleteEpoch/Shuffled/Warm/Train512_Validation256_Batch256` | 69,360,264 | 0 | 0 | 21,022,720 | 21,022,720 |

## Interpretation

The wide sequence board copies 8,409,088 logical bytes for construction,
whole access, every complete batch traversal, and every split. A single
256-row aligned batch copy moves 4,204,544 logical bytes.

Warm fit evaluation and complete epochs meet the existing zero-allocation
contract, but allocation reuse does not remove copy traffic. One warmed
ordinary epoch copies 13,369,344 logical data bytes. One warmed length-aware
epoch copies 21,022,720 logical data bytes: the training board once into batch
scratch, the complete training board again for evaluation, and the complete
validation board once. These results establish why allocation count alone is
not the acceptance metric for the view implementation.

No after measurements or performance claims are included in this baseline.
