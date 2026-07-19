# CNN Benchmarks

Captured on July 18, 2026.

## Environment

| Field | Value |
| --- | --- |
| OS | Darwin 25.5.0 |
| Architecture | arm64 |
| CPU | Apple M3 |
| Go version | go1.26.5 |

## Commands

The layer and model packages were measured separately. Each timing is the
median of five runs.

```sh
go test ./layer -run '^$' -bench='Conv2D|MaxPool2D' -benchmem -count=5
go test ./layer -run '^$' -bench='Dense(Forward|Backward)_MediumBatch$' -benchmem -count=5
go test ./model -run '^$' -bench='SequentialTrainBatch_(CNN|SyntheticDense)$' -benchmem -count=5
```

The focused allocation and correctness gates are:

```sh
go test ./layer ./model -run 'Test_(Conv2D|MaxPool2D).*SteadyStateAllocations|Test_Sequential_CNNTrainBatchDoesNotAllocateAfterWarmUp' -count=1
go test ./layer -run 'Conv2D|MaxPool2D' -count=1
```

The full repository verification remains:

```sh
go fmt ./...
go vet ./...
go test ./... -race
```

## Scenarios

The single-image convolution scenario uses an input shape of `[1, 1, 12, 10]`,
four output channels, a `3x3` kernel, unit stride, and unit padding. The batched
multi-channel scenario uses `[8, 3, 16, 12]` with eight output channels and the
same kernel, stride, and padding.

The single-image pooling scenario uses `[1, 1, 12, 10]`. The batched
multi-channel scenario uses `[8, 8, 16, 12]`. Both use a `2x3` window and `2x2`
stride.

The mixed training benchmark uses this model with batch size eight:

```text
[8, 3, 16, 12] -> Conv2D(8, 3x3) -> ReLU -> MaxPool2D(2x3) -> Flatten -> Dense(6)
```

It includes mean-squared-error forward and gradient calculation, full-model
backward propagation, and an SGD update of convolution and dense parameters.
Constructors and cold-path ownership copies are outside timed regions.

## Baseline

The baseline was captured after adding the benchmark fixtures and before the
allocation change.

| Benchmark | Median ns/op | Median B/op | Median allocs/op |
| --- | ---: | ---: | ---: |
| `Conv2DForward/SingleImage` | 6,991 | 32 | 2 |
| `Conv2DForward/BatchMultiChannel` | 510,140 | 32 | 2 |
| `Conv2DBackward/SingleImage` | 8,288 | 32 | 2 |
| `Conv2DBackward/BatchMultiChannel` | 591,448 | 32 | 2 |
| `MaxPool2DForward/SingleImage` | 277.6 | 0 | 0 |
| `MaxPool2DForward/BatchMultiChannel` | 26,847 | 0 | 0 |
| `MaxPool2DBackward/SingleImage` | 74.58 | 0 | 0 |
| `MaxPool2DBackward/BatchMultiChannel` | 3,848 | 0 | 0 |
| `SequentialTrainBatch_CNN` | 1,206,356 | 64 | 4 |

`Conv2D` validation constructed two gradient-label strings on every forward
and backward call. The mixed training path invokes both calls, accounting for
its four allocations. Pooling already reused all required scratch storage and
needed no production change.

## Final

The validation helper now receives static value and gradient labels. This
preserves contextual errors while avoiding hot-path string construction.

| Benchmark | Median ns/op | Median B/op | Median allocs/op |
| --- | ---: | ---: | ---: |
| `Conv2DForward/SingleImage` | 6,965 | 0 | 0 |
| `Conv2DForward/BatchMultiChannel` | 518,232 | 0 | 0 |
| `Conv2DBackward/SingleImage` | 8,344 | 0 | 0 |
| `Conv2DBackward/BatchMultiChannel` | 603,453 | 0 | 0 |
| `MaxPool2DForward/SingleImage` | 279.8 | 0 | 0 |
| `MaxPool2DForward/BatchMultiChannel` | 26,897 | 0 | 0 |
| `MaxPool2DBackward/SingleImage` | 73.67 | 0 | 0 |
| `MaxPool2DBackward/BatchMultiChannel` | 3,780 | 0 | 0 |
| `SequentialTrainBatch_CNN` | 1,208,855 | 0 | 0 |

The accepted steady-state allocation floor is zero for every measured CNN
forward, backward, and mixed training path. Timing variation stayed within
2.1%, so no timing improvement is claimed from this allocation-only change.

## ANN Regression Check

Representative ANN benchmarks remained allocation-free:

| Benchmark | Baseline median ns/op | Final median ns/op | Final B/op | Final allocs/op |
| --- | ---: | ---: | ---: | ---: |
| `DenseForward_MediumBatch` | 151,126 | 153,210 | 0 | 0 |
| `DenseBackward_MediumBatch` | 309,919 | 311,823 | 0 | 0 |
| `SequentialTrainBatch_SyntheticDense` | 758,533 | 757,535 | 0 | 0 |

Timing variation stayed within 1.4%. Existing ANN allocation gates are also
part of the full test suite. The convolution-only validation change does not
affect ANN execution paths.
