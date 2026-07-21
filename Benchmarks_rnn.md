# RNN Benchmarks

Captured on July 20, 2026.

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
go test ./layer -run '^$' -bench='(SimpleRNN|LastStep)' -benchmem -count=5
go test ./model -run '^$' -bench='SequentialTrainBatch_RNN$' -benchmem -count=5
```

The focused allocation and correctness gates are:

```sh
go test ./layer ./model -run 'Test_(SimpleRNN|LastStep).*SteadyStateAllocations|Test_Sequential_RNNTrainBatchDoesNotAllocateAfterWarmUp' -count=1
go test ./layer -run 'SimpleRNN|LastStep' -count=1
go test ./model ./examples/rnn -run 'RNN|LastStep|TemporalOrder' -count=1
```

Representative ANN and CNN regression benchmarks use:

```sh
go test ./layer -run '^$' -bench='Dense(Forward|Backward)_MediumBatch$' -benchmem -count=5
go test ./layer -run '^$' -bench='(Conv2DForward|Conv2DBackward|MaxPool2DForward|MaxPool2DBackward)$' -benchmem -count=5
go test ./model -run '^$' -bench='SequentialTrainBatch_(CNN|SyntheticDense)$' -benchmem -count=5
```

The full repository verification remains:

```sh
go fmt ./...
go vet ./...
go test ./... -race
```

## Scenarios

The `SingleSequence` layer scenario uses batch size one, four steps, three
input features, and five hidden values. `Batched` uses batch size 16, eight
steps, 16 input features, and 32 hidden values. `LongSequence` uses batch size
eight, 32 steps, eight input features, and 16 hidden values. The `LastStep`
benchmarks use the corresponding batch, step, and hidden-feature dimensions.

The mixed training benchmark uses this model with batch size 16:

```text
[16, 8, 16] -> SimpleRNN(32) -> LastStep -> Dense(8)
```

It includes mean-squared-error forward and gradient calculation, full
backpropagation through time, and an SGD update of recurrent and dense
parameters. Constructors, parameter storage, initializer work, and cold-path
ownership copies are outside timed regions.

## Baseline

The baseline was captured after adding the benchmark fixtures and before any
production change.

| Benchmark | Median ns/op | Median B/op | Median allocs/op |
| --- | ---: | ---: | ---: |
| `SimpleRNNForward/SingleSequence` | 274.1 | 0 | 0 |
| `SimpleRNNForward/Batched` | 147,574 | 0 | 0 |
| `SimpleRNNForward/LongSequence` | 79,374 | 0 | 0 |
| `SimpleRNNBackward/SingleSequence` | 347.8 | 0 | 0 |
| `SimpleRNNBackward/Batched` | 348,319 | 0 | 0 |
| `SimpleRNNBackward/LongSequence` | 160,227 | 0 | 0 |
| `LastStepForward/SingleSequence` | 28.32 | 0 | 0 |
| `LastStepForward/Batched` | 302.3 | 0 | 0 |
| `LastStepForward/LongSequence` | 282.5 | 0 | 0 |
| `LastStepBackward/SingleSequence` | 30.25 | 0 | 0 |
| `LastStepBackward/Batched` | 462.2 | 0 | 0 |
| `LastStepBackward/LongSequence` | 420.6 | 0 | 0 |
| `SequentialTrainBatch_RNN` | 507,526 | 0 | 0 |

Every measured steady-state path already reused its matrix pools, growable
value buffers, and fixed-size recurrent workspaces after warm-up. The
allocation measurements therefore found no avoidable allocation and did not
justify a temporal-kernel or production-code change. The explicit constructor
allocation, parameter storage, input ownership copy, hidden-history cache, and
non-aliasing result storage remain unchanged.

## Final

The final measurement was captured after adding the allocation gates and
formatting the benchmark fixtures. Because the baseline allocation floor was
already zero, no production optimization was applied.

| Benchmark | Median ns/op | Median B/op | Median allocs/op |
| --- | ---: | ---: | ---: |
| `SimpleRNNForward/SingleSequence` | 276.9 | 0 | 0 |
| `SimpleRNNForward/Batched` | 148,953 | 0 | 0 |
| `SimpleRNNForward/LongSequence` | 80,101 | 0 | 0 |
| `SimpleRNNBackward/SingleSequence` | 346.9 | 0 | 0 |
| `SimpleRNNBackward/Batched` | 351,279 | 0 | 0 |
| `SimpleRNNBackward/LongSequence` | 161,330 | 0 | 0 |
| `LastStepForward/SingleSequence` | 28.32 | 0 | 0 |
| `LastStepForward/Batched` | 302.4 | 0 | 0 |
| `LastStepForward/LongSequence` | 280.3 | 0 | 0 |
| `LastStepBackward/SingleSequence` | 30.21 | 0 | 0 |
| `LastStepBackward/Batched` | 441.8 | 0 | 0 |
| `LastStepBackward/LongSequence` | 413.5 | 0 | 0 |
| `SequentialTrainBatch_RNN` | 508,110 | 0 | 0 |

The accepted steady-state allocation floor is zero for `SimpleRNN` forward
and backward, `LastStep` forward and backward, and mixed RNN training. Timing
variation ranged from a 4.4% reduction to a 1.1% increase, so no timing change
is claimed.

## ANN and CNN Regression Check

The current medians below are compared with the final measurements recorded in
`Benchmarks_cnn.md` on the same platform.

| Benchmark | Recorded median ns/op | Current median ns/op | Current B/op | Current allocs/op |
| --- | ---: | ---: | ---: | ---: |
| `DenseForward_MediumBatch` | 153,210 | 154,675 | 0 | 0 |
| `DenseBackward_MediumBatch` | 311,823 | 311,991 | 0 | 0 |
| `Conv2DForward/BatchMultiChannel` | 518,232 | 534,521 | 0 | 0 |
| `Conv2DBackward/BatchMultiChannel` | 603,453 | 608,610 | 0 | 0 |
| `MaxPool2DForward/BatchMultiChannel` | 26,897 | 27,886 | 0 | 0 |
| `MaxPool2DBackward/BatchMultiChannel` | 3,780 | 3,826 | 0 | 0 |
| `SequentialTrainBatch_SyntheticDense` | 757,535 | 785,859 | 0 | 0 |
| `SequentialTrainBatch_CNN` | 1,208,855 | 1,212,808 | 0 | 0 |

All representative ANN and CNN paths remain allocation-free. Timing variation
is at most 3.8%. Only benchmark, test, and benchmark documentation files were
added, so no ANN, CNN, or RNN runtime path changed.
