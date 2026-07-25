# Data Loading, Batching, and Splitting

The `data` package stores supervised examples as paired input and target
matrices. Each row is one sample. Input and target row counts must match.

## Dataset Ownership

`NewDataset` copies the input and target matrices into dataset-owned storage.
`LoadCSV` builds matrices from CSV data and returns a dataset with the same
copying behavior.

Accessors also return copies:

* `Dataset.Inputs`
* `Dataset.Targets`
* `Batch.Inputs`
* `Batch.Targets`

Callers may mutate the original matrices or accessor results without mutating
the stored dataset or batch.

## CSV Loading

`LoadCSV` reads data from an `io.Reader` using `CSVConfig`.

`CSVConfig.InputColumns` is the number of input feature columns at the start of
each data row. `CSVConfig.TargetColumns` is the number of target value columns
after the input columns. Both counts must be positive.

When `CSVConfig.HasHeader` is true, `LoadCSV` reads and discards the first CSV
record before reading data rows.

Every non-blank data row must contain exactly
`InputColumns + TargetColumns` values. Values are trimmed and parsed as
`float32`. Blank records are skipped. Loading fails when there are no data rows,
when a row has the wrong column count, or when a value cannot be parsed.

The returned dataset has shape:

* inputs: `rowCount x InputColumns`
* targets: `rowCount x TargetColumns`

## Batches

`Dataset.Batches` returns mini-batches with copied input and target matrices.
`batchSize` must be positive.

When the random source is nil, batches preserve dataset row order. When the
random source is non-nil, rows are shuffled with that source before batching,
which makes ordering deterministic when callers provide a seeded `*rand.Rand`.

The final batch may be partial when the sample count is not evenly divisible by
`batchSize`.

## Splits

`Dataset.Split` returns copied train and test datasets. `testFraction` must be
greater than `0` and less than `1`.

The test sample count is computed with:

```go
testCount = int(float64(sampleCount) * testFraction)
```

This floors fractional results. The remaining samples become the train split.
Both train and test splits must be non-empty, so very small datasets or extreme
fractions can return an error even when `testFraction` is between `0` and `1`.

When the random source is nil, splitting preserves dataset row order: train rows
come first and test rows follow. When the random source is non-nil, rows are
shuffled with that source before the train/test boundary is applied.

## Aligned Sequence Lengths

`SequenceLengths` owns one positive logical length per padded sequence row.
Construct it with `NewSequenceLengths(steps, values)`. The physical step count
must be positive, the values must be non-empty, and every value must be in the
inclusive range `[1, steps]`.

The constructor copies the caller's integer slice. `Values` returns another
copy, while `ValuesInto` and `SelectRowsInto` copy into caller-owned
destinations without retaining them. The type is immutable after construction;
its nil receiver and zero value are invalid.

`SequenceDataset` keeps inputs, targets, and logical lengths aligned without
changing the existing `Dataset` or `Batch` APIs. Construct it with:

```go
lengths, err := data.NewSequenceLengths(3, []int{2, 3})
if err != nil {
	return err
}

dataset, err := data.NewSequenceDataset(inputs, targets, lengths)
if err != nil {
	return err
}
```

Input and target row counts must match the length count. The flattened input
width must also divide evenly by `steps`, leaving a positive feature count for
each physical step.

`SequenceDataset` owns copies of all three inputs. Its `Inputs`, `Targets`, and
`Lengths` accessors return independent copies. The corresponding `Into`
methods, plus `SelectRowsInto`, support reusable caller-owned destinations.
Row selection preserves index order and supports repeated indexes.

`SequenceDataset.Batches` returns owned `SequenceBatch` values. A nil random
source preserves order; a caller-provided source shuffles inputs, targets, and
lengths with one shared permutation. The final batch may be partial.

`SequenceDataset.Split` applies the same ordered or caller-shuffled index list
to all three values. Train and test results own their selected data, use the
same floored split calculation as `Dataset.Split`, and must both be non-empty.

These types carry supervised row alignment only. They do not reinterpret
padding, mask recurrent computation, or make the existing matrix-only dataset
APIs length-aware. Use them with `model.FitWithLengths` and
`model.SequenceFitConfig`; direct prediction and one-batch training instead
pair `SequenceLengths` with `PredictWithLengths` and
`TrainBatchWithLengths`.

See the [RNN guide](rnn.md) and the runnable
[explicit-length RNN example](../examples/rnn_lengths/main.go) for the complete
last-valid workflow.
