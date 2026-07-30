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

These copying APIs are the safe default. Use the opt-in view APIs only when
the caller can keep the owner and every overlapping alias unchanged for the
complete synchronous callback.

## Opt-In Data Views

`Dataset.WithView` and `Batch.WithView` publish paired input and target
matrices without copying their values:

```go
err := dataset.WithView(func(view *data.DatasetView) error {
	inputs, err := view.Inputs()
	if err != nil {
		return err
	}
	targets, err := view.Targets()
	if err != nil {
		return err
	}

	return consumeSynchronously(inputs, targets)
})
```

The callback is a scoped borrow, not an ownership transfer. The view and its
matrices expire when the callback returns or panics. Do not retain them,
mutate them, send them to another goroutine, or start asynchronous work from
the callback. Go cannot enforce those read-only rules because existing layers
consume mutable `*matrix.Matrix` values.

`WithRowView(start, end, use)` uses inclusive/exclusive row bounds and aliases
one contiguous window. `WithSelectedRows` classifies an explicit index list:

* `data.ViewOnly` views a strictly contiguous increasing selection and rejects
  every skipped, reordered, or repeated selection.
* `data.ViewOrCopy` still views contiguous rows, but explicitly copies a
  non-contiguous selection. `view.Copied()` reports that fallback.

Ordered traversal views every full or partial batch without copying:

```go
err := dataset.ViewBatches(
	256,
	nil,
	data.ViewOnly,
	func(view *data.DatasetView) error {
		return consumeDatasetView(view)
	},
)
```

A non-nil random source requests shuffled traversal. Shuffling is
non-contiguous, so `ViewOnly` rejects it before consuming random state.
`ViewOrCopy` permits the deterministic selected-row copy:

```go
err := dataset.ViewBatches(
	256,
	rand.New(rand.NewSource(17)),
	data.ViewOrCopy,
	func(view *data.DatasetView) error {
		if !view.Copied() {
			return errors.New("expected shuffled copied fallback")
		}
		return consumeDatasetView(view)
	},
)
```

`ViewSplit` follows the same policy. An ordered split publishes leading train
and trailing test views together without copying. A shuffled split is rejected
by `ViewOnly` or materialized explicitly by `ViewOrCopy`. Existing `Batches`
and `Split` always return independently owned copies and remain preferable
when results must outlive one callback.

Concurrent or reentrant access to the same dataset, batch, view, matrix,
model, layer, or optimizer is unsupported. Independent, non-aliasing objects
may be consumed by separate goroutines. Safe copying accessors remain the
boundary to use when mutation, retention, or shared-owner concurrency is
required.

### Opt-in ordinary fitting

`Sequential.FitWithViews` wraps the unchanged `FitConfig`:

```go
history, err := network.FitWithViews(
	trainingData,
	model.ViewFitConfig{
		FitConfig: model.FitConfig{
			Epochs:         20,
			BatchSize:      256,
			Optimizer:      optimizerRule,
			Loss:           loss.MeanSquaredError{},
			ValidationData: validationData,
		},
		Policy: data.ViewOnly,
	},
)
```

Ordered training uses contiguous batch views. Complete training and validation
evaluation use whole-board views. If `FitConfig.Shuffle` is true, the policy
must be `ViewOrCopy`: training retains the existing reusable selected-row copy
while both complete evaluations remain views. A strict shuffled request fails
during preflight before randomness, schedules, callbacks, layer traversal, or
parameter updates.

Opt-in fitting performs dimension preflight from supported built-in layer
shapes. Use the unchanged `Fit` or `FitWithLengths` operation for a graph
containing a custom layer whose dimensions are not exposed by the
`layer.Layer` interface.

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

The aligned view APIs keep all three values indivisible. `SequenceDatasetView`
exposes `Inputs`, `Targets`, `Lengths`, and `Steps` only inside one callback;
there is no independent `SequenceLengths` view:

```go
err := sequences.ViewBatches(
	192,
	nil,
	data.ViewOnly,
	func(view *data.SequenceDatasetView) error {
		inputs, err := view.Inputs()
		if err != nil {
			return err
		}
		lengths, err := view.Lengths()
		if err != nil {
			return err
		}

		return consumeAlignedSynchronously(inputs, lengths)
	},
)
```

The returned length slice follows the same no-retention, no-mutation, and
no-concurrency rules as the temporary matrices. A retained Go slice header
cannot be revoked after callback expiry, so correct use depends on caller
discipline.

Use `FitWithLengthViews` for the complete opt-in training path:

```go
history, err := network.FitWithLengthViews(
	trainingData,
	model.SequenceViewFitConfig{
		SequenceFitConfig: model.SequenceFitConfig{
			Epochs:         20,
			BatchSize:      192,
			Optimizer:      optimizerRule,
			Loss:           loss.MeanSquaredError{},
			ValidationData: validationData,
		},
		Policy: data.ViewOnly,
	},
)
```

Logical lengths remain aligned through every batch and evaluation.
`GatherLastValid` keeps its existing private length snapshot for backward, so
later backward work never depends on an expired view.

## Performance Guidance

Views remove host-to-host data-boundary copies; they do not remove layer
caches, predictions, gradients, optimizer state, or required Metal
synchronization. Ordered whole-board, batch, split, and evaluation views copy
zero logical input, target, or length bytes. Shuffled `ViewOrCopy` operations
report and account for their selected-row copy honestly.

Use the safe APIs by default. Consider views for large, read-only in-memory
boards when copy traffic is material and the scoped lifetime is easy to
enforce. See [the reproducible benchmark report](../Benchmarks_data_views.md)
and [the full design contract](data-views-design.md).

See the [RNN guide](rnn.md) and the runnable
[explicit-length RNN example](../examples/rnn_lengths/main.go) for the complete
last-valid workflow.
