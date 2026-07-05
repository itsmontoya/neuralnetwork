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
`float64`. Blank records are skipped. Loading fails when there are no data rows,
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
