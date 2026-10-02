# Batch-Source Fitting

`FitWithBatchSource` trains from application-owned batches without first
copying a complete corpus into one matrix. Image decoding, augmentation, and
storage remain application responsibilities while the model retains schedules,
callbacks, validation, metrics, and early stopping.

```go
type BatchSource interface {
    Reset(epoch int, random *rand.Rand) error
    Next(ctx context.Context) (inputs, targets *matrix.Matrix, done bool, err error)
}

history, err := network.FitWithBatchSource(ctx, training, model.BatchSourceFitConfig{
    Epochs:           20,
    Random:           rand.New(rand.NewSource(7)),
    Optimizer:        optimizerRule,
    Loss:             loss.MeanSquaredError{},
    ValidationSource: validation,
    Callback:         callback,
})
```

## Lifecycle

For every one-based epoch, fitting performs these steps:

1. Apply the learning-rate schedule.
2. Reset the training source with the configured random source and train every
   yielded batch.
3. Reset the training source again with a nil random source and evaluate final
   epoch training metrics.
4. When configured, reset the independent validation source with a nil random
   source and evaluate it.
5. Record metrics, invoke the callback, and evaluate early stopping.

`Reset` must therefore support more than one pass for an epoch. A non-nil
random source identifies the training pass and lets the source produce a
deterministic shuffle from caller-owned randomness. A nil random source requests
the source's canonical evaluation order. Reset should do bounded setup; `Next`
must honor context cancellation while performing I/O or decoding.

`Next` returns one non-empty, valid input/target matrix pair with matching row
counts. End of pass is represented only by `done=true`, nil matrices, and a nil
error. An empty pass is rejected. `SizedBatchSource` may additionally report
known sample and batch counts; fitting verifies those counts after each pass.

## Ownership

The source owns matrices returned from `Next`. The model completes forward,
backward, optimizer, loss, and metric work for that batch before invoking
`Next` or `Reset` again. The source may then overwrite or reuse the same matrix
objects. The model does not retain source matrices after
`FitWithBatchSource` returns.

Training and validation sources are reset and consumed independently. Metrics
are weighted by batch row count, so a smaller final batch contributes the
correct proportion. Source, schedule, callback, prediction, loss, accuracy, and
cancellation errors include the applicable epoch, phase, and batch context.
