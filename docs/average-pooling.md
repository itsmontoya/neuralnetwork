# Average Pooling

The `layer` package provides fixed-window and fixed-output average pooling for
flattened NCHW inputs. Both layers are parameter-free, preserve the channel
count, and operate independently on every channel and batch row.

## Fixed-window pooling

`AveragePool2D` accepts rectangular window and stride dimensions.

```go
inputShape, err := layer.NewSpatialShape(16, 28, 20)
if err != nil {
    return err
}
config, err := layer.NewAveragePool2DConfig(inputShape, 2, 4, 2, 4)
if err != nil {
    return err
}
pooling, err := layer.NewAveragePool2D(config)
```

The layer uses valid windows without padding. It emits only windows that fit
completely within the input, so each output dimension is:

```text
floor((input - window) / stride) + 1
```

Any trailing input rows or columns not covered by a complete window are
ignored. Their input gradient is zero unless another complete, overlapping
window covers them. Backward distributes each output gradient evenly across
the full window and sums contributions from overlapping windows.

## Adaptive pooling

`AdaptiveAveragePool2D` accepts an explicit positive output height and width.
An output shape of `1x1` is global average pooling.

```go
config, err := layer.NewAdaptiveAveragePool2DConfig(inputShape, 1, 1)
if err != nil {
    return err
}
pooling, err := layer.NewAdaptiveAveragePool2D(config)
```

For output position `i`, input size `I`, and output size `O`, the bin is the
half-open interval:

```text
[floor(i * I / O), ceil((i + 1) * I / O))
```

This floor/ceiling rule covers every input position. Bins overlap when a
dimension does not divide evenly. Output dimensions larger than the input are
also supported and use overlapping, non-empty bins. Backward divides each
output gradient by its bin area and accumulates contributions in positions
shared by multiple bins.

## Shape, ownership, and persistence

Each matrix row contains one flattened NCHW sample in channel-major order. The
configuration exposes validated `InputShape` and `OutputShape` values before a
model is built.

Forward and backward results use layer-owned scratch storage. A later call on
the same layer may overwrite a previously returned result, but input matrices
are never used as output storage. Warmed calls with unchanged shapes reuse the
scratch buffers without steady-state allocations.

JSON and binary sequential-model persistence store the complete fixed-window
or adaptive configuration. Inference sessions clone both pooling layer types
with independent scratch storage.
