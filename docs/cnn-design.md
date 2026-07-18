# CNN Architecture and Public Contract

Status: proposed contract for maintainer review.

This document freezes the public and behavioral contract for the initial
convolutional neural network path. The milestone is additive to the reviewed
dense-network v1 API. It does not change `layer.Layer`, `model.Sequential`,
`data.Dataset`, losses, metrics, optimizers, or the physical representation of
`matrix.Matrix`.

## Milestone

The supported model shape is:

```text
flattened image rows -> Conv2D -> ReLU -> MaxPool2D -> Flatten -> Dense -> output activation
```

Callers construct, train, evaluate, save, and load that model through the
existing APIs. Each implementation is a pure-Go CPU reference path. Correctness
and deterministic behavior take priority over kernel optimization.

## Logical and Physical Layout

A matrix row is one image. Images use logical channels-first `CHW` order, so a
batched matrix is logically `NCHW`, with the matrix row serving as `N`. The
physical matrix shape is:

```text
[batch, channels*height*width]
```

Within a row, the flattened column for `(channel, height, width)` is:

```text
channel*Height*Width + height*Width + width
```

Convolution and pooling outputs use the same formula with their output shape.
The batch dimension is never included in this calculation. `data.Dataset`
continues to store one sample per row, so its existing batching behavior
preserves complete images without knowing about the logical spatial shape.

Spatial dimensions and every intermediate product must fit in an `int`.
Validation detects overflow before multiplication or allocation. Channels,
height, and width are positive; matrix inputs must have exactly
`SpatialShape.Size()` columns.

## Public API

All new symbols are in package `layer`. Constructors return errors with `layer`
context when dimensions, dependencies, or derived shapes are invalid. The zero
values of the new shape, configuration, and layer types are invalid. Valid shape
and configuration values are immutable after construction.

### SpatialShape

```go
func NewSpatialShape(channels, height, width int) (shape SpatialShape, err error)

type SpatialShape struct { /* unexported fields */ }

func (s SpatialShape) Channels() (channels int)
func (s SpatialShape) Height() (height int)
func (s SpatialShape) Width() (width int)
func (s SpatialShape) Size() (size int)
```

`NewSpatialShape` validates positive dimensions and computes
`channels*height*width` with overflow checks. `SpatialShape` contains only
comparable value fields, so callers use Go's `==` and `!=` operators for exact
shape equality. No public offset helper is added; the single CHW formula above
is the public indexing contract.

### Conv2DConfig and Conv2D

```go
func NewConv2DConfig(
	inputShape SpatialShape,
	outputChannels, kernelHeight, kernelWidth int,
	strideHeight, strideWidth int,
	paddingHeight, paddingWidth int,
) (config Conv2DConfig, err error)

type Conv2DConfig struct { /* unexported fields */ }

func (c Conv2DConfig) InputShape() (shape SpatialShape)
func (c Conv2DConfig) OutputShape() (shape SpatialShape)
func (c Conv2DConfig) OutputChannels() (channels int)
func (c Conv2DConfig) KernelHeight() (height int)
func (c Conv2DConfig) KernelWidth() (width int)
func (c Conv2DConfig) StrideHeight() (height int)
func (c Conv2DConfig) StrideWidth() (width int)
func (c Conv2DConfig) PaddingHeight() (height int)
func (c Conv2DConfig) PaddingWidth() (width int)

func NewConv2D(config Conv2DConfig, initializer WeightInitializer) (out *Conv2D, err error)

type Conv2D struct { /* unexported fields */ }

func (c *Conv2D) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error)
func (c *Conv2D) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error)
func (c *Conv2D) Config() (config Conv2DConfig)
func (c *Conv2D) InputShape() (shape SpatialShape)
func (c *Conv2D) OutputShape() (shape SpatialShape)
func (c *Conv2D) Weights() (weights *optimizer.Parameter)
func (c *Conv2D) Biases() (biases *optimizer.Parameter)
func (c *Conv2D) Parameters() (parameters []*optimizer.Parameter)
func (c *Conv2D) AppendParameters(parameters []*optimizer.Parameter) (out []*optimizer.Parameter)
func (c *Conv2D) ResetGradients() (err error)
```

The constructor revalidates the configuration because a caller can still pass
its invalid zero value. It rejects a nil initializer and an initializer result
with an invalid or unexpected shape. `Parameters` and `AppendParameters` use
stable weight-then-bias order, matching `Dense` and existing sequential
parameter discovery.

### MaxPool2DConfig and MaxPool2D

```go
func NewMaxPool2DConfig(
	inputShape SpatialShape,
	windowHeight, windowWidth int,
	strideHeight, strideWidth int,
) (config MaxPool2DConfig, err error)

type MaxPool2DConfig struct { /* unexported fields */ }

func (c MaxPool2DConfig) InputShape() (shape SpatialShape)
func (c MaxPool2DConfig) OutputShape() (shape SpatialShape)
func (c MaxPool2DConfig) WindowHeight() (height int)
func (c MaxPool2DConfig) WindowWidth() (width int)
func (c MaxPool2DConfig) StrideHeight() (height int)
func (c MaxPool2DConfig) StrideWidth() (width int)

func NewMaxPool2D(config MaxPool2DConfig) (out *MaxPool2D, err error)

type MaxPool2D struct { /* unexported fields */ }

func (m *MaxPool2D) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error)
func (m *MaxPool2D) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error)
func (m *MaxPool2D) Config() (config MaxPool2DConfig)
func (m *MaxPool2D) InputShape() (shape SpatialShape)
func (m *MaxPool2D) OutputShape() (shape SpatialShape)
```

`NewMaxPool2D` revalidates its configuration, including its invalid zero value.
Pooling is parameter-free and does not expose parameter or training-mode
methods.

### Flatten

```go
func NewFlatten(inputShape SpatialShape) (out *Flatten, err error)

type Flatten struct { /* unexported fields */ }

func (f *Flatten) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error)
func (f *Flatten) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error)
func (f *Flatten) InputShape() (shape SpatialShape)
func (f *Flatten) OutputSize() (size int)
```

`OutputSize()` equals `InputShape().Size()`. `Flatten` exposes no general reshape
operation and has no parameters or training mode.

## Convolution Semantics

`Conv2D` performs cross-correlation, the operation conventionally called
convolution by neural network libraries. It does not spatially flip kernels.
It supports multiple input and output channels, rectangular kernels, positive
rectangular strides, and explicit symmetric zero padding.

For each spatial dimension, output size uses floor division:

```text
outputHeight = floor((inputHeight + 2*paddingHeight - kernelHeight) / strideHeight) + 1
outputWidth  = floor((inputWidth + 2*paddingWidth - kernelWidth) / strideWidth) + 1
```

Output channels, kernel dimensions, and stride dimensions must be positive.
Padding dimensions must be non-negative. The padded size calculation must not
overflow, the kernel must fit within the padded input, and every output
dimension and the flattened output size must be positive and representable.

Weights have matrix shape:

```text
[inputChannels*kernelHeight*kernelWidth, outputChannels]
```

A weight row for `(inputChannel, kernelRow, kernelColumn)` is:

```text
(inputChannel*kernelHeight + kernelRow)*kernelWidth + kernelColumn
```

Each column is one output channel. Biases have shape `[1, outputChannels]` and
are shared by every batch item and spatial position in that output channel.
The output column for `(outputChannel, outputRow, outputColumn)` follows the CHW
formula.

`NewConv2D` invokes `WeightInitializer` with:

```text
fanIn  = inputChannels*kernelHeight*kernelWidth
fanOut = outputChannels
```

These values are both the initializer's existing matrix dimensions and the
stored weight shape. Biases start at zero. Randomness is entirely controlled by
the initializer supplied by the caller; convolution does not seed, retain, or
read any hidden random source. Repeating construction with equivalent seeded
initializers produces equivalent parameters.

`Forward` accepts `[batch, inputShape.Size()]` and returns
`[batch, outputShape.Size()]`. It copies the last valid input into layer-owned
cache for `Backward`; mutating the caller's input after `Forward` cannot change
the gradient calculation. A later valid forward pass replaces that cache.

`Backward` requires a preceding valid forward pass and a gradient with the most
recent batch size and `outputShape.Size()` columns. It returns an input-shaped
gradient and accumulates weight and bias gradients across calls. Accumulation
sums across batch items and spatial positions without applying a mean. Losses
continue to control scaling through the output gradient they provide.

## Max-Pooling Semantics

`MaxPool2D` applies each rectangular window independently to every batch item
and channel. It uses valid/no padding. Output dimensions are:

```text
outputHeight = floor((inputHeight - windowHeight) / strideHeight) + 1
outputWidth  = floor((inputWidth - windowWidth) / strideWidth) + 1
```

Window and stride dimensions must be positive, each window must fit its input
dimension, and every derived dimension and flattened size must be positive and
representable. Floor behavior intentionally ignores incomplete windows at the
bottom and right edges.

Forward traversal visits kernel rows first and kernel columns second. When
values tie for a maximum, the first visited input position wins. Forward caches
only the selected flat input position for each output value; it does not retain
caller-owned input storage.

`Backward` before a valid `Forward` returns an error. Otherwise the gradient
must match the most recent batch size and output shape. Each output gradient is
routed to its cached selected input position. Input gradients start at zero,
and gradients are added when overlapping windows selected the same input
position.

## Flatten and Matrix Ownership

`Flatten` is a validating semantic boundary. Physical values are already in
the required order, so it does not reorder values or combine batch rows.
Forward validates `[batch, inputShape.Size()]`; backward validates the most
recent batch size and `OutputSize()` and returns the same numeric shape.
Backward before a valid forward pass returns an error.

Valid `Forward` and `Backward` calls copy values into layer-owned scratch
matrices. The returned matrix never aliases the argument, and the layer does not
retain caller-owned matrix storage. This matches existing built-in layers and
allows adjacent sequential layers to use independent scratch matrices. As with
other layer scratch results, a later call on the same layer may reuse and
overwrite a previously returned matrix; callers that need longer ownership
must clone it.

Convolution and pooling outputs and backward results follow the same
non-aliasing scratch-result rule.

## Interoperability

The existing `layer.Layer` matrix contract remains unchanged. Existing
elementwise activation and dropout layers can operate directly on flattened
spatial values. Existing batch normalization remains per flattened feature; it
is not channel-wise spatial batch normalization. `Flatten` makes the boundary
before `Dense` explicit without changing physical storage.

The initial data path is `data.Dataset` populated with pre-flattened image rows.
Losses and metrics see the final matrix output and require no CNN-specific
changes. `Conv2D` exposes `optimizer.Parameter` values and participates in the
private sequential parameter enumeration interfaces; pooling and flattening
contribute no parameters.

## Serialization Compatibility

CNN support extends the existing `neuralnetwork.sequential` JSON format at
version `1`; it does not introduce version `2`. The additive layer type names
are `conv2d`, `max_pool2d`, and `flatten`. Their documents store the input shape
and constructor configuration. `conv2d` also stores weights and biases in the
layout defined above. Output shapes are derived and validated rather than
serialized.

Serialization stores architecture and parameter values only. It does not store
accumulated gradients, optimizer state, scratch matrices, argmax positions,
cached inputs, the last batch size, or other forward-pass state. A loaded
`Conv2D` therefore has zero gradients and requires a new forward pass before
backward, like a newly constructed layer.

CNN layer records add only constructor inputs and trainable values to the
existing version `1` layer object:

* Every CNN layer stores `input_channels`, `input_height`, and `input_width`.
* `conv2d` stores `output_channels`, kernel, stride, and padding dimensions plus
  `weights` and `biases` in stable weight-then-bias order.
* `max_pool2d` stores window and stride dimensions.
* `flatten` needs no fields beyond its input shape.

Zero padding dimensions may be omitted by JSON's zero-value encoding and load
as zero. Output shapes remain derived from validated constructor configuration
and are not stored.

Compatibility is:

* Existing ANN-only version `1` documents remain byte-stable when saved and
  continue to load with the extended reader.
* Existing ANN programs and public APIs remain source-compatible.
* An older version `1` reader can still read ANN-only documents. When given a
  CNN document, it rejects the unknown CNN layer type; it cannot load that
  model and must not silently substitute or skip the layer.
* Unsupported custom layers and unknown serialized layer types continue to
  fail with layer-index context.

Keeping version `1` treats new layer variants as additive vocabulary while
preserving the meaning of every existing field and layer. A future incompatible
schema or changed meaning requires a new format version and explicit migration
handling.

## Supported and Deferred Features

The initial milestone supports:

* Batched, flattened NCHW input in row-major `float32` matrices.
* Multi-channel `Conv2D` cross-correlation with multiple output channels,
  rectangular kernels, rectangular stride, and symmetric explicit zero
  padding.
* Rectangular valid `MaxPool2D` with deterministic ties and overlapping-window
  gradient accumulation.
* A validating, order-preserving `Flatten` adapter.
* Composition with existing activations, dropout, dense layers, datasets,
  losses, metrics, optimizers, training, and sequential serialization.
* A clear pure-Go CPU reference implementation with caller-controlled
  initialization randomness.

The following remain deferred:

* Dilation, grouped or depthwise convolution, transposed convolution, and
  implicit `same` or other padding modes.
* Average pooling, global average pooling, and spatial dropout.
* Channel-wise spatial batch normalization.
* Image decoding, directory-backed image datasets, augmentation, and external
  dataset integrations.
* SIMD, Metal, parallel, or other accelerator-specific convolution kernels.
* A generic tensor abstraction or any change to the stable `layer.Layer`
  matrix contract.
* RNN representations and layers, sequence masking or state, and a general
  automatic-differentiation graph.
