# Convolutional Neural Networks

The initial convolutional neural network (CNN) path composes with the same
matrix, dataset, model, loss, metric, optimizer, and serialization APIs as a
dense artificial neural network (ANN). It is a pure-Go CPU reference
implementation for pre-flattened image datasets.

For the complete behavioral and public API contract, see
[cnn-design.md](cnn-design.md). This guide focuses on constructing and training
the supported model shape:

```text
flattened image rows -> Conv2D -> ReLU -> MaxPool2D -> Flatten -> Dense -> Softmax
```

## Input Layout

Each `matrix.Matrix` row is one image. An image uses channels-first `CHW` order,
so a batch is logically `NCHW` and physically has shape:

```text
[batch, channels*height*width]
```

The matrix column for an image coordinate is:

```text
channel*Height*Width + row*Width + column
```

For example, a batch of ten RGB images with height 8 and width 6 is a
`[10, 144]` matrix. `data.Dataset` already treats each matrix row as one sample,
so batching preserves image boundaries without a separate tensor type.

`layer.NewSpatialShape` validates positive dimensions and rejects flattened
sizes that overflow `int`:

```go
inputShape, err := layer.NewSpatialShape(3, 8, 6)
if err != nil {
	return err
}
```

Input matrices must have exactly `inputShape.Size()` columns. Image decoding,
augmentation, and directory-backed image datasets are not part of the initial
CNN API; callers prepare flattened rows before constructing a `data.Dataset`.

## Constructing a CNN

The following configuration maps one-channel `5x5` images to two output
classes. Every derived output shape becomes the next spatial layer's input
shape:

```go
inputShape, err := layer.NewSpatialShape(1, 5, 5)
if err != nil {
	return nil, err
}

convConfig, err := layer.NewConv2DConfig(
	inputShape,
	4,       // output channels
	3, 3,    // kernel height and width
	1, 1,    // stride height and width
	1, 1,    // symmetric padding height and width
)
if err != nil {
	return nil, err
}

conv, err := layer.NewConv2D(convConfig, layer.HeNormalWeights(random))
if err != nil {
	return nil, err
}

relu, err := layer.NewActivation(activation.ReLU{})
if err != nil {
	return nil, err
}

poolConfig, err := layer.NewMaxPool2DConfig(
	conv.OutputShape(),
	2, 2, // window height and width
	2, 2, // stride height and width
)
if err != nil {
	return nil, err
}

pool, err := layer.NewMaxPool2D(poolConfig)
if err != nil {
	return nil, err
}

flatten, err := layer.NewFlatten(pool.OutputShape())
if err != nil {
	return nil, err
}

output, err := layer.NewDense(
	flatten.OutputSize(),
	2,
	layer.XavierUniformWeights(random),
)
if err != nil {
	return nil, err
}

softmax, err := layer.NewActivation(activation.Softmax{})
if err != nil {
	return nil, err
}

network, err := model.NewSequential(conv, relu, pool, flatten, output, softmax)
if err != nil {
	return nil, err
}
```

`random` is a caller-owned `*rand.Rand`. Passing a source created with the same
seed gives deterministic initialization. `Conv2D`, `MaxPool2D`, and `Flatten`
do not create or read a random source themselves.

### Convolution Shapes

`Conv2D` performs cross-correlation: kernels are not spatially flipped. For
each spatial dimension, the output uses floor division:

```text
outputHeight = floor((inputHeight + 2*paddingHeight - kernelHeight) / strideHeight) + 1
outputWidth  = floor((inputWidth + 2*paddingWidth - kernelWidth) / strideWidth) + 1
```

Padding is explicit, symmetric, and zero-filled. `paddingHeight` is applied at
the top and bottom; `paddingWidth` is applied at the left and right.

Weights have shape
`[inputChannels*kernelHeight*kernelWidth, outputChannels]`. Biases have shape
`[1, outputChannels]` and are shared across all spatial positions in an output
channel. Backward propagation accumulates summed parameter gradients across the
batch and spatial positions; the loss controls any mean scaling.

### Pooling Shapes

`MaxPool2D` uses valid/no padding and applies each window independently to every
channel:

```text
outputHeight = floor((inputHeight - windowHeight) / strideHeight) + 1
outputWidth  = floor((inputWidth - windowWidth) / strideWidth) + 1
```

Incomplete bottom and right windows are ignored. Ties select the first value in
window-row, then window-column traversal order. Backward propagation adds
gradients when overlapping windows selected the same input position.

### Flatten

`Flatten` is an explicit validating boundary before a dense layer. Values are
already physically flat, so it preserves CHW order and never combines batch
rows. `OutputSize()` equals `InputShape().Size()`.

## Training and Evaluation

CNNs train through `model.Sequential.Fit` without a CNN-specific training
loop. Inputs contain flattened image rows and targets follow the selected loss.
For the two-class Softmax model above, targets are one-hot rows and training can
use categorical cross entropy:

```go
adam, err := optimizer.NewAdam(0.02)
if err != nil {
	return err
}

_, err = network.Fit(trainingData, model.FitConfig{
	Epochs:         80,
	BatchSize:      8,
	Shuffle:        true,
	Random:         rand.New(rand.NewSource(43)),
	Optimizer:      adam,
	Loss:           loss.CategoricalCrossEntropy{},
	ValidationData: validationData,
	Accuracy:       metric.CategoricalAccuracy{}.Value,
})
if err != nil {
	return err
}

predictions, err := network.Predict(inputs)
```

When shuffling is enabled, `FitConfig.Random` is required. Repeating dataset
generation, initialization, and training with equivalent caller-provided
sources produces deterministic results.

The runnable [CNN example](../examples/cnn/main.go) trains on synthetic
horizontal and vertical line images. Synthetic inputs keep the command fast,
reproducible, and independent of downloads. Run it with:

```sh
go run ./examples/cnn
```

With the checked-in seeds, the meaningful stable expectation is that both
training and validation reach successful classification and the canonical
horizontal and vertical inputs map to their matching classes. Exact printed
floating-point values and timing are not compatibility guarantees.

## Serialization

CNN layers use the existing `neuralnetwork.sequential` JSON format at version
`1`:

```go
var document bytes.Buffer
if err := network.Save(&document); err != nil {
	return err
}

restored, err := model.LoadSequential(&document)
if err != nil {
	return err
}
```

The additive layer names are `conv2d`, `max_pool2d`, and `flatten`. Saving
stores spatial configuration plus convolution weights and biases. Loading
restores architecture and parameter values with zero accumulated gradients and
empty forward caches. Optimizer state, training history, and random source
state are not stored.

ANN-only version `1` documents retain their existing encoding and continue to
load. Older readers that do not know the additive CNN layer names reject CNN
documents instead of skipping or substituting layers.

## Matrix Ownership

Valid forward and backward calls do not mutate or retain caller-owned matrix
storage. Results use layer-owned scratch matrices and do not alias the current
argument. A later call on the same layer may reuse and overwrite a previously
returned result, so callers that need to keep a result across calls must clone
it.

`Conv2D` copies its most recent valid input for backward propagation.
`MaxPool2D` stores only selected input positions, and `Flatten` copies values
without reordering them. Backward before a valid forward call returns an error.

## Supported and Deferred Features

| Area | Initial support | Deferred |
| --- | --- | --- |
| Data layout | Batched flattened NCHW/CHW `float32` matrix rows | A general tensor contract |
| Convolution | Multiple input/output channels, rectangular kernels and strides, explicit symmetric zero padding | Dilation, groups, depthwise and transposed convolution, implicit padding modes |
| Pooling | Rectangular valid max pooling, deterministic ties, overlapping-window gradients | Average and global-average pooling |
| Composition | Existing activations, dropout, dense layers, datasets, losses, metrics, optimizers, training, and serialization | Channel-wise spatial batch normalization and spatial dropout |
| Data loading | Caller-prepared flattened image rows | Image decoding, directory datasets, augmentation, and external dataset integrations |
| Runtime | Clear pure-Go CPU reference kernels | SIMD, Metal, goroutine-parallel, and other accelerator-specific kernels |
| Model families | Sequential CNN and ANN composition through `layer.Layer` | RNN layers, sequence state and masking, and automatic-differentiation graphs |

Existing batch normalization remains per flattened matrix feature; it is not
channel-wise spatial batch normalization.
