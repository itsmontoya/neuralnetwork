# v1 API Review

Status: reviewed for v1 stability.

This review tags the public library surface that should remain stable before
post-v1 features are added. Runnable commands under `examples/` and helpers
under `internal/` are not part of the stable library API.

## Package Names

The stable public packages are:

| Package | Role | Review |
| --- | --- | --- |
| `activation` | Stateless activation functions. | Short noun package with no stutter. |
| `data` | In-memory supervised datasets, batching, splitting, and CSV loading. | Short noun package with a clear boundary. |
| `layer` | Layer contracts and implementations for dense, activation, dropout, and batch-normalization layers. | Short noun package with no top-level `neuralnetwork.Layer` wrapper. |
| `loss` | Training losses and prediction gradients. | Short noun package separate from metrics. |
| `matrix` | Dense row-major numeric primitives. | Short noun package for low-level storage and operations. |
| `metric` | Reporting, accuracy, confusion-matrix, precision, recall, and F1 metrics. | Singular package name follows Go package naming convention. |
| `model` | Model composition, prediction, training, and serialization. | Short noun package with `Sequential` as the concrete model type. |
| `optimizer` | Parameter update rules, learning-rate schedules, and regularization helpers. | Short noun package for training updates. |

There is no broad root package API, so the module avoids names such as
`neuralnetwork.Network`. Focused subpackages remain the intended import style.

## Stable Surface

The following exported APIs are tagged as stable for v1:

| Package | Stable APIs |
| --- | --- |
| `activation` | `Activation` with `Forward` and `Backward`; built-ins `ELU`, `GELU`, `LeakyReLU`, `Linear`, `ReLU`, `Sigmoid`, `Softmax`, and `Tanh`; `Name`; and `FromName`. Stable serialization names are `elu`, `gelu`, `leaky_relu`, `linear`, `relu`, `sigmoid`, `softmax`, and `tanh`. |
| `data` | `NewDataset`; `Dataset` with `Inputs`, `Targets`, `SampleCount`, `InputSize`, `TargetSize`, `Batches`, and `Split`; `Batch` with `Inputs`, `Targets`, and `SampleCount`; `CSVConfig` fields `InputColumns`, `TargetColumns`, and `HasHeader`; and `LoadCSV`. `Batch` construction remains owned by `Dataset.Batches`. |
| `layer` | `Layer` with `Forward` and `Backward`; `NewDense`; `Dense` with `Forward`, `Backward`, `InputSize`, `OutputSize`, `Weights`, `Biases`, `Parameters`, and `ResetGradients`; `NewActivation`; `Activation` with `Forward`, `Backward`, and `Function`; `NewDropout`; `Dropout` with `Forward`, `Backward`, `Rate`, `SetTraining`, and `Training`; `NewBatchNormalization`; `NewBatchNormalizationWithConfig`; `BatchNormalization` with `Forward`, `Backward`, `FeatureSize`, `Momentum`, `Epsilon`, `Gamma`, `Beta`, `RunningMean`, `RunningVariance`, `Parameters`, `ResetGradients`, `SetTraining`, and `Training`; `WeightInitializer`; `ZeroWeights`; `UniformWeights`; `NormalWeights`; `XavierUniformWeights`; and `HeNormalWeights`. |
| `loss` | `Loss` with `Value` and `Gradient`; `MeanSquaredError`; `BinaryCrossEntropy`; and `CategoricalCrossEntropy`. |
| `matrix` | `New`; `FromSlice`; `NewRandom`; `NewUniform`; `NewNormal`; `NewXavierUniform`; `NewHeNormal`; and `Matrix` with `Rows`, `Cols`, `Shape`, `Validate`, `Values`, `At`, `Set`, `Fill`, `Clone`, `CopyFrom`, `Add`, `AddInto`, `AddInPlace`, `AddScaledInPlace`, `MultiplyScalarInPlace`, `Subtract`, `MultiplyElements`, `DivideElements`, `AddScalar`, `MultiplyScalar`, `DivideScalar`, `MatMul`, `MatMulInto`, `Transpose`, `TransposeInto`, `RowSums`, `ColumnSums`, `ColumnSumsInto`, `AddRowVectorInPlace`, and `Apply`. |
| `metric` | `Metric` with `Value`; `MeanSquaredError`; `NewBinaryAccuracy`; `BinaryAccuracy`; `NewBinaryPrecision`; `BinaryPrecision`; `NewBinaryRecall`; `BinaryRecall`; `NewBinaryF1`; `BinaryF1`; `CategoricalAccuracy`; `CategoricalMacroPrecision`; `CategoricalMacroRecall`; `CategoricalMacroF1`; `NewBinaryConfusionMatrix`; `NewBinaryConfusionMatrixWithThreshold`; `NewCategoricalConfusionMatrix`; and `ConfusionMatrix` with `ClassCount`, `Total`, `Counts`, `At`, `Accuracy`, `Precision`, `Recall`, `F1`, `MacroPrecision`, `MacroRecall`, and `MacroF1`. |
| `model` | `NewSequential`; `LoadSequential`; `Sequential` with `Add`, `Predict`, `Backward`, `Parameters`, `SetTraining`, `Training`, `TrainBatch`, `Fit`, and `Save`; `FitConfig` fields `Epochs`, `BatchSize`, `Shuffle`, `Random`, `Optimizer`, `LearningRateSchedule`, `EarlyStopping`, `Loss`, `ValidationData`, `Accuracy`, and `Callback`; `AccuracyFunc`; `FitCallback`; `TrainingHistory` field `Epochs`; `EpochMetrics` fields `Epoch`, `Loss`, `ValidationLoss`, `HasValidationLoss`, `Accuracy`, `HasAccuracy`, `ValidationAccuracy`, and `HasValidationAccuracy`; `TrainMetrics` field `Loss`; `NewEarlyStopping`; and `EarlyStopping` with `Patience` and `MinDelta`. |
| `optimizer` | `DefaultAdamBeta1`; `DefaultAdamBeta2`; `DefaultAdamEpsilon`; `NewParameter`; `Parameter` with `Values`, `Gradient`, `AccumulateGradient`, and `ResetGradient`; `Optimizer` with `Update`, `LearningRate`, and `SetLearningRate`; `NewSGD`; `SGD`; `NewMomentum`; `NewMomentumWithCoefficient`; `Momentum`; `NewAdam`; `NewAdamWithConfig`; `Adam`; `LearningRateSchedule` with `LearningRate`; `NewConstantLearningRate`; `ConstantLearningRate`; `NewStepDecay`; `StepDecay`; `NewExponentialDecay`; `ExponentialDecay`; `Regularizer` with `Apply`; `NewL1`; `L1`; `NewL2WeightDecay`; `L2WeightDecay`; `NewRegularized`; and `Regularized`. |

The APIs above include the dropout, batch-normalization, serialization,
early-stopping, learning-rate schedule, regularization, CSV, activation, and
classification metric additions from the current code split. They are accepted
as part of the v1 stable surface rather than labeled as post-v1 additions.

Post-v1 work may add packages, functions, methods, or implementations, but it
should not break this surface without an explicit maintainer decision.

## Additive Post-v1 CNN Surface

The initial CNN milestone adds the following `layer` APIs without revising the
accepted ANN v1 surface above:

| Type | Additive APIs |
| --- | --- |
| Spatial shape | `NewSpatialShape`; `SpatialShape` with `Channels`, `Height`, `Width`, and `Size`. |
| Convolution configuration | `NewConv2DConfig`; `Conv2DConfig` with `InputShape`, `OutputShape`, `OutputChannels`, `KernelHeight`, `KernelWidth`, `StrideHeight`, `StrideWidth`, `PaddingHeight`, and `PaddingWidth`. |
| Convolution layer | `NewConv2D`; `Conv2D` with `Forward`, `Backward`, `Config`, `InputShape`, `OutputShape`, `Weights`, `Biases`, `Parameters`, `AppendParameters`, and `ResetGradients`. |
| Pooling configuration | `NewMaxPool2DConfig`; `MaxPool2DConfig` with `InputShape`, `OutputShape`, `WindowHeight`, `WindowWidth`, `StrideHeight`, and `StrideWidth`. |
| Pooling layer | `NewMaxPool2D`; `MaxPool2D` with `Forward`, `Backward`, `Config`, `InputShape`, and `OutputShape`. |
| Flatten adapter | `NewFlatten`; `Flatten` with `Forward`, `Backward`, `InputShape`, and `OutputSize`. |

These layers retain the v1 `layer.Layer` matrix contract. Spatial values use
flattened channels-first rows, `Conv2D` performs cross-correlation with explicit
symmetric zero padding, `MaxPool2D` uses valid padding, and `Flatten` preserves
physical value order and batch rows. The detailed contract is recorded in
[cnn-design.md](cnn-design.md), with construction and training guidance in
[cnn.md](cnn.md).

The `model.Sequential.Save` and `model.LoadSequential` APIs are unchanged. The
version `1` serialization vocabulary now also accepts `conv2d`, `max_pool2d`,
and `flatten` layer records. ANN-only documents retain their existing encoding;
older readers reject documents containing unknown additive CNN layer types.

## Constructor Review

Constructors validate required dimensions and nil dependencies before returning
usable values. Matrix and dataset constructors copy caller-owned mutable data
where ownership matters. `NewDense` requires an explicit weight initializer so
callers choose deterministic or random initialization deliberately.

Optimizer constructors reject invalid learning rates, schedule settings,
regularization coefficients, and configuration values. Metric, layer, and model
constructors reject nil dependencies instead of deferring failures to the first
training step.

## Shape Errors

Public shape errors include package context and diagnostic dimensions. Matrix
operations report got/want shapes or row and column indexes. Losses and metrics
report prediction and target shapes. Dense layers report the received input or
gradient shape and the expected shape.

## Determinism

Randomness is caller-controlled through `*rand.Rand`:

* Matrix random constructors require a random source.
* Layer initializer helpers close over caller-provided random sources.
* Dropout layers require a random source and expose training-mode control.
* Serialized dropout layers are restored with deterministic local random
  sources.
* Dataset batching and splitting use caller-provided random sources when
  shuffling is requested.
* `FitConfig` requires `Random` when `Shuffle` is enabled.

Library code does not seed or read from hidden global random state.

## Library Output

Library packages do not write to stdout or stderr during normal operation.
Progress reporting is exposed through returned history and `FitCallback`.
Only runnable example commands print output.
