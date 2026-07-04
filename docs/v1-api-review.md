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
| `activation` | `Activation`, `ELU`, `GELU`, `LeakyReLU`, `Linear`, `ReLU`, `Sigmoid`, `Softmax`, `Tanh`, `Name`, and `FromName`. Stable serialization names are `elu`, `gelu`, `leaky_relu`, `linear`, `relu`, `sigmoid`, `softmax`, and `tanh`. |
| `data` | `NewDataset`, `Dataset`, `Batch`, `CSVConfig`, `LoadCSV`, and their exported read, batch, split, CSV-load, and size methods. `Batch` construction remains owned by `Dataset.Batches`. |
| `layer` | `Layer`, `NewDense`, `Dense`, `NewActivation`, `Activation`, `NewDropout`, `Dropout`, `NewBatchNormalization`, `NewBatchNormalizationWithConfig`, `BatchNormalization`, `WeightInitializer`, `ZeroWeights`, `UniformWeights`, `NormalWeights`, `XavierUniformWeights`, and `HeNormalWeights`. |
| `loss` | `Loss`, `MeanSquaredError`, `BinaryCrossEntropy`, and `CategoricalCrossEntropy`. |
| `matrix` | `Matrix`, `New`, `FromSlice`, random initialization helpers, shape/value accessors, copy/clone methods, elementwise operations, scalar operations, multiplication, transpose, sums, and apply helpers. |
| `metric` | `Metric`, `MeanSquaredError`, `NewBinaryAccuracy`, `BinaryAccuracy`, `NewBinaryPrecision`, `BinaryPrecision`, `NewBinaryRecall`, `BinaryRecall`, `NewBinaryF1`, `BinaryF1`, `CategoricalAccuracy`, `CategoricalMacroPrecision`, `CategoricalMacroRecall`, `CategoricalMacroF1`, `NewBinaryConfusionMatrix`, `NewBinaryConfusionMatrixWithThreshold`, `NewCategoricalConfusionMatrix`, `ConfusionMatrix`, and exported confusion-matrix count, accuracy, precision, recall, and F1 methods. |
| `model` | `NewSequential`, `LoadSequential`, `Sequential`, `Sequential.Save`, exported sequential composition, prediction, training, serialization, and training-mode methods, `FitConfig`, `AccuracyFunc`, `FitCallback`, `TrainingHistory`, `EpochMetrics`, `TrainMetrics`, `NewEarlyStopping`, and `EarlyStopping`. |
| `optimizer` | `DefaultAdamBeta1`, `DefaultAdamBeta2`, `DefaultAdamEpsilon`, `NewParameter`, `Parameter`, `Optimizer`, `NewSGD`, `SGD`, `NewMomentum`, `NewMomentumWithCoefficient`, `Momentum`, `NewAdam`, `NewAdamWithConfig`, `Adam`, `LearningRateSchedule`, `NewConstantLearningRate`, `ConstantLearningRate`, `NewStepDecay`, `StepDecay`, `NewExponentialDecay`, `ExponentialDecay`, `Regularizer`, `NewL1`, `L1`, `NewL2WeightDecay`, `L2WeightDecay`, `NewRegularized`, and `Regularized`. |

Post-v1 work may add packages, functions, methods, or implementations, but it
should not break this surface without an explicit maintainer decision.

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
