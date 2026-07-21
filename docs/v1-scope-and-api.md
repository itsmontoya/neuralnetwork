# v1 Scope and Public API Shape

Status: draft for maintainer review.

This document records the planned v1 scope before implementation work begins. It is intentionally limited to package goals, feature boundaries, and public API shape. It does not define production logic.

## Goal

The v1 target is a reusable Go library for training supervised dense feed-forward neural networks with backpropagation.

Runnable commands belong in `examples/` and should demonstrate the library. They should not be the primary product shape.

The runtime dependency target for v1 is the Go standard library.

## In Scope

The first implementation should focus on supervised dense networks:

* Fully connected layers.
* Batched matrix inputs and outputs.
* Common activations: ReLU, Sigmoid, Tanh, Linear, and Softmax.
* Common losses: Mean Squared Error and Cross Entropy.
* Basic optimizers, starting with SGD before Momentum and Adam.
* Deterministic training when callers provide seeded random sources.

## Deferred

The following features should remain out of scope until the dense-network core is stable:

* Convolutional layers.
* Recurrent layers.
* Additional accelerator backends beyond the optional Metal build tag.
* Automatic differentiation graphs.
* Distributed training.

## Package Boundaries

The proposed v1 package layout is:

* `matrix`: dense numeric storage and matrix operations.
* `activation`: activation functions and their backward computations.
* `loss`: loss functions and prediction gradients.
* `optimizer`: parameter update rules.
* `layer`: trainable and non-trainable layer implementations.
* `model`: model composition, prediction, and training orchestration.
* `data`: in-memory supervised datasets and batching helpers.
* `examples`: runnable training examples.

The core packages should be public subpackages rather than Go `internal` packages, because the project goal is a reusable library.

## Top-Level Package Shape

The module path should be `github.com/itsmontoya/neuralnetwork`.

For v1, most behavior should live in focused subpackages. The top-level package can remain minimal and documentation-oriented until the lower-level APIs stabilize.

Expected import style:

* `github.com/itsmontoya/neuralnetwork/model` for sequential model composition.
* `github.com/itsmontoya/neuralnetwork/layer` for dense layers.
* `github.com/itsmontoya/neuralnetwork/activation` for activation choices.
* `github.com/itsmontoya/neuralnetwork/loss` for loss functions.
* `github.com/itsmontoya/neuralnetwork/optimizer` for training updates.
* `github.com/itsmontoya/neuralnetwork/matrix` for numeric primitives.
* `github.com/itsmontoya/neuralnetwork/data` for datasets and batching.

This keeps package responsibilities explicit and avoids forcing all concepts through a broad root package before the API has earned that shape.

## API Design Notes

Public APIs should favor explicit configuration and deterministic behavior:

* Constructors should validate dimensions and configuration where possible.
* User-facing operations should return errors for shape mismatches or invalid configuration.
* Random initialization should accept caller-provided random sources.
* Library code should not write to stdout or stderr during normal operation.
* Examples may print progress because they are commands, not library code.

The exact function signatures should be finalized alongside the first implementation and tests for each package.

## Numeric Type

The v1 implementation uses `float32` as its numeric type. This matches the
Metal compute path directly and keeps matrix storage, layers, losses, metrics,
optimizers, training metrics, data loading, and serialized model values on one
precision boundary.

Gradient checks and floating-point assertions should use f32-sized finite
difference steps and tolerances rather than float64 precision assumptions.

## Convolutional Layers

Convolutional layers remain outside the reviewed dense-network v1 surface. The
initial CNN milestone is additive post-v1 work and does not revise the accepted
ANN contract. In particular, `layer.Layer` and `model.Sequential` continue to
exchange batched 2D matrices, and existing ANN constructors and serialized ANN
models remain compatible.

The implemented post-v1 CNN path represents each image as one flattened matrix
row in channels-first order. Its additive spatial shape, convolution, pooling,
and flattening APIs compose through the stable matrix contract without
introducing a general tensor API. The supported workflow is documented in
[cnn.md](cnn.md); layout, shape, padding, stride, pooling, ownership,
determinism, and serialization decisions are recorded in
[cnn-design.md](cnn-design.md).

## Recurrent Layers

Recurrent layers remain outside the reviewed dense-network v1 surface. The
initial RNN milestone is additive post-v1 work and does not revise the accepted
ANN contract or the implemented CNN contract. `layer.Layer`,
`model.Sequential`, and `data.Dataset` continue to exchange batched 2D
matrices, and existing ANN/CNN constructors and serialized models remain
compatible.

The implemented post-v1 RNN path represents each fixed-length sequence as one
flattened matrix row in time-major order. A stateless `SimpleRNN` returns every
hidden step, and `LastStep` provides an explicit many-to-one boundary before
`Dense`. These additions compose through the stable matrix contract without
introducing a general tensor or replacement sequence-container API. The
supported workflow is documented in [rnn.md](rnn.md); layout, shape, state,
backpropagation, ownership, determinism, and serialization decisions are
recorded in [rnn-design.md](rnn-design.md).
