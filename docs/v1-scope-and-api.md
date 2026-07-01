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
* GPU acceleration.
* Automatic differentiation graphs.
* Distributed training.
* `float32` support, unless a measured need appears after v1.

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

The v1 implementation should use `float64` as its default numeric type. This
keeps the API simple while preserving enough precision for deterministic tests
and gradient checking.

`float32` support is deferred until a measured need justifies the additional API
and test surface.

A later complexity pass confirmed that numeric precision is part of the public
API across matrix storage, layer inputs and outputs, losses, metrics,
optimizers, training metrics, data loading, and serialized model values.
Supporting `float32` would require either a parallel type family or generics
through those package boundaries, so it remains deferred until benchmarks or
real workloads justify the broader surface.

## Convolutional Layers

Convolutional layers remain out of scope for the dense-network v1 API. The core
layer contract currently accepts batched 2D matrices, while convolution support
needs an explicit tensor or image representation, channel ordering, padding and
stride semantics, pooling or flattening behavior, and serialization fields for
those shapes.

The dense-network baseline is reliable enough to revisit this after v1, but
adding convolutional layers now would expand the public API beyond the current
scope instead of fitting into the existing abstractions.
