package layer_test

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_CloneForInference_SharesParametersAndOwnsRuntimeState(t *testing.T) {
	var (
		dense            *layer.Dense
		denseLayer       layer.Layer
		denseClone       *layer.Dense
		shape            layer.SpatialShape
		convConfig       layer.Conv2DConfig
		conv             *layer.Conv2D
		convLayer        layer.Layer
		convClone        *layer.Conv2D
		batchNorm        *layer.BatchNormalization
		batchNormLayer   layer.Layer
		batchNormClone   *layer.BatchNormalization
		batchNorm2D      *layer.BatchNormalization2D
		batchNorm2DLayer layer.Layer
		batchNorm2DClone *layer.BatchNormalization2D
		err              error
	)

	if dense, err = layer.NewDense(2, 3, layer.ZeroWeights); err != nil {
		t.Fatalf("NewDense returned error: %v", err)
	}
	if denseLayer, err = layer.CloneForInference(dense); err != nil {
		t.Fatalf("dense CloneForInference returned error: %v", err)
	}
	denseClone = denseLayer.(*layer.Dense)
	if denseClone == dense || denseClone.Weights() != dense.Weights() || denseClone.Biases() != dense.Biases() {
		t.Fatal("dense clone did not own runtime state and share parameters")
	}

	if shape, err = layer.NewSpatialShape(1, 2, 2); err != nil {
		t.Fatalf("NewSpatialShape returned error: %v", err)
	}
	if convConfig, err = layer.NewConv2DConfig(shape, 2, 1, 1, 1, 1, 0, 0); err != nil {
		t.Fatalf("NewConv2DConfig returned error: %v", err)
	}
	if conv, err = layer.NewConv2D(convConfig, layer.ZeroWeights); err != nil {
		t.Fatalf("NewConv2D returned error: %v", err)
	}
	if convLayer, err = layer.CloneForInference(conv); err != nil {
		t.Fatalf("conv CloneForInference returned error: %v", err)
	}
	convClone = convLayer.(*layer.Conv2D)
	if convClone == conv || convClone.Weights() != conv.Weights() || convClone.Biases() != conv.Biases() {
		t.Fatal("conv clone did not own runtime state and share parameters")
	}

	if batchNorm, err = layer.NewBatchNormalization(2); err != nil {
		t.Fatalf("NewBatchNormalization returned error: %v", err)
	}
	if batchNormLayer, err = layer.CloneForInference(batchNorm); err != nil {
		t.Fatalf("batch normalization CloneForInference returned error: %v", err)
	}
	batchNormClone = batchNormLayer.(*layer.BatchNormalization)
	if batchNormClone == batchNorm || batchNormClone.Gamma() != batchNorm.Gamma() ||
		batchNormClone.RunningMean() != batchNorm.RunningMean() || batchNormClone.Training() {
		t.Fatal("batch normalization clone did not share state in evaluation mode")
	}

	if batchNorm2D, err = layer.NewBatchNormalization2D(layer.BatchNormalization2DConfig{
		InputShape: shape,
		Momentum:   0.9,
		Epsilon:    1e-5,
	}); err != nil {
		t.Fatalf("NewBatchNormalization2D returned error: %v", err)
	}
	if batchNorm2DLayer, err = layer.CloneForInference(batchNorm2D); err != nil {
		t.Fatalf("batch normalization2d CloneForInference returned error: %v", err)
	}
	batchNorm2DClone = batchNorm2DLayer.(*layer.BatchNormalization2D)
	if batchNorm2DClone == batchNorm2D || batchNorm2DClone.Gamma() != batchNorm2D.Gamma() ||
		batchNorm2DClone.RunningMean() != batchNorm2D.RunningMean() || batchNorm2DClone.Training() {
		t.Fatal("batch normalization2d clone did not share state in evaluation mode")
	}
}

func Test_CloneForInference_SupportsRuntimeOnlyLayers(t *testing.T) {
	var (
		shape           layer.SpatialShape
		sequenceShape   layer.SequenceShape
		segmented       *activation.SegmentedSoftmax
		activationLayer *layer.Activation
		dropout         *layer.Dropout
		layers          []layer.Layer
		current         layer.Layer
		clone           layer.Layer
		err             error
	)

	if shape, err = layer.NewSpatialShape(1, 2, 2); err != nil {
		t.Fatalf("NewSpatialShape returned error: %v", err)
	}
	if sequenceShape, err = layer.NewSequenceShape(2, 2); err != nil {
		t.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if segmented, err = activation.NewSegmentedSoftmax(2, 2); err != nil {
		t.Fatalf("NewSegmentedSoftmax returned error: %v", err)
	}
	if activationLayer, err = layer.NewActivation(segmented); err != nil {
		t.Fatalf("NewActivation returned error: %v", err)
	}
	if dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(7))); err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}
	layers = []layer.Layer{
		activationLayer,
		dropout,
		mustInferenceFlatten(t, shape),
		mustInferenceGatherLastValid(t, sequenceShape),
		mustInferenceLastStep(t, sequenceShape),
		mustInferenceMaxPool2D(t, shape),
		mustInferenceSimpleRNN(t, sequenceShape),
	}

	for _, current = range layers {
		if clone, err = layer.CloneForInference(current); err != nil {
			t.Fatalf("CloneForInference(%T) returned error: %v", current, err)
		}
		if clone == current {
			t.Fatalf("CloneForInference(%T) returned source", current)
		}
	}
	if clone.(*layer.SimpleRNN).InputWeights() != layers[len(layers)-1].(*layer.SimpleRNN).InputWeights() {
		t.Fatal("simple RNN clone did not share input weights")
	}
}

func Test_CloneForInference_RejectsUnsupportedLayer(t *testing.T) {
	var (
		clone layer.Layer
		err   error
	)

	clone, err = layer.CloneForInference(inferenceCloneUnsupportedLayer{})
	if err == nil {
		t.Fatal("CloneForInference error = nil, want unsupported error")
	}
	if clone != nil {
		t.Fatal("CloneForInference returned clone on error")
	}
	if !strings.Contains(err.Error(), "does not support") {
		t.Fatalf("CloneForInference error = %q, want unsupported type", err)
	}
}

type inferenceCloneUnsupportedLayer struct{}

func (inferenceCloneUnsupportedLayer) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	return input, nil
}

func (inferenceCloneUnsupportedLayer) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	return outputGradient, nil
}

func mustInferenceFlatten(tb testing.TB, shape layer.SpatialShape) (flatten *layer.Flatten) {
	var err error

	tb.Helper()
	if flatten, err = layer.NewFlatten(shape); err != nil {
		tb.Fatalf("NewFlatten returned error: %v", err)
	}
	return flatten
}

func mustInferenceGatherLastValid(tb testing.TB, shape layer.SequenceShape) (gather *layer.GatherLastValid) {
	var err error

	tb.Helper()
	if gather, err = layer.NewGatherLastValid(shape); err != nil {
		tb.Fatalf("NewGatherLastValid returned error: %v", err)
	}
	return gather
}

func mustInferenceLastStep(tb testing.TB, shape layer.SequenceShape) (last *layer.LastStep) {
	var err error

	tb.Helper()
	if last, err = layer.NewLastStep(shape); err != nil {
		tb.Fatalf("NewLastStep returned error: %v", err)
	}
	return last
}

func mustInferenceMaxPool2D(tb testing.TB, shape layer.SpatialShape) (pool *layer.MaxPool2D) {
	var (
		config layer.MaxPool2DConfig
		err    error
	)

	tb.Helper()
	if config, err = layer.NewMaxPool2DConfig(shape, 2, 2, 2, 2); err != nil {
		tb.Fatalf("NewMaxPool2DConfig returned error: %v", err)
	}
	if pool, err = layer.NewMaxPool2D(config); err != nil {
		tb.Fatalf("NewMaxPool2D returned error: %v", err)
	}
	return pool
}

func mustInferenceSimpleRNN(tb testing.TB, shape layer.SequenceShape) (recurrent *layer.SimpleRNN) {
	var (
		config layer.SimpleRNNConfig
		err    error
	)

	tb.Helper()
	if config, err = layer.NewSimpleRNNConfig(shape, 3); err != nil {
		tb.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	if recurrent, err = layer.NewSimpleRNN(config, layer.ZeroWeights, layer.ZeroWeights); err != nil {
		tb.Fatalf("NewSimpleRNN returned error: %v", err)
	}
	return recurrent
}
