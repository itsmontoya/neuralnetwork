package layer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_Conv2DForwardSteadyStateAllocations(t *testing.T) {
	var (
		convolution *layer.Conv2D
		input       *matrix.Matrix
		err         error
	)

	convolution = benchmarkConv2D(t, 3, 16, 12, 8)
	input = allocationLayerMatrix(t, 8, convolution.InputShape().Size())
	if _, err = convolution.Forward(input); err != nil {
		t.Fatalf("warm-up Forward returned error: %v", err)
	}

	requireMaxAllocs(t, "Conv2D.Forward", 0, func() {
		if allocationLayerResult, err = convolution.Forward(input); err != nil {
			panic(err)
		}
	})
}

func Test_Conv2DBackwardSteadyStateAllocations(t *testing.T) {
	var (
		convolution    *layer.Conv2D
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		err            error
	)

	convolution = benchmarkConv2D(t, 3, 16, 12, 8)
	input = allocationLayerMatrix(t, 8, convolution.InputShape().Size())
	outputGradient = allocationLayerMatrix(t, 8, convolution.OutputShape().Size())
	if _, err = convolution.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if _, err = convolution.Backward(outputGradient); err != nil {
		t.Fatalf("warm-up Backward returned error: %v", err)
	}
	if err = convolution.ResetGradients(); err != nil {
		t.Fatalf("ResetGradients returned error: %v", err)
	}

	requireMaxAllocs(t, "Conv2D.Backward", 0, func() {
		if allocationLayerResult, err = convolution.Backward(outputGradient); err != nil {
			panic(err)
		}
	})
}

func Test_MaxPool2DForwardSteadyStateAllocations(t *testing.T) {
	var (
		pooling *layer.MaxPool2D
		input   *matrix.Matrix
		err     error
	)

	pooling = benchmarkMaxPool2D(t, 8, 16, 12)
	input = allocationLayerMatrix(t, 8, pooling.InputShape().Size())
	if _, err = pooling.Forward(input); err != nil {
		t.Fatalf("warm-up Forward returned error: %v", err)
	}

	requireMaxAllocs(t, "MaxPool2D.Forward", 0, func() {
		if allocationLayerResult, err = pooling.Forward(input); err != nil {
			panic(err)
		}
	})
}

func Test_MaxPool2DBackwardSteadyStateAllocations(t *testing.T) {
	var (
		pooling        *layer.MaxPool2D
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		err            error
	)

	pooling = benchmarkMaxPool2D(t, 8, 16, 12)
	input = allocationLayerMatrix(t, 8, pooling.InputShape().Size())
	outputGradient = allocationLayerMatrix(t, 8, pooling.OutputShape().Size())
	if _, err = pooling.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if _, err = pooling.Backward(outputGradient); err != nil {
		t.Fatalf("warm-up Backward returned error: %v", err)
	}

	requireMaxAllocs(t, "MaxPool2D.Backward", 0, func() {
		if allocationLayerResult, err = pooling.Backward(outputGradient); err != nil {
			panic(err)
		}
	})
}
