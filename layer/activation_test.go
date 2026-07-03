package layer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_ActivationLayer_ImplementsLayer(t *testing.T) {
	var _ layer.Layer = (*layer.Activation)(nil)
}

func Test_NewActivation_ValidatesFunction(t *testing.T) {
	var (
		activationLayer *layer.Activation
		err             error
	)

	activationLayer, err = layer.NewActivation(nil)
	if err == nil {
		t.Fatal("NewActivation error = nil, want error")
	}

	if activationLayer != nil {
		t.Fatal("NewActivation returned activation layer on error")
	}
}

func Test_ActivationLayer_ForwardBackward(t *testing.T) {
	var (
		activationLayer *layer.Activation
		input           *matrix.Matrix
		output          *matrix.Matrix
		outputGradient  *matrix.Matrix
		inputGradient   *matrix.Matrix
		err             error
	)

	activationLayer, err = layer.NewActivation(activation.Sigmoid{})
	if err != nil {
		t.Fatalf("NewActivation returned error: %v", err)
	}

	input = mustMatrix(t, 1, 3, []float64{-1, 0, 2})
	output, err = activationLayer.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	requireMatrixValues(t, output, []float64{
		0.2689414213699951,
		0.5,
		0.8807970779778823,
	})

	err = input.Set(0, 0, 10)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	outputGradient = mustMatrix(t, 1, 3, []float64{1, 2, 3})
	inputGradient, err = activationLayer.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float64{
		0.19661193324148185,
		0.5,
		0.31498075621051985,
	})
}

func Test_ActivationLayer_BackwardRequiresForward(t *testing.T) {
	var (
		activationLayer *layer.Activation
		inputGradient   *matrix.Matrix
		err             error
	)

	activationLayer, err = layer.NewActivation(activation.ReLU{})
	if err != nil {
		t.Fatalf("NewActivation returned error: %v", err)
	}

	inputGradient, err = activationLayer.Backward(mustMatrix(t, 1, 1, []float64{1}))
	if err == nil {
		t.Fatalf("Backward returned gradient %v and nil error, want error", inputGradient)
	}
}
