package layer_test

import (
	"errors"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

type fallbackActivation struct {
	forwardError  error
	forwardCalls  int
	backwardCalls int
}

func (a *fallbackActivation) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	a.forwardCalls++
	if a.forwardError != nil {
		return nil, a.forwardError
	}

	output, err = input.Clone()
	return output, err
}

func (a *fallbackActivation) Backward(input, outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	a.backwardCalls++
	inputGradient, err = input.MultiplyElements(outputGradient)
	return inputGradient, err
}

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

func Test_ActivationLayer_Function(t *testing.T) {
	var (
		function        activation.Activation
		activationLayer *layer.Activation
		err             error
	)

	function = activation.Sigmoid{}
	activationLayer, err = layer.NewActivation(function)
	if err != nil {
		t.Fatalf("NewActivation returned error: %v", err)
	}

	if activationLayer.Function() != function {
		t.Fatal("Function did not return wrapped activation")
	}
}

func Test_ActivationLayer_NilReceiverBehavior(t *testing.T) {
	var (
		activationLayer *layer.Activation
		err             error
	)

	if activationLayer.Function() != nil {
		t.Fatal("Function returned value for nil receiver")
	}

	_, err = activationLayer.Forward(mustMatrix(t, 1, 1, []float32{1}))
	if err == nil {
		t.Fatal("Forward error = nil, want nil receiver error")
	}

	_, err = activationLayer.Backward(mustMatrix(t, 1, 1, []float32{1}))
	if err == nil {
		t.Fatal("Backward error = nil, want nil receiver error")
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

	input = mustMatrix(t, 1, 3, []float32{-1, 0, 2})
	output, err = activationLayer.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	requireMatrixValues(t, output, []float32{
		0.2689414213699951,
		0.5,
		0.8807970779778823,
	})

	err = input.Set(0, 0, 10)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	outputGradient = mustMatrix(t, 1, 3, []float32{1, 2, 3})
	inputGradient, err = activationLayer.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float32{
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

	inputGradient, err = activationLayer.Backward(mustMatrix(t, 1, 1, []float32{1}))
	if err == nil {
		t.Fatalf("Backward returned gradient %v and nil error, want error", inputGradient)
	}
}

func Test_ActivationLayer_BackwardReturnsNilOnShapeError(t *testing.T) {
	var (
		activationLayer *layer.Activation
		inputGradient   *matrix.Matrix
		err             error
	)

	activationLayer, err = layer.NewActivation(activation.ReLU{})
	if err != nil {
		t.Fatalf("NewActivation returned error: %v", err)
	}

	if _, err = activationLayer.Forward(mustMatrix(t, 1, 2, []float32{1, 2})); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	inputGradient, err = activationLayer.Backward(mustMatrix(t, 2, 1, []float32{3, 4}))
	if err == nil {
		t.Fatal("Backward error = nil for mismatched gradient shape")
	}

	if inputGradient != nil {
		t.Fatal("Backward returned input gradient on error")
	}
}

func Test_ActivationLayer_CustomActivationFallback(t *testing.T) {
	var (
		function        fallbackActivation
		activationLayer *layer.Activation
		input           *matrix.Matrix
		output          *matrix.Matrix
		outputGradient  *matrix.Matrix
		inputGradient   *matrix.Matrix
		err             error
	)

	activationLayer, err = layer.NewActivation(&function)
	if err != nil {
		t.Fatalf("NewActivation returned error: %v", err)
	}

	input = mustMatrix(t, 1, 2, []float32{2, 3})
	output, err = activationLayer.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	requireMatrixValues(t, output, []float32{2, 3})

	outputGradient = mustMatrix(t, 1, 2, []float32{5, 7})
	inputGradient, err = activationLayer.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	requireMatrixValues(t, inputGradient, []float32{10, 21})

	if function.forwardCalls != 1 || function.backwardCalls != 1 {
		t.Fatalf(
			"custom activation calls = forward %d backward %d, want 1 each",
			function.forwardCalls,
			function.backwardCalls,
		)
	}
}

func Test_ActivationLayer_FailedForwardPreservesInputCache(t *testing.T) {
	var (
		function        fallbackActivation
		activationLayer *layer.Activation
		inputGradient   *matrix.Matrix
		err             error
	)

	activationLayer, err = layer.NewActivation(&function)
	if err != nil {
		t.Fatalf("NewActivation returned error: %v", err)
	}

	if _, err = activationLayer.Forward(mustMatrix(t, 1, 2, []float32{2, 3})); err != nil {
		t.Fatalf("initial Forward returned error: %v", err)
	}

	function.forwardError = errors.New("forward failed")
	if _, err = activationLayer.Forward(mustMatrix(t, 2, 1, []float32{11, 13})); err == nil {
		t.Fatal("failed Forward error = nil")
	}

	inputGradient, err = activationLayer.Backward(mustMatrix(t, 1, 2, []float32{5, 7}))
	if err != nil {
		t.Fatalf("Backward after failed Forward returned error: %v", err)
	}
	requireMatrixValues(t, inputGradient, []float32{10, 21})
}
