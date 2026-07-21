package layer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_SimpleRNNForwardSteadyStateAllocations(t *testing.T) {
	var (
		recurrent *layer.SimpleRNN
		input     *matrix.Matrix
		err       error
	)

	recurrent = benchmarkSimpleRNN(t, 8, 16, 32)
	input = allocationLayerMatrix(t, 16, recurrent.InputShape().Size())
	if _, err = recurrent.Forward(input); err != nil {
		t.Fatalf("warm-up Forward returned error: %v", err)
	}

	requireMaxAllocs(t, "SimpleRNN.Forward", 0, func() {
		if allocationLayerResult, err = recurrent.Forward(input); err != nil {
			panic(err)
		}
	})
}

func Test_SimpleRNNBackwardSteadyStateAllocations(t *testing.T) {
	var (
		recurrent      *layer.SimpleRNN
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		err            error
	)

	recurrent = benchmarkSimpleRNN(t, 8, 16, 32)
	input = allocationLayerMatrix(t, 16, recurrent.InputShape().Size())
	outputGradient = allocationLayerMatrix(t, 16, recurrent.OutputShape().Size())
	if _, err = recurrent.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if _, err = recurrent.Backward(outputGradient); err != nil {
		t.Fatalf("warm-up Backward returned error: %v", err)
	}
	if err = recurrent.ResetGradients(); err != nil {
		t.Fatalf("ResetGradients returned error: %v", err)
	}

	requireMaxAllocs(t, "SimpleRNN.Backward", 0, func() {
		if allocationLayerResult, err = recurrent.Backward(outputGradient); err != nil {
			panic(err)
		}
	})
}

func Test_LastStepForwardSteadyStateAllocations(t *testing.T) {
	var (
		lastStep *layer.LastStep
		input    *matrix.Matrix
		err      error
	)

	lastStep = benchmarkLastStep(t, 8, 32)
	input = allocationLayerMatrix(t, 16, lastStep.InputShape().Size())
	if _, err = lastStep.Forward(input); err != nil {
		t.Fatalf("warm-up Forward returned error: %v", err)
	}

	requireMaxAllocs(t, "LastStep.Forward", 0, func() {
		if allocationLayerResult, err = lastStep.Forward(input); err != nil {
			panic(err)
		}
	})
}

func Test_LastStepBackwardSteadyStateAllocations(t *testing.T) {
	var (
		lastStep       *layer.LastStep
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		err            error
	)

	lastStep = benchmarkLastStep(t, 8, 32)
	input = allocationLayerMatrix(t, 16, lastStep.InputShape().Size())
	outputGradient = allocationLayerMatrix(t, 16, lastStep.OutputSize())
	if _, err = lastStep.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if _, err = lastStep.Backward(outputGradient); err != nil {
		t.Fatalf("warm-up Backward returned error: %v", err)
	}

	requireMaxAllocs(t, "LastStep.Backward", 0, func() {
		if allocationLayerResult, err = lastStep.Backward(outputGradient); err != nil {
			panic(err)
		}
	})
}
