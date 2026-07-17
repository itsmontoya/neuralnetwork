package layer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

const (
	conv2DGradientCheckStep      = 1e-3
	conv2DGradientCheckTolerance = 1e-2
)

func Test_Conv2D_GradientCheck_Weights(t *testing.T) {
	var (
		conv              *layer.Conv2D
		input             *matrix.Matrix
		outputGradient    *matrix.Matrix
		numericalGradient *matrix.Matrix
		err               error
	)

	conv = mustGradientConv2D(t)
	input = mustGradientConv2DInput(t)
	outputGradient = mustGradientConv2DOutput(t)
	if _, err = conv.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	if _, err = conv.Backward(outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	numericalGradient, err = testutil.FiniteDifferenceGradient(
		conv.Weights().Values(),
		conv2DGradientCheckStep,
		func() (value float32, err error) {
			var output *matrix.Matrix

			if output, err = conv.Forward(input); err != nil {
				return 0, err
			}

			value, err = testutil.WeightedMatrixSum(output, outputGradient)
			return value, err
		},
	)
	if err != nil {
		t.Fatalf("FiniteDifferenceGradient returned error: %v", err)
	}

	testutil.RequireMatrixAlmostEqual(
		t,
		conv.Weights().Gradient(),
		numericalGradient,
		conv2DGradientCheckTolerance,
	)
}

func Test_Conv2D_GradientCheck_Input(t *testing.T) {
	var (
		conv              *layer.Conv2D
		input             *matrix.Matrix
		outputGradient    *matrix.Matrix
		inputGradient     *matrix.Matrix
		numericalGradient *matrix.Matrix
		err               error
	)

	conv = mustGradientConv2D(t)
	input = mustGradientConv2DInput(t)
	outputGradient = mustGradientConv2DOutput(t)
	if _, err = conv.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	if inputGradient, err = conv.Backward(outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	numericalGradient, err = testutil.FiniteDifferenceGradient(
		input,
		conv2DGradientCheckStep,
		func() (value float32, err error) {
			var output *matrix.Matrix

			if output, err = conv.Forward(input); err != nil {
				return 0, err
			}

			value, err = testutil.WeightedMatrixSum(output, outputGradient)
			return value, err
		},
	)
	if err != nil {
		t.Fatalf("FiniteDifferenceGradient returned error: %v", err)
	}

	testutil.RequireMatrixAlmostEqual(
		t,
		inputGradient,
		numericalGradient,
		conv2DGradientCheckTolerance,
	)
}

func mustGradientConv2D(tb testing.TB) (conv *layer.Conv2D) {
	var config layer.Conv2DConfig

	tb.Helper()
	config = mustConv2DConfig(tb, 2, 2, 3, 2, 2, 2, 2, 1, 1, 1)
	conv = mustConv2D(
		tb,
		config,
		[]float32{
			0.25, -0.4,
			0.1, 0.35,
			-0.2, 0.5,
			0.45, -0.15,
			-0.3, 0.2,
			0.4, 0.1,
			0.15, -0.25,
			-0.35, 0.3,
		},
		[]float32{0.05, -0.1},
	)
	return conv
}

func mustGradientConv2DInput(tb testing.TB) (input *matrix.Matrix) {
	tb.Helper()
	input = mustMatrix(tb, 2, 12, []float32{
		0.6, -0.4, 0.2,
		1.2, 0.3, -0.7,
		-0.5, 0.8, 0.1,
		0.9, -1.1, 0.4,
		-0.2, 0.7, 1.1,
		0.5, -0.6, 0.3,
		0.4, -0.9, 0.2,
		1.3, 0.6, -0.8,
	})
	return input
}

func mustGradientConv2DOutput(tb testing.TB) (outputGradient *matrix.Matrix) {
	tb.Helper()
	outputGradient = mustMatrix(tb, 2, 16, []float32{
		0.7, -0.3, 0.2, 0.5,
		-0.2, 0.4, 0.1, -0.6,
		0.3, -0.5, 0.8, 0.2,
		-0.4, 0.6, -0.1, 0.9,
		-0.6, 0.2, 0.5, -0.3,
		0.4, 0.7, -0.8, 0.1,
		0.9, -0.2, 0.3, -0.5,
		0.6, 0.1, -0.4, 0.8,
	})
	return outputGradient
}
