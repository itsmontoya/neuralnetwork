package layer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

const (
	denseGradientCheckStep      = 1e-3
	denseGradientCheckTolerance = 5e-3
)

func Test_Dense_GradientCheck_Weights(t *testing.T) {
	var (
		dense             = mustGradientDense(t)
		input             = mustGradientInput(t)
		outputGradient    = mustGradientOutput(t)
		numericalGradient *matrix.Matrix
		err               error
	)

	_, err = dense.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	_, err = dense.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	numericalGradient, err = testutil.FiniteDifferenceGradient(
		dense.Weights().Values(),
		denseGradientCheckStep,
		func() (value float32, err error) {
			var output *matrix.Matrix

			if output, err = dense.Forward(input); err != nil {
				return 0, err
			}

			value, err = testutil.WeightedMatrixSum(output, outputGradient)
			return value, err
		},
	)
	if err != nil {
		t.Fatalf("FiniteDifferenceGradient returned error: %v", err)
	}

	testutil.RequireMatrixAlmostEqual(t, dense.Weights().Gradient(), numericalGradient, denseGradientCheckTolerance)
}

func Test_Dense_GradientCheck_Biases(t *testing.T) {
	var (
		dense             = mustGradientDense(t)
		input             = mustGradientInput(t)
		outputGradient    = mustGradientOutput(t)
		numericalGradient *matrix.Matrix
		err               error
	)

	_, err = dense.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	_, err = dense.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	numericalGradient, err = testutil.FiniteDifferenceGradient(
		dense.Biases().Values(),
		denseGradientCheckStep,
		func() (value float32, err error) {
			var output *matrix.Matrix

			if output, err = dense.Forward(input); err != nil {
				return 0, err
			}

			value, err = testutil.WeightedMatrixSum(output, outputGradient)
			return value, err
		},
	)
	if err != nil {
		t.Fatalf("FiniteDifferenceGradient returned error: %v", err)
	}

	testutil.RequireMatrixAlmostEqual(t, dense.Biases().Gradient(), numericalGradient, denseGradientCheckTolerance)
}

func Test_Dense_GradientCheck_Input(t *testing.T) {
	var (
		dense             = mustGradientDense(t)
		input             = mustGradientInput(t)
		outputGradient    = mustGradientOutput(t)
		inputGradient     *matrix.Matrix
		numericalGradient *matrix.Matrix
		err               error
	)

	_, err = dense.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	inputGradient, err = dense.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	numericalGradient, err = testutil.FiniteDifferenceGradient(
		input,
		denseGradientCheckStep,
		func() (value float32, err error) {
			var output *matrix.Matrix

			if output, err = dense.Forward(input); err != nil {
				return 0, err
			}

			value, err = testutil.WeightedMatrixSum(output, outputGradient)
			return value, err
		},
	)
	if err != nil {
		t.Fatalf("FiniteDifferenceGradient returned error: %v", err)
	}

	testutil.RequireMatrixAlmostEqual(t, inputGradient, numericalGradient, denseGradientCheckTolerance)
}

func mustGradientDense(tb testing.TB) (dense *layer.Dense) {
	tb.Helper()

	dense = mustDense(
		tb,
		2,
		2,
		[]float32{
			0.35, -0.2,
			0.15, 0.4,
		},
		[]float32{0.05, -0.1},
	)
	return dense
}

func mustGradientInput(tb testing.TB) (input *matrix.Matrix) {
	tb.Helper()

	input = mustMatrix(tb, 2, 2, []float32{
		0.6, -0.4,
		1.2, 0.3,
	})
	return input
}

func mustGradientOutput(tb testing.TB) (outputGradient *matrix.Matrix) {
	tb.Helper()

	outputGradient = mustMatrix(tb, 2, 2, []float32{
		0.7, -0.3,
		-0.2, 0.5,
	})
	return outputGradient
}
