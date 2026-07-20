package layer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

const (
	simpleRNNGradientCheckStep      = 1e-3
	simpleRNNGradientCheckTolerance = 1e-2
)

func Test_SimpleRNN_GradientCheck_InputWeights(t *testing.T) {
	var (
		recurrent         = mustGradientSimpleRNN(t)
		input             = mustGradientSimpleRNNInput(t)
		outputGradient    = mustGradientSimpleRNNOutput(t)
		numericalGradient *matrix.Matrix
		err               error
	)

	if _, err = recurrent.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if _, err = recurrent.Backward(outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	numericalGradient, err = testutil.FiniteDifferenceGradient(
		recurrent.InputWeights().Values(),
		simpleRNNGradientCheckStep,
		func() (value float32, err error) {
			var output *matrix.Matrix

			if output, err = recurrent.Forward(input); err != nil {
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
		recurrent.InputWeights().Gradient(),
		numericalGradient,
		simpleRNNGradientCheckTolerance,
	)
}

func Test_SimpleRNN_GradientCheck_RecurrentWeights(t *testing.T) {
	var (
		recurrent         = mustGradientSimpleRNN(t)
		input             = mustGradientSimpleRNNInput(t)
		outputGradient    = mustGradientSimpleRNNOutput(t)
		numericalGradient *matrix.Matrix
		err               error
	)

	if _, err = recurrent.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if _, err = recurrent.Backward(outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	numericalGradient, err = testutil.FiniteDifferenceGradient(
		recurrent.RecurrentWeights().Values(),
		simpleRNNGradientCheckStep,
		func() (value float32, err error) {
			var output *matrix.Matrix

			if output, err = recurrent.Forward(input); err != nil {
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
		recurrent.RecurrentWeights().Gradient(),
		numericalGradient,
		simpleRNNGradientCheckTolerance,
	)
}

func Test_SimpleRNN_GradientCheck_Biases(t *testing.T) {
	var (
		recurrent         = mustGradientSimpleRNN(t)
		input             = mustGradientSimpleRNNInput(t)
		outputGradient    = mustGradientSimpleRNNOutput(t)
		numericalGradient *matrix.Matrix
		err               error
	)

	if _, err = recurrent.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if _, err = recurrent.Backward(outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	numericalGradient, err = testutil.FiniteDifferenceGradient(
		recurrent.Biases().Values(),
		simpleRNNGradientCheckStep,
		func() (value float32, err error) {
			var output *matrix.Matrix

			if output, err = recurrent.Forward(input); err != nil {
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
		recurrent.Biases().Gradient(),
		numericalGradient,
		simpleRNNGradientCheckTolerance,
	)
}

func Test_SimpleRNN_GradientCheck_Input(t *testing.T) {
	var (
		recurrent         = mustGradientSimpleRNN(t)
		input             = mustGradientSimpleRNNInput(t)
		outputGradient    = mustGradientSimpleRNNOutput(t)
		inputGradient     *matrix.Matrix
		numericalGradient *matrix.Matrix
		err               error
	)

	if _, err = recurrent.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if inputGradient, err = recurrent.Backward(outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	numericalGradient, err = testutil.FiniteDifferenceGradient(
		input,
		simpleRNNGradientCheckStep,
		func() (value float32, err error) {
			var output *matrix.Matrix

			if output, err = recurrent.Forward(input); err != nil {
				return 0, err
			}

			value, err = testutil.WeightedMatrixSum(output, outputGradient)
			return value, err
		},
	)
	if err != nil {
		t.Fatalf("FiniteDifferenceGradient returned error: %v", err)
	}

	testutil.RequireMatrixAlmostEqual(t, inputGradient, numericalGradient, simpleRNNGradientCheckTolerance)
}

func mustGradientSimpleRNN(tb testing.TB) (recurrent *layer.SimpleRNN) {
	var (
		config layer.SimpleRNNConfig
		err    error
	)

	tb.Helper()
	config, err = layer.NewSimpleRNNConfig(mustSequenceShape(tb, 3, 2), 2)
	if err != nil {
		tb.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	recurrent = mustSimpleRNN(
		tb,
		config,
		[]float32{
			0.35, -0.2,
			0.15, 0.4,
		},
		[]float32{
			0.2, -0.1,
			0.3, 0.25,
		},
		[]float32{0.05, -0.1},
	)
	return recurrent
}

func mustGradientSimpleRNNInput(tb testing.TB) (input *matrix.Matrix) {
	tb.Helper()
	input = mustMatrix(tb, 2, 6, []float32{
		0.6, -0.4,
		1.2, 0.3,
		-0.2, 0.8,
		0.1, 0.5,
		-0.7, 0.2,
		0.9, -0.3,
	})
	return input
}

func mustGradientSimpleRNNOutput(tb testing.TB) (outputGradient *matrix.Matrix) {
	tb.Helper()
	outputGradient = mustMatrix(tb, 2, 6, []float32{
		0.7, -0.3,
		-0.2, 0.5,
		0.4, 0.1,
		-0.6, 0.2,
		0.3, -0.4,
		0.5, 0.8,
	})
	return outputGradient
}
