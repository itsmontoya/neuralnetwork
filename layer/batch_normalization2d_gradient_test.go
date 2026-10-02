package layer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

const (
	batchNormalization2DGradientCheckStep      = 1e-3
	batchNormalization2DGradientCheckTolerance = 1e-2
)

func Test_BatchNormalization2D_GradientCheck(t *testing.T) {
	var (
		batchNorm              *layer.BatchNormalization2D
		input                  *matrix.Matrix
		outputGradient         *matrix.Matrix
		inputGradient          *matrix.Matrix
		numericalInputGradient *matrix.Matrix
		numericalGammaGradient *matrix.Matrix
		numericalBetaGradient  *matrix.Matrix
		err                    error
	)

	batchNorm = mustBatchNormalization2D(t, 2, 1, 2, 0.9, 1e-4)
	input = mustMatrix(t, 2, 4, []float32{
		0.5, -0.2, 1.1, 0.3,
		-0.7, 0.9, -0.4, 0.8,
	})
	outputGradient = mustMatrix(t, 2, 4, []float32{
		0.2, -0.5, 0.7, 0.1,
		-0.3, 0.9, -0.2, 0.4,
	})
	if _, err = batchNorm.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if inputGradient, err = batchNorm.Backward(outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	numericalInputGradient = finiteDifferenceBatchNormalization2D(
		t,
		batchNorm,
		input,
		outputGradient,
		input,
	)
	numericalGammaGradient = finiteDifferenceBatchNormalization2D(
		t,
		batchNorm,
		input,
		outputGradient,
		batchNorm.Gamma().Values(),
	)
	numericalBetaGradient = finiteDifferenceBatchNormalization2D(
		t,
		batchNorm,
		input,
		outputGradient,
		batchNorm.Beta().Values(),
	)

	testutil.RequireMatrixAlmostEqual(t, inputGradient, numericalInputGradient, batchNormalization2DGradientCheckTolerance)
	testutil.RequireMatrixAlmostEqual(
		t,
		batchNorm.Gamma().Gradient(),
		numericalGammaGradient,
		batchNormalization2DGradientCheckTolerance,
	)
	testutil.RequireMatrixAlmostEqual(
		t,
		batchNorm.Beta().Gradient(),
		numericalBetaGradient,
		batchNormalization2DGradientCheckTolerance,
	)
}

func finiteDifferenceBatchNormalization2D(
	tb testing.TB,
	batchNorm *layer.BatchNormalization2D,
	input,
	outputGradient,
	values *matrix.Matrix,
) (gradient *matrix.Matrix) {
	var err error

	tb.Helper()
	gradient, err = testutil.FiniteDifferenceGradient(
		values,
		batchNormalization2DGradientCheckStep,
		func() (value float32, err error) {
			var output *matrix.Matrix

			if output, err = batchNorm.Forward(input); err != nil {
				return 0, err
			}
			value, err = testutil.WeightedMatrixSum(output, outputGradient)
			return value, err
		},
	)
	if err != nil {
		tb.Fatalf("FiniteDifferenceGradient returned error: %v", err)
	}

	return gradient
}
