package activation_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

const (
	activationGradientCheckStep      = 1e-3
	activationGradientCheckTolerance = 5e-3
)

func Test_Activation_GradientCheck(t *testing.T) {
	type testcase struct {
		name       string
		activation activation.Activation
		input      *matrix.Matrix
		gradient   *matrix.Matrix
	}

	var tests []testcase

	tests = []testcase{
		{
			name:       "elu",
			activation: activation.ELU{},
			input: mustMatrix(t, 2, 2, []float32{
				-0.8, 0.4,
				1.1, -0.3,
			}),
			gradient: mustMatrix(t, 2, 2, []float32{
				0.2, -0.7,
				1.3, 0.5,
			}),
		},
		{
			name:       "gelu",
			activation: activation.GELU{},
			input: mustMatrix(t, 2, 2, []float32{
				-1.2, 0.7,
				1.6, -0.4,
			}),
			gradient: mustMatrix(t, 2, 2, []float32{
				0.6, -0.2,
				-0.9, 1.4,
			}),
		},
		{
			name:       "leaky relu",
			activation: activation.LeakyReLU{},
			input: mustMatrix(t, 2, 2, []float32{
				-0.9, 0.25,
				1.2, -1.5,
			}),
			gradient: mustMatrix(t, 2, 2, []float32{
				-0.4, 0.8,
				1.1, -0.6,
			}),
		},
		{
			name:       "relu",
			activation: activation.ReLU{},
			input: mustMatrix(t, 2, 2, []float32{
				-0.8, 0.4,
				1.1, -0.3,
			}),
			gradient: mustMatrix(t, 2, 2, []float32{
				0.2, -0.7,
				1.3, 0.5,
			}),
		},
		{
			name:       "sigmoid",
			activation: activation.Sigmoid{},
			input: mustMatrix(t, 2, 2, []float32{
				-1.2, 0.7,
				1.6, -0.4,
			}),
			gradient: mustMatrix(t, 2, 2, []float32{
				0.6, -0.2,
				-0.9, 1.4,
			}),
		},
		{
			name:       "tanh",
			activation: activation.Tanh{},
			input: mustMatrix(t, 2, 2, []float32{
				-0.9, 0.25,
				1.2, -1.5,
			}),
			gradient: mustMatrix(t, 2, 2, []float32{
				-0.4, 0.8,
				1.1, -0.6,
			}),
		},
		{
			name:       "linear",
			activation: activation.Linear{},
			input: mustMatrix(t, 2, 2, []float32{
				-0.6, 0.2,
				0.9, 1.3,
			}),
			gradient: mustMatrix(t, 2, 2, []float32{
				0.3, -0.5,
				1.7, -1.1,
			}),
		},
		{
			name:       "softmax",
			activation: activation.Softmax{},
			input: mustMatrix(t, 2, 3, []float32{
				0.3, -0.7, 1.1,
				-1.2, 0.4, 0.8,
			}),
			gradient: mustMatrix(t, 2, 3, []float32{
				0.5, -0.25, 1.2,
				-0.4, 0.9, -0.1,
			}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(tb *testing.T) {
			var (
				inputGradient     *matrix.Matrix
				numericalGradient *matrix.Matrix
				err               error
			)

			inputGradient, err = tt.activation.Backward(tt.input, tt.gradient)
			if err != nil {
				tb.Fatalf("Backward returned error: %v", err)
			}

			numericalGradient, err = testutil.FiniteDifferenceGradient(
				tt.input,
				activationGradientCheckStep,
				func() (value float32, err error) {
					var output *matrix.Matrix

					if output, err = tt.activation.Forward(tt.input); err != nil {
						return 0, err
					}

					value, err = testutil.WeightedMatrixSum(output, tt.gradient)
					return value, err
				},
			)
			if err != nil {
				tb.Fatalf("FiniteDifferenceGradient returned error: %v", err)
			}

			testutil.RequireMatrixAlmostEqual(tb, inputGradient, numericalGradient, activationGradientCheckTolerance)
		})
	}
}
