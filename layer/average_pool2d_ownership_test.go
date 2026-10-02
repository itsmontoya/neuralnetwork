package layer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_AveragePoolingLayersReuseScratchWithoutAliasing(t *testing.T) {
	type testcase struct {
		name       string
		newPooling func(testing.TB) layer.Layer
	}

	var tests []testcase
	tests = []testcase{
		{
			name: "average",
			newPooling: func(tb testing.TB) layer.Layer {
				var (
					pooling *layer.AveragePool2D
					err     error
				)

				if pooling, err = layer.NewAveragePool2D(mustAveragePool2DConfig(tb, 1, 2, 2, 2, 2, 2, 2)); err != nil {
					tb.Fatalf("NewAveragePool2D returned error: %v", err)
				}
				return pooling
			},
		},
		{
			name: "adaptive",
			newPooling: func(tb testing.TB) layer.Layer {
				var (
					pooling *layer.AdaptiveAveragePool2D
					err     error
				)

				if pooling, err = layer.NewAdaptiveAveragePool2D(mustAdaptiveAveragePool2DConfig(tb, 1, 2, 2, 1, 1)); err != nil {
					tb.Fatalf("NewAdaptiveAveragePool2D returned error: %v", err)
				}
				return pooling
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				pooling              layer.Layer
				firstInput           *matrix.Matrix
				secondInput          *matrix.Matrix
				firstOutput          *matrix.Matrix
				secondOutput         *matrix.Matrix
				firstOutputGradient  *matrix.Matrix
				secondOutputGradient *matrix.Matrix
				firstInputGradient   *matrix.Matrix
				secondInputGradient  *matrix.Matrix
				err                  error
			)

			pooling = tt.newPooling(t)
			firstInput = mustMatrix(t, 1, 4, []float32{1, 2, 3, 4})
			secondInput = mustMatrix(t, 1, 4, []float32{4, 3, 2, 1})
			if firstOutput, err = pooling.Forward(firstInput); err != nil {
				t.Fatalf("first Forward returned error: %v", err)
			}
			if secondOutput, err = pooling.Forward(secondInput); err != nil {
				t.Fatalf("second Forward returned error: %v", err)
			}
			if firstOutput != secondOutput {
				t.Fatal("Forward did not reuse output scratch")
			}
			if secondOutput == secondInput {
				t.Fatal("Forward output aliases caller input")
			}

			firstOutputGradient = mustMatrix(t, 1, 1, []float32{1})
			secondOutputGradient = mustMatrix(t, 1, 1, []float32{2})
			if firstInputGradient, err = pooling.Backward(firstOutputGradient); err != nil {
				t.Fatalf("first Backward returned error: %v", err)
			}
			if secondInputGradient, err = pooling.Backward(secondOutputGradient); err != nil {
				t.Fatalf("second Backward returned error: %v", err)
			}
			if firstInputGradient != secondInputGradient {
				t.Fatal("Backward did not reuse input-gradient scratch")
			}
			if secondInputGradient == secondOutputGradient {
				t.Fatal("Backward input gradient aliases caller output gradient")
			}
		})
	}
}
