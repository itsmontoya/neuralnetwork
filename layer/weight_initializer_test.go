package layer_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_WeightInitializers_WrapMatrixConstructors(t *testing.T) {
	type testcase struct {
		name        string
		initializer layer.WeightInitializer
		expected    func(tb testing.TB) (weights *matrix.Matrix)
	}

	tests := []testcase{
		{
			name:        "zero",
			initializer: layer.ZeroWeights,
			expected: func(tb testing.TB) (weights *matrix.Matrix) {
				var err error

				tb.Helper()

				weights, err = matrix.New(2, 3)
				if err != nil {
					tb.Fatalf("New returned error: %v", err)
				}

				return weights
			},
		},
		{
			name:        "uniform",
			initializer: layer.UniformWeights(-0.25, 0.75, rand.New(rand.NewSource(11))),
			expected: func(tb testing.TB) (weights *matrix.Matrix) {
				var err error

				tb.Helper()

				weights, err = matrix.NewUniform(2, 3, -0.25, 0.75, rand.New(rand.NewSource(11)))
				if err != nil {
					tb.Fatalf("NewUniform returned error: %v", err)
				}

				return weights
			},
		},
		{
			name:        "normal",
			initializer: layer.NormalWeights(1.5, 0.25, rand.New(rand.NewSource(13))),
			expected: func(tb testing.TB) (weights *matrix.Matrix) {
				var err error

				tb.Helper()

				weights, err = matrix.NewNormal(2, 3, 1.5, 0.25, rand.New(rand.NewSource(13)))
				if err != nil {
					tb.Fatalf("NewNormal returned error: %v", err)
				}

				return weights
			},
		},
		{
			name:        "xavier uniform",
			initializer: layer.XavierUniformWeights(rand.New(rand.NewSource(17))),
			expected: func(tb testing.TB) (weights *matrix.Matrix) {
				var err error

				tb.Helper()

				weights, err = matrix.NewXavierUniform(2, 3, rand.New(rand.NewSource(17)))
				if err != nil {
					tb.Fatalf("NewXavierUniform returned error: %v", err)
				}

				return weights
			},
		},
		{
			name:        "he normal",
			initializer: layer.HeNormalWeights(rand.New(rand.NewSource(19))),
			expected: func(tb testing.TB) (weights *matrix.Matrix) {
				var err error

				tb.Helper()

				weights, err = matrix.NewHeNormal(2, 3, rand.New(rand.NewSource(19)))
				if err != nil {
					tb.Fatalf("NewHeNormal returned error: %v", err)
				}

				return weights
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				got  *matrix.Matrix
				err  error
				want *matrix.Matrix
			)

			got, err = tt.initializer(2, 3)
			if err != nil {
				t.Fatalf("initializer returned error: %v", err)
			}

			want = tt.expected(t)
			testutil.RequireMatrixAlmostEqual(t, got, want, epsilon)
		})
	}
}
