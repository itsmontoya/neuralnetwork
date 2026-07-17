package layer

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_MaxPool2D_ValidatesInternalConfigurationState(t *testing.T) {
	type testcase struct {
		name      string
		mutate    func(pool *MaxPool2D)
		wantError string
	}

	tests := []testcase{
		{
			name: "zero configuration",
			mutate: func(pool *MaxPool2D) {
				pool.config = MaxPool2DConfig{}
			},
			wantError: "configuration invalid",
		},
		{
			name: "wrong output shape",
			mutate: func(pool *MaxPool2D) {
				pool.config.outputShape = mustInternalMaxPool2DShape(t, 1, 1, 2)
			},
			wantError: "output shape mismatch",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				pool *MaxPool2D
				err  error
			)

			pool = mustInternalMaxPool2D(t)
			tt.mutate(pool)
			_, err = pool.Forward(mustInternalMaxPool2DMatrix(t, 1, 4))
			if err == nil {
				t.Fatal("Forward error = nil, want invalid state error")
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("Forward error = %q, want substring %q", err, tt.wantError)
			}
		})
	}
}

func Test_MaxPool2D_BackwardValidatesArgmaxState(t *testing.T) {
	type testcase struct {
		name      string
		mutate    func(pool *MaxPool2D)
		wantError string
	}

	tests := []testcase{
		{
			name: "wrong cache length",
			mutate: func(pool *MaxPool2D) {
				pool.argmax = nil
			},
			wantError: "argmax cache length mismatch: got=0 want=1",
		},
		{
			name: "position out of range",
			mutate: func(pool *MaxPool2D) {
				pool.argmax[0] = pool.config.InputShape().Size()
			},
			wantError: "argmax cache position out of range",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				pool           *MaxPool2D
				input          *matrix.Matrix
				outputGradient *matrix.Matrix
				err            error
			)

			pool = mustInternalMaxPool2D(t)
			input = mustInternalMaxPool2DMatrix(t, 1, pool.config.InputShape().Size())
			if _, err = pool.Forward(input); err != nil {
				t.Fatalf("Forward returned error: %v", err)
			}

			tt.mutate(pool)
			outputGradient = mustInternalMaxPool2DMatrix(t, 1, pool.config.OutputShape().Size())
			if _, err = pool.Backward(outputGradient); err == nil {
				t.Fatal("Backward error = nil, want invalid cache error")
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("Backward error = %q, want substring %q", err, tt.wantError)
			}
		})
	}
}

func mustInternalMaxPool2D(tb testing.TB) (pool *MaxPool2D) {
	var (
		inputShape SpatialShape
		config     MaxPool2DConfig
		err        error
	)

	tb.Helper()
	inputShape = mustInternalMaxPool2DShape(tb, 1, 2, 2)
	config, err = NewMaxPool2DConfig(inputShape, 2, 2, 1, 1)
	if err != nil {
		tb.Fatalf("NewMaxPool2DConfig returned error: %v", err)
	}

	pool, err = NewMaxPool2D(config)
	if err != nil {
		tb.Fatalf("NewMaxPool2D returned error: %v", err)
	}

	return pool
}

func mustInternalMaxPool2DShape(tb testing.TB, channels, height, width int) (shape SpatialShape) {
	var err error

	tb.Helper()
	shape, err = NewSpatialShape(channels, height, width)
	if err != nil {
		tb.Fatalf("NewSpatialShape returned error: %v", err)
	}

	return shape
}

func mustInternalMaxPool2DMatrix(tb testing.TB, rows, cols int) (value *matrix.Matrix) {
	var err error

	tb.Helper()
	value, err = matrix.New(rows, cols)
	if err != nil {
		tb.Fatalf("matrix.New returned error: %v", err)
	}

	return value
}
