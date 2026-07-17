package layer

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_Conv2D_ValidatesInternalParameterState(t *testing.T) {
	type testcase struct {
		name      string
		mutate    func(conv *Conv2D)
		wantError string
	}

	tests := []testcase{
		{
			name: "invalid configuration",
			mutate: func(conv *Conv2D) {
				conv.config = Conv2DConfig{}
			},
			wantError: "configuration invalid",
		},
		{
			name: "nil weights",
			mutate: func(conv *Conv2D) {
				conv.weights = nil
			},
			wantError: "weights parameter is nil",
		},
		{
			name: "wrong weight shape",
			mutate: func(conv *Conv2D) {
				conv.weights = mustInternalParameter(t, 3, 1)
			},
			wantError: "weights shape mismatch: got 3x1, want 4x1",
		},
		{
			name: "nil biases",
			mutate: func(conv *Conv2D) {
				conv.biases = nil
			},
			wantError: "biases parameter is nil",
		},
		{
			name: "wrong bias shape",
			mutate: func(conv *Conv2D) {
				conv.biases = mustInternalParameter(t, 1, 2)
			},
			wantError: "biases shape mismatch: got 1x2, want 1x1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				conv *Conv2D
				err  error
			)

			conv = mustInternalConv2D(t)
			tt.mutate(conv)
			_, err = conv.Forward(mustInternalConv2DMatrix(t, 1, 4))
			if err == nil {
				t.Fatal("Forward error = nil, want invalid state error")
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("Forward error = %q, want substring %q", err, tt.wantError)
			}
		})
	}
}

func Test_Conv2D_BackwardValidatesInputCacheState(t *testing.T) {
	var (
		conv           *Conv2D
		outputGradient *matrix.Matrix
		err            error
	)

	conv = mustInternalConv2D(t)
	if _, err = conv.Forward(mustInternalConv2DMatrix(t, 1, 4)); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	conv.inputCache = mustInternalConv2DMatrix(t, 1, 3)
	outputGradient = mustInternalConv2DMatrix(t, 1, 1)
	if _, err = conv.Backward(outputGradient); err == nil {
		t.Fatal("Backward error = nil, want invalid cache error")
	}

	if !strings.Contains(err.Error(), "input cache shape mismatch: got 1x3, want 1x4") {
		t.Fatalf("Backward error = %q, want cache dimensions", err)
	}
}

func mustInternalConv2D(tb testing.TB) (conv *Conv2D) {
	var (
		inputShape SpatialShape
		config     Conv2DConfig
		err        error
	)

	tb.Helper()
	inputShape, err = NewSpatialShape(1, 2, 2)
	if err != nil {
		tb.Fatalf("NewSpatialShape returned error: %v", err)
	}

	config, err = NewConv2DConfig(inputShape, 1, 2, 2, 1, 1, 0, 0)
	if err != nil {
		tb.Fatalf("NewConv2DConfig returned error: %v", err)
	}

	conv, err = NewConv2D(config, ZeroWeights)
	if err != nil {
		tb.Fatalf("NewConv2D returned error: %v", err)
	}

	return conv
}

func mustInternalParameter(tb testing.TB, rows, cols int) (parameter *optimizer.Parameter) {
	var (
		values *matrix.Matrix
		err    error
	)

	tb.Helper()
	values = mustInternalConv2DMatrix(tb, rows, cols)
	parameter, err = optimizer.NewParameter(values)
	if err != nil {
		tb.Fatalf("NewParameter returned error: %v", err)
	}

	return parameter
}

func mustInternalConv2DMatrix(tb testing.TB, rows, cols int) (value *matrix.Matrix) {
	var err error

	tb.Helper()
	value, err = matrix.New(rows, cols)
	if err != nil {
		tb.Fatalf("matrix.New returned error: %v", err)
	}

	return value
}
