package matrix_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_SoftmaxRowsInto(t *testing.T) {
	type testcase struct {
		name   string
		rows   int
		cols   int
		input  []float32
		output []float32
	}

	var tests []testcase

	tests = []testcase{
		{
			name:   "one by one",
			rows:   1,
			cols:   1,
			input:  []float32{4},
			output: []float32{1},
		},
		{
			name:   "single row",
			rows:   1,
			cols:   3,
			input:  []float32{1, 2, 3},
			output: []float32{0.09003057, 0.24472848, 0.66524094},
		},
		{
			name: "multiple rows",
			rows: 2,
			cols: 3,
			input: []float32{
				1, 2, 3,
				3, 2, 1,
			},
			output: []float32{
				0.09003057, 0.24472848, 0.66524094,
				0.66524094, 0.24472848, 0.09003057,
			},
		},
		{
			name:   "large logits",
			rows:   1,
			cols:   3,
			input:  []float32{1000, 1001, 1002},
			output: []float32{0.09003057, 0.24472848, 0.66524094},
		},
		{
			name:   "negative logits",
			rows:   1,
			cols:   3,
			input:  []float32{-1, -2, -3},
			output: []float32{0.66524094, 0.24472848, 0.09003057},
		},
		{
			name: "uneven rectangle",
			rows: 3,
			cols: 2,
			input: []float32{
				0, 1,
				2, 2,
				-3, -1,
			},
			output: []float32{
				0.26894143, 0.7310586,
				0.5, 0.5,
				0.11920292, 0.8807971,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input  *matrix.Matrix
				output *matrix.Matrix
				err    error
			)

			input = mustMatrix(t, tt.rows, tt.cols, tt.input)
			output = mustMatrix(t, tt.rows, tt.cols, make([]float32, len(tt.input)))
			if err = input.SoftmaxRowsInto(output); err != nil {
				t.Fatalf("SoftmaxRowsInto returned error: %v", err)
			}

			requireMatrixValues(t, output, tt.output)
		})
	}
}

func Test_SoftmaxRowsInto_AllowsInputAlias(t *testing.T) {
	var (
		input *matrix.Matrix
		err   error
	)

	input = mustMatrix(t, 2, 3, []float32{
		1, 2, 3,
		-1, -2, -3,
	})
	if err = input.SoftmaxRowsInto(input); err != nil {
		t.Fatalf("SoftmaxRowsInto returned error: %v", err)
	}

	requireMatrixValues(t, input, []float32{
		0.09003057, 0.24472848, 0.66524094,
		0.66524094, 0.24472848, 0.09003057,
	})
}

func Test_SoftmaxRowsInto_ValidatesMatrices(t *testing.T) {
	var (
		input      *matrix.Matrix
		output     *matrix.Matrix
		wrongShape *matrix.Matrix
		invalid    matrix.Matrix
		nilMatrix  *matrix.Matrix
		err        error
	)

	input = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	output = mustMatrix(t, 2, 3, []float32{0, 0, 0, 0, 0, 0})
	wrongShape = mustMatrix(t, 3, 2, []float32{0, 0, 0, 0, 0, 0})

	if err = nilMatrix.SoftmaxRowsInto(output); err == nil {
		t.Fatal("SoftmaxRowsInto error = nil for nil input")
	}

	if err = input.SoftmaxRowsInto(nil); err == nil {
		t.Fatal("SoftmaxRowsInto error = nil for nil destination")
	}

	if err = input.SoftmaxRowsInto(wrongShape); err == nil {
		t.Fatal("SoftmaxRowsInto error = nil for wrong destination shape")
	}

	if err = invalid.SoftmaxRowsInto(output); err == nil {
		t.Fatal("SoftmaxRowsInto error = nil for invalid input")
	}

	if err = input.SoftmaxRowsInto(&invalid); err == nil {
		t.Fatal("SoftmaxRowsInto error = nil for invalid destination")
	}
}

func Test_SoftmaxRowsBackwardInto(t *testing.T) {
	var (
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	input = mustMatrix(t, 2, 2, []float32{
		0, 0,
		1, 2,
	})
	outputGradient = mustMatrix(t, 2, 2, []float32{
		1, 3,
		2, -1,
	})
	inputGradient = mustMatrix(t, 2, 2, []float32{9, 9, 9, 9})

	if err = input.SoftmaxRowsBackwardInto(outputGradient, inputGradient); err != nil {
		t.Fatalf("SoftmaxRowsBackwardInto returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float32{-0.5, 0.5, 0.5898358, -0.58983576})
}

func Test_SoftmaxRowsBackwardInto_AliasRules(t *testing.T) {
	var (
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		err            error
	)

	input = mustMatrix(t, 1, 2, []float32{0, 0})
	outputGradient = mustMatrix(t, 1, 2, []float32{1, 3})
	if err = input.SoftmaxRowsBackwardInto(outputGradient, input); err != nil {
		t.Fatalf("SoftmaxRowsBackwardInto returned error for input alias: %v", err)
	}
	requireMatrixValues(t, input, []float32{-0.5, 0.5})

	input = mustMatrix(t, 1, 2, []float32{0, 0})
	if err = input.SoftmaxRowsBackwardInto(outputGradient, outputGradient); err == nil {
		t.Fatal("SoftmaxRowsBackwardInto error = nil for output-gradient alias")
	}
	requireMatrixValues(t, outputGradient, []float32{1, 3})
}

func Test_SoftmaxRowsBackwardInto_ValidatesMatrices(t *testing.T) {
	var (
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		wrongShape     *matrix.Matrix
		invalid        matrix.Matrix
		nilMatrix      *matrix.Matrix
		err            error
	)

	input = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	outputGradient = mustMatrix(t, 2, 3, []float32{1, 1, 1, 1, 1, 1})
	inputGradient = mustMatrix(t, 2, 3, []float32{0, 0, 0, 0, 0, 0})
	wrongShape = mustMatrix(t, 3, 2, []float32{0, 0, 0, 0, 0, 0})

	if err = nilMatrix.SoftmaxRowsBackwardInto(outputGradient, inputGradient); err == nil {
		t.Fatal("SoftmaxRowsBackwardInto error = nil for nil input")
	}

	if err = input.SoftmaxRowsBackwardInto(nil, inputGradient); err == nil {
		t.Fatal("SoftmaxRowsBackwardInto error = nil for nil output gradient")
	}

	if err = input.SoftmaxRowsBackwardInto(outputGradient, nil); err == nil {
		t.Fatal("SoftmaxRowsBackwardInto error = nil for nil destination")
	}

	if err = input.SoftmaxRowsBackwardInto(wrongShape, inputGradient); err == nil {
		t.Fatal("SoftmaxRowsBackwardInto error = nil for wrong output-gradient shape")
	}

	if err = input.SoftmaxRowsBackwardInto(outputGradient, wrongShape); err == nil {
		t.Fatal("SoftmaxRowsBackwardInto error = nil for wrong destination shape")
	}

	if err = invalid.SoftmaxRowsBackwardInto(outputGradient, inputGradient); err == nil {
		t.Fatal("SoftmaxRowsBackwardInto error = nil for invalid input")
	}

	if err = input.SoftmaxRowsBackwardInto(&invalid, inputGradient); err == nil {
		t.Fatal("SoftmaxRowsBackwardInto error = nil for invalid output gradient")
	}

	if err = input.SoftmaxRowsBackwardInto(outputGradient, &invalid); err == nil {
		t.Fatal("SoftmaxRowsBackwardInto error = nil for invalid destination")
	}
}

func Test_SoftmaxRowsDestinationAllocations(t *testing.T) {
	var (
		input          *matrix.Matrix
		output         *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		forwardAllocs  float64
		backwardAllocs float64
		err            error
	)

	input = mustMatrix(t, 128, 64, make([]float32, 128*64))
	output = mustMatrix(t, 128, 64, make([]float32, 128*64))
	outputGradient = mustMatrix(t, 128, 64, make([]float32, 128*64))
	inputGradient = mustMatrix(t, 128, 64, make([]float32, 128*64))

	forwardAllocs = testing.AllocsPerRun(100, func() {
		if err = input.SoftmaxRowsInto(output); err != nil {
			panic(err)
		}
	})
	if forwardAllocs != 0 {
		t.Fatalf("SoftmaxRowsInto allocations = %.0f, want 0", forwardAllocs)
	}

	backwardAllocs = testing.AllocsPerRun(100, func() {
		if err = input.SoftmaxRowsBackwardInto(outputGradient, inputGradient); err != nil {
			panic(err)
		}
	})
	if backwardAllocs != 0 {
		t.Fatalf("SoftmaxRowsBackwardInto allocations = %.0f, want 0", backwardAllocs)
	}
}
