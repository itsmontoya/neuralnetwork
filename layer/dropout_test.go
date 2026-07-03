package layer_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_Dropout_ImplementsLayer(t *testing.T) {
	var _ layer.Layer = (*layer.Dropout)(nil)
}

func Test_NewDropout_ValidatesConfig(t *testing.T) {
	type testcase struct {
		name   string
		rate   float64
		random *rand.Rand
	}

	tests := []testcase{
		{
			name:   "negative rate",
			rate:   -0.1,
			random: rand.New(rand.NewSource(1)),
		},
		{
			name:   "one rate",
			rate:   1,
			random: rand.New(rand.NewSource(1)),
		},
		{
			name:   "nan rate",
			rate:   math.NaN(),
			random: rand.New(rand.NewSource(1)),
		},
		{
			name:   "nil random",
			rate:   0.5,
			random: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				dropout *layer.Dropout
				err     error
			)

			dropout, err = layer.NewDropout(tt.rate, tt.random)
			if err == nil {
				t.Fatal("NewDropout error = nil, want error")
			}

			if dropout != nil {
				t.Fatal("NewDropout returned layer on error")
			}
		})
	}
}

func Test_Dropout_ForwardTrainingIsDeterministicWithSeed(t *testing.T) {
	var (
		first         *layer.Dropout
		second        *layer.Dropout
		input         *matrix.Matrix
		firstOutput   *matrix.Matrix
		secondOutput  *matrix.Matrix
		outputValues  []float64
		inputValues   []float64
		index         int
		kept          bool
		dropped       bool
		expectedScale float64
		err           error
	)

	first, err = layer.NewDropout(0.5, rand.New(rand.NewSource(7)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	second, err = layer.NewDropout(0.5, rand.New(rand.NewSource(7)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	input = mustMatrix(t, 2, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
	})

	firstOutput, err = first.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	secondOutput, err = second.Forward(input)
	if err != nil {
		t.Fatalf("second Forward returned error: %v", err)
	}

	requireMatrixValues(t, secondOutput, mustDropoutValues(t, firstOutput))

	inputValues = mustDropoutValues(t, input)
	outputValues = mustDropoutValues(t, firstOutput)
	expectedScale = 2
	for index = range outputValues {
		switch outputValues[index] {
		case 0:
			dropped = true
		case inputValues[index] * expectedScale:
			kept = true
		default:
			t.Fatalf("output value at index %d = %g, want 0 or %g", index, outputValues[index], inputValues[index]*expectedScale)
		}
	}

	if !kept {
		t.Fatal("Forward kept no values, want at least one kept value")
	}

	if !dropped {
		t.Fatal("Forward dropped no values, want at least one dropped value")
	}
}

func Test_Dropout_BackwardUsesTrainingMask(t *testing.T) {
	var (
		dropout        *layer.Dropout
		input          *matrix.Matrix
		output         *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		outputValues   []float64
		inputValues    []float64
		maskValues     []float64
		index          int
		err            error
	)

	dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(11)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	input = mustMatrix(t, 2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	output, err = dropout.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	inputValues = mustDropoutValues(t, input)
	outputValues = mustDropoutValues(t, output)
	maskValues = make([]float64, len(outputValues))
	for index = range maskValues {
		maskValues[index] = outputValues[index] / inputValues[index]
	}

	outputGradient = mustMatrix(t, 2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	inputGradient, err = dropout.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	for index = range maskValues {
		maskValues[index] *= float64(index + 1)
	}

	requireMatrixValues(t, inputGradient, maskValues)
}

func Test_Dropout_EvaluationModeIsIdentity(t *testing.T) {
	var (
		dropout        *layer.Dropout
		input          *matrix.Matrix
		output         *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	dropout, err = layer.NewDropout(0.75, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	dropout.SetTraining(false)
	if dropout.Training() {
		t.Fatal("Training = true, want false")
	}

	input = mustMatrix(t, 1, 3, []float64{1, 2, 3})
	output, err = dropout.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	err = input.Set(0, 0, 10)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	requireMatrixValues(t, output, []float64{1, 2, 3})

	outputGradient = mustMatrix(t, 1, 3, []float64{4, 5, 6})
	inputGradient, err = dropout.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	err = outputGradient.Set(0, 0, 20)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float64{4, 5, 6})
}

func Test_Dropout_BackwardRequiresForward(t *testing.T) {
	var (
		dropout       *layer.Dropout
		inputGradient *matrix.Matrix
		err           error
	)

	dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	inputGradient, err = dropout.Backward(mustMatrix(t, 1, 1, []float64{1}))
	if err == nil {
		t.Fatalf("Backward returned gradient %v and nil error, want error", inputGradient)
	}
}

func Test_Dropout_BackwardReportsShapeMismatch(t *testing.T) {
	var (
		dropout *layer.Dropout
		input   *matrix.Matrix
		err     error
	)

	dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	input = mustMatrix(t, 2, 2, []float64{
		1, 2,
		3, 4,
	})
	_, err = dropout.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	_, err = dropout.Backward(mustMatrix(t, 1, 2, []float64{1, 2}))
	if err == nil {
		t.Fatal("Backward error = nil, want shape error")
	}
}

func mustDropoutValues(tb testing.TB, m *matrix.Matrix) (values []float64) {
	var err error

	tb.Helper()

	values, err = m.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	return values
}
