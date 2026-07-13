package layer_test

import (
	"math"
	"math/rand"
	"strings"
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
		rate   float32
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
			rate:   float32(math.NaN()),
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

func Test_Dropout_AccessorsAndNilReceiverBehavior(t *testing.T) {
	var (
		dropout    *layer.Dropout
		nilDropout *layer.Dropout
		err        error
	)

	dropout, err = layer.NewDropout(0.25, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	if dropout.Rate() != 0.25 {
		t.Fatalf("Rate = %g, want 0.25", dropout.Rate())
	}

	if !dropout.Training() {
		t.Fatal("Training = false, want true")
	}

	dropout.SetTraining(false)
	if dropout.Training() {
		t.Fatal("Training = true, want false")
	}

	if nilDropout.Rate() != 0 {
		t.Fatalf("nil Rate = %g, want 0", nilDropout.Rate())
	}

	if nilDropout.Training() {
		t.Fatal("nil Training = true, want false")
	}

	nilDropout.SetTraining(true)
	if nilDropout.Training() {
		t.Fatal("nil Training changed after SetTraining")
	}

	_, err = nilDropout.Forward(mustMatrix(t, 1, 1, []float32{1}))
	if err == nil {
		t.Fatal("nil Forward error = nil, want error")
	}

	_, err = nilDropout.Backward(mustMatrix(t, 1, 1, []float32{1}))
	if err == nil {
		t.Fatal("nil Backward error = nil, want error")
	}
}

func Test_Dropout_ForwardTrainingIsDeterministicWithSeed(t *testing.T) {
	var (
		first         *layer.Dropout
		second        *layer.Dropout
		input         *matrix.Matrix
		firstOutput   *matrix.Matrix
		secondOutput  *matrix.Matrix
		outputValues  []float32
		inputValues   []float32
		index         int
		kept          bool
		dropped       bool
		expectedScale float32
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

	input = mustMatrix(t, 2, 4, []float32{
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

func Test_Dropout_RateZeroTrainingIsIdentity(t *testing.T) {
	var (
		dropout        *layer.Dropout
		input          *matrix.Matrix
		output         *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	dropout, err = layer.NewDropout(0, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	input = mustMatrix(t, 1, 3, []float32{1, 2, 3})
	output, err = dropout.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	err = input.Set(0, 0, 10)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	requireMatrixValues(t, output, []float32{1, 2, 3})

	outputGradient = mustMatrix(t, 1, 3, []float32{4, 5, 6})
	inputGradient, err = dropout.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	err = outputGradient.Set(0, 0, 20)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float32{4, 5, 6})
}

func Test_Dropout_BackwardUsesTrainingMask(t *testing.T) {
	var (
		dropout        *layer.Dropout
		input          *matrix.Matrix
		output         *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		outputValues   []float32
		inputValues    []float32
		maskValues     []float32
		index          int
		err            error
	)

	dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(11)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	input = mustMatrix(t, 2, 3, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	output, err = dropout.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	inputValues = mustDropoutValues(t, input)
	outputValues = mustDropoutValues(t, output)
	maskValues = make([]float32, len(outputValues))
	for index = range maskValues {
		maskValues[index] = outputValues[index] / inputValues[index]
	}

	outputGradient = mustMatrix(t, 2, 3, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	inputGradient, err = dropout.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	for index = range maskValues {
		maskValues[index] *= float32(index + 1)
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

	input = mustMatrix(t, 1, 3, []float32{1, 2, 3})
	output, err = dropout.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	err = input.Set(0, 0, 10)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	requireMatrixValues(t, output, []float32{1, 2, 3})

	outputGradient = mustMatrix(t, 1, 3, []float32{4, 5, 6})
	inputGradient, err = dropout.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	err = outputGradient.Set(0, 0, 20)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float32{4, 5, 6})
}

func Test_Dropout_EvaluationModeIgnoresPreviousTrainingMask(t *testing.T) {
	var (
		dropout        *layer.Dropout
		input          *matrix.Matrix
		output         *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(7)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	input = mustMatrix(t, 2, 4, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	})
	_, err = dropout.Forward(input)
	if err != nil {
		t.Fatalf("training Forward returned error: %v", err)
	}

	dropout.SetTraining(false)
	input = mustMatrix(t, 2, 4, []float32{
		8, 7, 6, 5,
		4, 3, 2, 1,
	})
	output, err = dropout.Forward(input)
	if err != nil {
		t.Fatalf("evaluation Forward returned error: %v", err)
	}

	requireMatrixValues(t, output, []float32{
		8, 7, 6, 5,
		4, 3, 2, 1,
	})

	outputGradient = mustMatrix(t, 2, 4, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	})
	inputGradient, err = dropout.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	})
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

	inputGradient, err = dropout.Backward(mustMatrix(t, 1, 1, []float32{1}))
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

	input = mustMatrix(t, 2, 2, []float32{
		1, 2,
		3, 4,
	})
	_, err = dropout.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	_, err = dropout.Backward(mustMatrix(t, 1, 2, []float32{1, 2}))
	if err == nil {
		t.Fatal("Backward error = nil, want shape error")
	}

	if !strings.Contains(err.Error(), "got 1x2, want 2x2") {
		t.Fatalf("Backward error = %q, want received and expected shape", err.Error())
	}
}

func mustDropoutValues(tb testing.TB, m *matrix.Matrix) (values []float32) {
	var err error

	tb.Helper()

	values, err = m.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	return values
}
