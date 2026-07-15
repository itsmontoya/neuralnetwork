package layer_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var allocationLayerResult *matrix.Matrix

func Test_ActivationDestinationSteadyStateAllocations(t *testing.T) {
	var tests []struct {
		name     string
		function activation.Activation
	}

	tests = []struct {
		name     string
		function activation.Activation
	}{
		{name: "ELU", function: activation.ELU{}},
		{name: "GELU", function: activation.GELU{}},
		{name: "LeakyReLU", function: activation.LeakyReLU{}},
		{name: "Linear", function: activation.Linear{}},
		{name: "ReLU", function: activation.ReLU{}},
		{name: "Sigmoid", function: activation.Sigmoid{}},
		{name: "Tanh", function: activation.Tanh{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				activationLayer *layer.Activation
				input           *matrix.Matrix
				outputGradient  *matrix.Matrix
				err             error
			)

			activationLayer, err = layer.NewActivation(tt.function)
			if err != nil {
				t.Fatalf("NewActivation returned error: %v", err)
			}

			input = allocationLayerMatrix(t, 8, 4)
			outputGradient = allocationLayerMatrix(t, 8, 4)
			if _, err = activationLayer.Forward(input); err != nil {
				t.Fatalf("Forward returned error: %v", err)
			}

			if _, err = activationLayer.Backward(outputGradient); err != nil {
				t.Fatalf("Backward returned error: %v", err)
			}

			requireMaxAllocs(t, tt.name+" Forward and Backward", 0, func() {
				if _, err = activationLayer.Forward(input); err != nil {
					panic(err)
				}

				if allocationLayerResult, err = activationLayer.Backward(outputGradient); err != nil {
					panic(err)
				}
			})
		})
	}
}

func Test_ActivationDestinationAlternatingShapeSteadyStateAllocations(t *testing.T) {
	var (
		tests []struct {
			name     string
			function activation.Activation
		}
		inputs          []*matrix.Matrix
		outputGradients []*matrix.Matrix
	)

	tests = []struct {
		name     string
		function activation.Activation
	}{
		{name: "ELU", function: activation.ELU{}},
		{name: "GELU", function: activation.GELU{}},
		{name: "LeakyReLU", function: activation.LeakyReLU{}},
		{name: "Linear", function: activation.Linear{}},
		{name: "ReLU", function: activation.ReLU{}},
		{name: "Sigmoid", function: activation.Sigmoid{}},
		{name: "Tanh", function: activation.Tanh{}},
	}
	inputs = allocationAlternatingLayerMatrices(t, 4)
	outputGradients = allocationAlternatingLayerMatrices(t, 4)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				activationLayer *layer.Activation
				err             error
			)

			activationLayer, err = layer.NewActivation(tt.function)
			if err != nil {
				t.Fatalf("NewActivation returned error: %v", err)
			}

			warmAlternatingLayerShapes(t, activationLayer, func(index int) {}, inputs, outputGradients)
			requireMaxAllocs(t, tt.name+" alternating shapes", 0, func() {
				warmAlternatingLayerShapes(t, activationLayer, func(index int) {}, inputs, outputGradients)
			})
			requireStableLayerScratch(t, activationLayer, func(index int) {}, inputs[0], outputGradients[0])
		})
	}
}

func Test_DenseSteadyStateAllocations(t *testing.T) {
	var (
		dense          *layer.Dense
		input          *matrix.Matrix
		output         *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	dense = allocationDense(t)
	input = allocationLayerMatrix(t, 4, 3)
	output, err = dense.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	requireMaxAllocs(t, "Dense.Forward", 0, func() {
		output, err = dense.Forward(input)
		if err != nil {
			panic(err)
		}
	})
	allocationLayerResult = output

	dense = allocationDense(t)
	outputGradient = allocationLayerMatrix(t, 4, 2)
	if _, err = dense.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	inputGradient, err = dense.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	if err = dense.ResetGradients(); err != nil {
		t.Fatalf("ResetGradients returned error: %v", err)
	}

	requireMaxAllocs(t, "Dense.Backward", 0, func() {
		inputGradient, err = dense.Backward(outputGradient)
		if err != nil {
			panic(err)
		}
	})
	allocationLayerResult = inputGradient
}

func Test_DropoutSteadyStateAllocations(t *testing.T) {
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

	input = allocationLayerMatrix(t, 8, 4)
	output, err = dropout.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	requireMaxAllocs(t, "Dropout.Forward", 0, func() {
		output, err = dropout.Forward(input)
		if err != nil {
			panic(err)
		}
	})
	allocationLayerResult = output

	dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(11)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	outputGradient = allocationLayerMatrix(t, 8, 4)
	if _, err = dropout.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	inputGradient, err = dropout.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	requireMaxAllocs(t, "Dropout.Backward", 0, func() {
		inputGradient, err = dropout.Backward(outputGradient)
		if err != nil {
			panic(err)
		}
	})
	allocationLayerResult = inputGradient
}

func Test_BatchNormalizationSteadyStateAllocations(t *testing.T) {
	var (
		batchNorm      *layer.BatchNormalization
		input          *matrix.Matrix
		output         *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	batchNorm, err = layer.NewBatchNormalization(4)
	if err != nil {
		t.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	input = allocationLayerMatrix(t, 8, 4)
	output, err = batchNorm.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	requireMaxAllocs(t, "BatchNormalization.Forward", 0, func() {
		output, err = batchNorm.Forward(input)
		if err != nil {
			panic(err)
		}
	})
	allocationLayerResult = output

	batchNorm, err = layer.NewBatchNormalization(4)
	if err != nil {
		t.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	outputGradient = allocationLayerMatrix(t, 8, 4)
	if _, err = batchNorm.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	inputGradient, err = batchNorm.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	if err = batchNorm.ResetGradients(); err != nil {
		t.Fatalf("ResetGradients returned error: %v", err)
	}

	requireMaxAllocs(t, "BatchNormalization.Backward", 0, func() {
		inputGradient, err = batchNorm.Backward(outputGradient)
		if err != nil {
			panic(err)
		}
	})
	allocationLayerResult = inputGradient
}

func Test_LayerAlternatingShapeSteadyStateAllocations(t *testing.T) {
	type testcase struct {
		name        string
		target      layer.Layer
		setScenario func(index int)
	}

	var (
		dense           *layer.Dense
		dropout         *layer.Dropout
		batchNorm       *layer.BatchNormalization
		inputs          []*matrix.Matrix
		outputGradients []*matrix.Matrix
		err             error
	)

	dense = allocationSquareDense(t)
	dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(17)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	batchNorm, err = layer.NewBatchNormalization(4)
	if err != nil {
		t.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	inputs = allocationAlternatingLayerMatrices(t, 4)
	outputGradients = allocationAlternatingLayerMatrices(t, 4)
	tests := []testcase{
		{
			name:   "Dense",
			target: dense,
			setScenario: func(index int) {
			},
		},
		{
			name:   "Dropout",
			target: dropout,
			setScenario: func(index int) {
				dropout.SetTraining(index < 2)
			},
		},
		{
			name:   "BatchNormalization",
			target: batchNorm,
			setScenario: func(index int) {
				batchNorm.SetTraining(index < 2)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			warmAlternatingLayerShapes(t, tt.target, tt.setScenario, inputs, outputGradients)
			requireMaxAllocs(t, tt.name+" alternating shapes", 0, func() {
				warmAlternatingLayerShapes(t, tt.target, tt.setScenario, inputs, outputGradients)
			})
			requireStableLayerScratch(t, tt.target, tt.setScenario, inputs[0], outputGradients[0])
		})
	}
}

func Test_DenseScratchPoolsEvictOldestShapeAndRemainCorrect(t *testing.T) {
	var (
		dense              *layer.Dense
		inputs             []*matrix.Matrix
		outputGradients    []*matrix.Matrix
		firstOutput        *matrix.Matrix
		firstInputGradient *matrix.Matrix
		output             *matrix.Matrix
		inputGradient      *matrix.Matrix
		err                error
		index              int
	)

	dense = allocationSquareDense(t)
	inputs = allocationEvictionLayerMatrices(t, 4)
	outputGradients = allocationEvictionLayerMatrices(t, 4)
	for index = range inputs {
		if output, err = dense.Forward(inputs[index]); err != nil {
			t.Fatalf("Forward shape %d returned error: %v", index, err)
		}

		if inputGradient, err = dense.Backward(outputGradients[index]); err != nil {
			t.Fatalf("Backward shape %d returned error: %v", index, err)
		}

		if index == 0 {
			firstOutput = output
			firstInputGradient = inputGradient
		}
	}

	if output, err = dense.Forward(inputs[0]); err != nil {
		t.Fatalf("Forward after eviction returned error: %v", err)
	}

	if inputGradient, err = dense.Backward(outputGradients[0]); err != nil {
		t.Fatalf("Backward after eviction returned error: %v", err)
	}

	if output == firstOutput {
		t.Fatal("Forward reused the oldest output after five shapes")
	}

	if inputGradient == firstInputGradient {
		t.Fatal("Backward reused the oldest input gradient after five shapes")
	}

	requireMatrixValues(t, output, allocationMatrixValues(t, inputs[0]))
	requireMatrixValues(t, inputGradient, allocationMatrixValues(t, outputGradients[0]))
}

func Test_DropoutScratchPoolsEvictOldestShapeAndRemainCorrect(t *testing.T) {
	var (
		dropout             *layer.Dropout
		inputs              []*matrix.Matrix
		outputGradients     []*matrix.Matrix
		firstOutput         *matrix.Matrix
		firstInputGradient  *matrix.Matrix
		output              *matrix.Matrix
		inputGradient       *matrix.Matrix
		inputValues         []float32
		outputValues        []float32
		gradientValues      []float32
		inputGradientValues []float32
		mask                float32
		err                 error
		index               int
	)

	dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(23)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	inputs = allocationEvictionLayerMatrices(t, 4)
	outputGradients = allocationEvictionLayerMatrices(t, 4)
	for index = range inputs {
		if output, err = dropout.Forward(inputs[index]); err != nil {
			t.Fatalf("Forward shape %d returned error: %v", index, err)
		}

		if inputGradient, err = dropout.Backward(outputGradients[index]); err != nil {
			t.Fatalf("Backward shape %d returned error: %v", index, err)
		}

		if index == 0 {
			firstOutput = output
			firstInputGradient = inputGradient
		}
	}

	if output, err = dropout.Forward(inputs[0]); err != nil {
		t.Fatalf("Forward after eviction returned error: %v", err)
	}

	if inputGradient, err = dropout.Backward(outputGradients[0]); err != nil {
		t.Fatalf("Backward after eviction returned error: %v", err)
	}

	if output == firstOutput {
		t.Fatal("Forward reused the oldest output after five shapes")
	}

	if inputGradient == firstInputGradient {
		t.Fatal("Backward reused the oldest input gradient after five shapes")
	}

	inputValues = allocationMatrixValues(t, inputs[0])
	outputValues = allocationMatrixValues(t, output)
	gradientValues = allocationMatrixValues(t, outputGradients[0])
	inputGradientValues = allocationMatrixValues(t, inputGradient)
	for index = range inputValues {
		mask = outputValues[index] / inputValues[index]
		if mask != 0 && mask != 2 {
			t.Fatalf("dropout mask at index %d = %g, want 0 or 2", index, mask)
		}

		if inputGradientValues[index] != gradientValues[index]*mask {
			t.Fatalf(
				"input gradient at index %d = %g, want %g",
				index,
				inputGradientValues[index],
				gradientValues[index]*mask,
			)
		}
	}
}

func Test_BatchNormalizationScratchPoolsEvictOldestShapeAndRemainCorrect(t *testing.T) {
	var (
		batchNorm          *layer.BatchNormalization
		reference          *layer.BatchNormalization
		inputs             []*matrix.Matrix
		outputGradients    []*matrix.Matrix
		firstOutput        *matrix.Matrix
		firstInputGradient *matrix.Matrix
		output             *matrix.Matrix
		inputGradient      *matrix.Matrix
		wantOutput         *matrix.Matrix
		wantInputGradient  *matrix.Matrix
		err                error
		index              int
	)

	batchNorm, err = layer.NewBatchNormalization(4)
	if err != nil {
		t.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	reference, err = layer.NewBatchNormalization(4)
	if err != nil {
		t.Fatalf("NewBatchNormalization reference returned error: %v", err)
	}

	inputs = allocationEvictionLayerMatrices(t, 4)
	outputGradients = allocationEvictionLayerMatrices(t, 4)
	if wantOutput, err = reference.Forward(inputs[0]); err != nil {
		t.Fatalf("reference Forward returned error: %v", err)
	}

	if wantInputGradient, err = reference.Backward(outputGradients[0]); err != nil {
		t.Fatalf("reference Backward returned error: %v", err)
	}

	for index = range inputs {
		if output, err = batchNorm.Forward(inputs[index]); err != nil {
			t.Fatalf("Forward shape %d returned error: %v", index, err)
		}

		if inputGradient, err = batchNorm.Backward(outputGradients[index]); err != nil {
			t.Fatalf("Backward shape %d returned error: %v", index, err)
		}

		if index == 0 {
			firstOutput = output
			firstInputGradient = inputGradient
		}
	}

	if output, err = batchNorm.Forward(inputs[0]); err != nil {
		t.Fatalf("Forward after eviction returned error: %v", err)
	}

	if inputGradient, err = batchNorm.Backward(outputGradients[0]); err != nil {
		t.Fatalf("Backward after eviction returned error: %v", err)
	}

	if output == firstOutput {
		t.Fatal("Forward reused the oldest output after five shapes")
	}

	if inputGradient == firstInputGradient {
		t.Fatal("Backward reused the oldest input gradient after five shapes")
	}

	requireMatrixValues(t, output, allocationMatrixValues(t, wantOutput))
	requireMatrixValues(t, inputGradient, allocationMatrixValues(t, wantInputGradient))
}

func allocationSquareDense(tb testing.TB) (dense *layer.Dense) {
	tb.Helper()

	dense = mustDense(
		tb,
		4,
		4,
		[]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
		[]float32{0, 0, 0, 0},
	)
	return dense
}

func allocationAlternatingLayerMatrices(tb testing.TB, cols int) (matrices []*matrix.Matrix) {
	var (
		rows  []int
		index int
	)

	tb.Helper()

	rows = []int{128, 17, 1024, 257}
	matrices = make([]*matrix.Matrix, len(rows))
	for index = range rows {
		matrices[index] = allocationLayerMatrix(tb, rows[index], cols)
	}

	return matrices
}

func allocationEvictionLayerMatrices(tb testing.TB, cols int) (matrices []*matrix.Matrix) {
	var (
		rows  []int
		index int
	)

	tb.Helper()

	rows = []int{2, 3, 4, 5, 6}
	matrices = make([]*matrix.Matrix, len(rows))
	for index = range rows {
		matrices[index] = allocationLayerMatrix(tb, rows[index], cols)
	}

	return matrices
}

func warmAlternatingLayerShapes(
	tb testing.TB,
	target layer.Layer,
	setScenario func(index int),
	inputs, outputGradients []*matrix.Matrix,
) {
	var (
		err   error
		index int
	)

	tb.Helper()

	for index = range inputs {
		setScenario(index)
		if _, err = target.Forward(inputs[index]); err != nil {
			panic(err)
		}

		if allocationLayerResult, err = target.Backward(outputGradients[index]); err != nil {
			panic(err)
		}
	}
}

func requireStableLayerScratch(
	tb testing.TB,
	target layer.Layer,
	setScenario func(index int),
	input, outputGradient *matrix.Matrix,
) {
	var (
		firstOutput         *matrix.Matrix
		secondOutput        *matrix.Matrix
		firstInputGradient  *matrix.Matrix
		secondInputGradient *matrix.Matrix
		err                 error
	)

	tb.Helper()

	setScenario(0)
	if firstOutput, err = target.Forward(input); err != nil {
		tb.Fatalf("first Forward returned error: %v", err)
	}

	if firstInputGradient, err = target.Backward(outputGradient); err != nil {
		tb.Fatalf("first Backward returned error: %v", err)
	}

	if secondOutput, err = target.Forward(input); err != nil {
		tb.Fatalf("second Forward returned error: %v", err)
	}

	if secondInputGradient, err = target.Backward(outputGradient); err != nil {
		tb.Fatalf("second Backward returned error: %v", err)
	}

	if secondOutput != firstOutput {
		tb.Fatal("Forward did not reuse stable-shape output scratch")
	}

	if secondInputGradient != firstInputGradient {
		tb.Fatal("Backward did not reuse stable-shape input-gradient scratch")
	}
}

func allocationMatrixValues(tb testing.TB, value *matrix.Matrix) (values []float32) {
	var err error

	tb.Helper()

	values, err = value.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	return values
}

func allocationDense(tb testing.TB) (dense *layer.Dense) {
	tb.Helper()

	dense = mustDense(
		tb,
		3,
		2,
		[]float32{
			0.5, -1,
			1.5, 0,
			-0.5, 2,
		},
		[]float32{0.1, -0.2},
	)
	return dense
}

func allocationLayerMatrix(tb testing.TB, rows, cols int) (m *matrix.Matrix) {
	var (
		values []float32
		index  int
	)

	tb.Helper()

	values = make([]float32, rows*cols)
	for index = range values {
		values[index] = float32((index%11)+1) / 11
	}

	m = mustMatrix(tb, rows, cols, values)
	return m
}

func requireMaxAllocs(tb testing.TB, name string, max float64, run func()) {
	var got float64

	tb.Helper()

	got = testing.AllocsPerRun(100, run)
	if got > max {
		tb.Fatalf("%s allocations = %.0f, want <= %.0f", name, got, max)
	}
}
