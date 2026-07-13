package layer_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var allocationLayerResult *matrix.Matrix

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
