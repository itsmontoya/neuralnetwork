package optimizer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

const epsilon = 1e-5

func Test_NewParameter(t *testing.T) {
	var (
		source         *matrix.Matrix
		got            *optimizer.Parameter
		values         []float32
		gradientValues []float32
		err            error
	)

	source = mustMatrix(t, 1, 3, []float32{1, 2, 3})
	got, err = optimizer.NewParameter(source)
	if err != nil {
		t.Fatalf("NewParameter returned error: %v", err)
	}

	values, err = got.Values().Values()
	if err != nil {
		t.Fatalf("Values().Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(t, values, []float32{1, 2, 3}, epsilon)

	gradientValues, err = got.Gradient().Values()
	if err != nil {
		t.Fatalf("Gradient().Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(t, gradientValues, []float32{0, 0, 0}, epsilon)

	err = source.Set(0, 0, 99)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	values, err = got.Values().Values()
	if err != nil {
		t.Fatalf("Values().Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(t, values, []float32{1, 2, 3}, epsilon)
}

func Test_NewParameter_ValidatesValues(t *testing.T) {
	var (
		got *optimizer.Parameter
		err error
	)

	got, err = optimizer.NewParameter(nil)
	if err == nil {
		t.Fatal("NewParameter error = nil, want error")
	}

	if got != nil {
		t.Fatal("NewParameter returned parameter on error")
	}
}

func Test_Parameter_AccumulateGradient(t *testing.T) {
	var (
		parameter      *optimizer.Parameter
		firstGradient  *matrix.Matrix
		secondGradient *matrix.Matrix
		values         []float32
		err            error
	)

	parameter = mustParameter(t, 1, 3, []float32{1, 2, 3})
	firstGradient = mustMatrix(t, 1, 3, []float32{0.1, 0.2, 0.3})
	secondGradient = mustMatrix(t, 1, 3, []float32{0.4, 0.5, 0.6})

	err = parameter.AccumulateGradient(firstGradient)
	if err != nil {
		t.Fatalf("AccumulateGradient returned error: %v", err)
	}

	err = parameter.AccumulateGradient(secondGradient)
	if err != nil {
		t.Fatalf("AccumulateGradient returned error: %v", err)
	}

	values, err = parameter.Gradient().Values()
	if err != nil {
		t.Fatalf("Gradient().Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(t, values, []float32{0.5, 0.7, 0.9}, epsilon)
}

func Test_Parameter_AccumulateGradient_ValidatesShape(t *testing.T) {
	var (
		parameter *optimizer.Parameter
		gradient  *matrix.Matrix
		err       error
	)

	parameter = mustParameter(t, 1, 3, []float32{1, 2, 3})
	gradient = mustMatrix(t, 1, 2, []float32{0.1, 0.2})

	err = parameter.AccumulateGradient(gradient)
	if err == nil {
		t.Fatal("AccumulateGradient error = nil, want error")
	}
}

func Test_Parameter_ResetGradient(t *testing.T) {
	var (
		parameter *optimizer.Parameter
		gradient  *matrix.Matrix
		values    []float32
		err       error
	)

	parameter = mustParameter(t, 1, 3, []float32{1, 2, 3})
	gradient = mustMatrix(t, 1, 3, []float32{0.1, 0.2, 0.3})

	err = parameter.AccumulateGradient(gradient)
	if err != nil {
		t.Fatalf("AccumulateGradient returned error: %v", err)
	}

	err = parameter.ResetGradient()
	if err != nil {
		t.Fatalf("ResetGradient returned error: %v", err)
	}

	values, err = parameter.Gradient().Values()
	if err != nil {
		t.Fatalf("Gradient().Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(t, values, []float32{0, 0, 0}, epsilon)
}

func mustParameter(tb testing.TB, rows, cols int, values []float32) (parameter *optimizer.Parameter) {
	var (
		initialValues *matrix.Matrix
		err           error
	)

	tb.Helper()

	initialValues = mustMatrix(tb, rows, cols, values)
	parameter, err = optimizer.NewParameter(initialValues)
	if err != nil {
		tb.Fatalf("NewParameter returned error: %v", err)
	}

	return parameter
}

func mustMatrix(tb testing.TB, rows, cols int, values []float32) (m *matrix.Matrix) {
	var err error

	tb.Helper()

	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}
