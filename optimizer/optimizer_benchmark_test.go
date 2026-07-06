package optimizer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

var benchmarkOptimizerParameters []*optimizer.Parameter

func Benchmark_SGDUpdate_SteadyState(b *testing.B) {
	var (
		optimizerRule *optimizer.SGD
		parameters    []*optimizer.Parameter
		gradients     []*matrix.Matrix
		err           error
	)

	parameters, gradients = benchmarkOptimizerParametersAndGradients(b)
	optimizerRule, err = optimizer.NewSGD(0.01)
	if err != nil {
		b.Fatalf("NewSGD returned error: %v", err)
	}

	benchmarkOptimizerUpdate(b, optimizerRule, parameters, gradients)
}

func Benchmark_MomentumUpdate_SteadyState(b *testing.B) {
	var (
		optimizerRule *optimizer.Momentum
		parameters    []*optimizer.Parameter
		gradients     []*matrix.Matrix
		err           error
	)

	parameters, gradients = benchmarkOptimizerParametersAndGradients(b)
	optimizerRule, err = optimizer.NewMomentum(0.01)
	if err != nil {
		b.Fatalf("NewMomentum returned error: %v", err)
	}

	benchmarkOptimizerUpdate(b, optimizerRule, parameters, gradients)
}

func Benchmark_AdamUpdate_SteadyState(b *testing.B) {
	var (
		optimizerRule *optimizer.Adam
		parameters    []*optimizer.Parameter
		gradients     []*matrix.Matrix
		err           error
	)

	parameters, gradients = benchmarkOptimizerParametersAndGradients(b)
	optimizerRule, err = optimizer.NewAdam(0.001)
	if err != nil {
		b.Fatalf("NewAdam returned error: %v", err)
	}

	benchmarkOptimizerUpdate(b, optimizerRule, parameters, gradients)
}

func benchmarkOptimizerUpdate(
	b *testing.B,
	optimizerRule optimizer.Optimizer,
	parameters []*optimizer.Parameter,
	gradients []*matrix.Matrix,
) {
	var (
		err   error
		index int
	)

	if err = benchmarkAccumulateGradients(b, parameters, gradients); err != nil {
		b.Fatalf("AccumulateGradient returned error: %v", err)
	}

	if err = optimizerRule.Update(parameters); err != nil {
		b.Fatalf("Update returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if err = benchmarkAccumulateGradients(b, parameters, gradients); err != nil {
			b.Fatalf("AccumulateGradient returned error: %v", err)
		}

		if err = optimizerRule.Update(parameters); err != nil {
			b.Fatalf("Update returned error: %v", err)
		}
	}

	benchmarkOptimizerParameters = parameters
}

func benchmarkOptimizerParametersAndGradients(tb testing.TB) (parameters []*optimizer.Parameter, gradients []*matrix.Matrix) {
	var (
		parameter *optimizer.Parameter
		gradient  *matrix.Matrix
	)

	tb.Helper()

	parameter, gradient = benchmarkOptimizerParameter(tb, 32, 64, 1)
	parameters = append(parameters, parameter)
	gradients = append(gradients, gradient)

	parameter, gradient = benchmarkOptimizerParameter(tb, 1, 64, 3)
	parameters = append(parameters, parameter)
	gradients = append(gradients, gradient)

	parameter, gradient = benchmarkOptimizerParameter(tb, 64, 16, 5)
	parameters = append(parameters, parameter)
	gradients = append(gradients, gradient)

	parameter, gradient = benchmarkOptimizerParameter(tb, 1, 16, 7)
	parameters = append(parameters, parameter)
	gradients = append(gradients, gradient)

	return parameters, gradients
}

func benchmarkOptimizerParameter(tb testing.TB, rows, cols, offset int) (parameter *optimizer.Parameter, gradient *matrix.Matrix) {
	var (
		values *matrix.Matrix
		err    error
	)

	tb.Helper()

	values = benchmarkOptimizerMatrix(tb, rows, cols, offset)
	gradient = benchmarkOptimizerMatrix(tb, rows, cols, offset+11)
	parameter, err = optimizer.NewParameter(values)
	if err != nil {
		tb.Fatalf("NewParameter returned error: %v", err)
	}

	return parameter, gradient
}

func benchmarkOptimizerMatrix(tb testing.TB, rows, cols, offset int) (m *matrix.Matrix) {
	var (
		values []float64
		err    error
		index  int
	)

	tb.Helper()

	values = make([]float64, rows*cols)
	for index = range values {
		values[index] = float64((index+offset)%37)/37 - 0.5
	}

	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}

func benchmarkAccumulateGradients(tb testing.TB, parameters []*optimizer.Parameter, gradients []*matrix.Matrix) (err error) {
	var (
		index     int
		parameter *optimizer.Parameter
	)

	tb.Helper()

	for index, parameter = range parameters {
		if err = parameter.AccumulateGradient(gradients[index]); err != nil {
			return err
		}
	}

	return nil
}
