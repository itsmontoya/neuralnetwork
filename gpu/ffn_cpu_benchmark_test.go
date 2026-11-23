//go:build !metal

package gpu

import (
	"fmt"
	"testing"
	"time"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

// Benchmark_FFNCPU measures prepared host projections and the CPU SwiGLU
// sequence with reusable tensor destinations. Build without the metal tag.
func Benchmark_FFNCPU(b *testing.B) {
	for _, rows := range ffnRowCounts {
		b.Run(fmt.Sprintf("M%d", rows), func(b *testing.B) {
			benchmarkCPUFFN(b, rows)
		})
	}
}

func benchmarkCPUFFN(b *testing.B, rows int64) {
	var (
		inputValues  = ffnInputs(rows)
		weightValues = ffnWeightValues()
		want         = ffnReference(inputValues, weightValues, ffnHidden, ffnIntermediate)
		weightShape  = [3][]int64{{ffnIntermediate, ffnHidden}, {ffnIntermediate, ffnHidden}, {ffnHidden, ffnIntermediate}}
		weights      [3]*tensor.PreparedRightTranspose
		err          error
	)
	var start = time.Now()
	for i, shape := range weightShape {
		weights[i], err = tensor.PrepareRightTranspose(shape, weightValues[i])
		requireError(b, err, nil)
	}
	var weightPreparation = time.Since(start)
	weightValues = [3][]float32{}

	start = time.Now()
	var (
		input, inputErr = tensor.FromFloat32([]int64{rows, ffnHidden}, inputValues)
		gate            = newFFNHostTensor(b, rows, ffnIntermediate)
		up              = newFFNHostTensor(b, rows, ffnIntermediate)
		sigmoid         = newFFNHostTensor(b, rows, ffnIntermediate)
		silu            = newFFNHostTensor(b, rows, ffnIntermediate)
		activated       = newFFNHostTensor(b, rows, ffnIntermediate)
		output          = newFFNHostTensor(b, rows, ffnHidden)
	)
	requireError(b, inputErr, nil)
	var allocation = time.Since(start)

	var config = tensor.GemmConfig{TransposeB: true, Alpha: 1}
	var run = func() {
		requireError(b, weights[0].GemmInto(input, nil, gate, config), nil)
		requireError(b, weights[1].GemmInto(input, nil, up, config), nil)
		requireError(b, tensor.SigmoidInto(gate, sigmoid), nil)
		requireError(b, tensor.MulInto(gate, sigmoid, silu), nil)
		requireError(b, tensor.MulInto(silu, up, activated), nil)
		requireError(b, weights[2].GemmInto(activated, nil, output, config), nil)
	}
	var check = func() {
		var got, readErr = output.Float32Values()
		requireError(b, readErr, nil)
		checkFFN(b, got, want)
	}
	start = time.Now()
	run()
	var first = time.Since(start)
	check()
	for range 3 {
		run()
		check()
	}

	b.ReportAllocs()
	for b.Loop() {
		run()
		b.StopTimer()
		check()
		b.StartTimer()
	}
	b.ReportMetric(float64(weightPreparation), "prepare-weights-ns")
	b.ReportMetric(float64(allocation), "allocate-io-scratch-ns")
	b.ReportMetric(float64(first), "first-ffn-ns")
}

func newFFNHostTensor(b *testing.B, rows, cols int64) (out *tensor.Tensor) {
	b.Helper()
	var result, err = tensor.New(tensor.Float32, []int64{rows, cols})
	requireError(b, err, nil)
	return result
}
