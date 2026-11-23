package gpu

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

const ffnHidden, ffnIntermediate = 2048, 8192

var ffnRowCounts = [...]int64{1, 27, 28, 29}

func ffnInputs(rows int64) (input []float32) {
	var random = rand.New(rand.NewPCG(0x6e6e05, 0x2048))
	input = make([]float32, rows*ffnHidden)
	for row := range rows {
		var values = input[row*ffnHidden : (row+1)*ffnHidden]
		var sumSquares float64
		for i := range values {
			var value = 2*random.Float32() - 1
			values[i] = value
			sumSquares += float64(value) * float64(value)
		}
		var rms = math.Sqrt(sumSquares / ffnHidden)
		for i := range values {
			values[i] = float32(float64(values[i]) / rms)
		}
	}
	return input
}

func ffnWeightValues() (weights [3][]float32) {
	var random = rand.New(rand.NewPCG(0x6e6e05, 0x8192))
	for i, inner := range [...]int{ffnHidden, ffnHidden, ffnIntermediate} {
		var scale = float32(1 / math.Sqrt(float64(inner)))
		weights[i] = make([]float32, ffnHidden*ffnIntermediate)
		for j := range weights[i] {
			weights[i][j] = (2*random.Float32() - 1) * scale
		}
	}
	return weights
}

func ffnReference(input []float32, weights [3][]float32, hidden, intermediate int) (output []float64) {
	output = make([]float64, len(input))
	var activation = make([]float64, intermediate)
	for row := 0; row < len(input); row += hidden {
		var source = input[row : row+hidden]
		for col := range activation {
			var (
				gateWeight = weights[0][col*hidden : (col+1)*hidden]
				upWeight   = weights[1][col*hidden : (col+1)*hidden]
				gate, up   float64
			)
			for k, value := range source {
				gate += float64(value) * float64(gateWeight[k])
				up += float64(value) * float64(upWeight[k])
			}
			activation[col] = (gate / (1 + math.Exp(-gate))) * up
		}
		for col := range hidden {
			var (
				weight = weights[2][col*intermediate : (col+1)*intermediate]
				sum    float64
			)
			for k, value := range activation {
				sum += value * float64(weight[k])
			}
			output[row+col] = sum
		}
	}
	return output
}

func checkFFN(t testing.TB, got []float32, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("output has %d elements, want %d", len(got), len(want))
	}
	for i, expected := range want {
		var (
			delta = math.Abs(float64(got[i]) - expected)
			limit = 5e-6 + 3e-5*math.Abs(expected)
		)
		if math.IsNaN(delta) || delta > limit {
			t.Fatalf("element %d: got %.9g, FP64 reference %.9g, absolute error %.3g exceeds %.3g", i, got[i], expected, delta, limit)
		}
	}
}

func logFFNError(t testing.TB, got []float32, want []float64) {
	t.Helper()
	var deltas = make([]float64, len(want))
	for i, expected := range want {
		deltas[i] = math.Abs(float64(got[i]) - expected)
	}
	slices.Sort(deltas)
	var last = len(deltas) - 1
	t.Logf("absolute error vs FP64 (%d elements) p50 %.3g, p95 %.3g, p99 %.3g, max %.3g",
		len(deltas), deltas[last*50/100], deltas[last*95/100], deltas[last*99/100], deltas[last])
}
