package tensor_test

import (
	"fmt"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/float32product"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

func BenchmarkRightTranspose(b *testing.B) {
	var tests = []struct {
		name              string
		rows, inner, cols int
	}{
		{name: "Small", rows: 2, inner: 2, cols: 2},
		{name: "Irregular", rows: 17, inner: 33, cols: 19},
		{name: "SingleRowTail", rows: 1, inner: 2048, cols: 513},
		{name: "QAndOutput", rows: 1, inner: 2048, cols: 2048},
		{name: "QAndOutput", rows: 6, inner: 2048, cols: 2048},
		{name: "KAndV", rows: 1, inner: 2048, cols: 512},
		{name: "KAndV", rows: 6, inner: 2048, cols: 512},
		{name: "FFNGateAndUp", rows: 1, inner: 2048, cols: 8192},
		{name: "FFNGateAndUp", rows: 6, inner: 2048, cols: 8192},
		{name: "FFNDown", rows: 1, inner: 8192, cols: 2048},
		{name: "FFNDown", rows: 6, inner: 8192, cols: 2048},
		{name: "Vocabulary", rows: 1, inner: 2048, cols: 128256},
		{name: "Vocabulary", rows: 6, inner: 2048, cols: 128256},
	}
	for _, tt := range tests {
		name := fmt.Sprintf("%s/M%d", tt.name, tt.rows)
		b.Run("Kernel/"+name, func(b *testing.B) {
			var (
				left   = rightTransposeBenchmarkValues(tt.rows * tt.inner)
				right  = rightTransposeBenchmarkValues(tt.cols * tt.inner)
				output = make([]float32, tt.rows*tt.cols)
			)
			b.ReportAllocs()
			for b.Loop() {
				float32product.RightTranspose(left, right, output, tt.rows, tt.inner, tt.cols)
			}
		})
		b.Run("GemmInto/"+name, func(b *testing.B) {
			benchmarkRightTransposeGemm(b, tt.rows, tt.inner, tt.cols)
		})
	}
}

func benchmarkRightTransposeGemm(b *testing.B, rows, inner, cols int) {
	var (
		left, right, output *tensor.Tensor
		config              = tensor.GemmConfig{TransposeB: true, Alpha: 1}
		err                 error
	)
	if left, err = tensor.FromFloat32([]int64{int64(rows), int64(inner)}, rightTransposeBenchmarkValues(rows*inner)); err != nil {
		b.Fatal(err)
	}
	if right, err = tensor.FromFloat32([]int64{int64(cols), int64(inner)}, rightTransposeBenchmarkValues(cols*inner)); err != nil {
		b.Fatal(err)
	}
	if output, err = tensor.New(tensor.Float32, []int64{int64(rows), int64(cols)}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if err = tensor.GemmInto(left, right, nil, output, config); err != nil {
			b.Fatal(err)
		}
	}
}

func rightTransposeBenchmarkValues(size int) (values []float32) {
	values = make([]float32, size)
	for index := range values {
		values[index] = float32(index%127-63) / 127
	}
	return values
}
