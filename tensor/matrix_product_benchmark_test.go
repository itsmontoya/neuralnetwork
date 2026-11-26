package tensor_test

import (
	"fmt"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Benchmark_MatMulInto(b *testing.B) {
	var tests = []struct {
		name              string
		rows, inner, cols int
	}{
		{name: "M1_1024x4096", rows: 1, inner: 1024, cols: 4096},
		{name: "M32_1024x4096", rows: 32, inner: 1024, cols: 4096},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			var (
				left, right, output *tensor.Tensor
				err                 error
			)

			if left, err = tensor.FromFloat32([]int64{int64(tt.rows), int64(tt.inner)}, matrixProductBenchmarkValues(tt.rows*tt.inner)); err != nil {
				b.Fatal(err)
			}
			if right, err = tensor.FromFloat32([]int64{int64(tt.inner), int64(tt.cols)}, matrixProductBenchmarkValues(tt.inner*tt.cols)); err != nil {
				b.Fatal(err)
			}
			if output, err = tensor.New(tensor.Float32, []int64{int64(tt.rows), int64(tt.cols)}); err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err = tensor.MatMulInto(left, right, output); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func Benchmark_GemmInto(b *testing.B) {
	var tests = []struct {
		name              string
		rows, inner, cols int
	}{
		{name: "M1_1024x4096", rows: 1, inner: 1024, cols: 4096},
		{name: "M32_1024x4096", rows: 32, inner: 1024, cols: 4096},
		{name: "M32_2048x8192", rows: 32, inner: 2048, cols: 8192},
	}

	for _, tt := range tests {
		b.Run(fmt.Sprintf("Dense/%s", tt.name), func(b *testing.B) {
			var (
				left, right, bias, output *tensor.Tensor
				config                    = tensor.GemmConfig{TransposeB: true, Alpha: 1, Beta: 1}
				err                       error
			)

			if left, err = tensor.FromFloat32([]int64{int64(tt.rows), int64(tt.inner)}, matrixProductBenchmarkValues(tt.rows*tt.inner)); err != nil {
				b.Fatal(err)
			}
			if right, err = tensor.FromFloat32([]int64{int64(tt.cols), int64(tt.inner)}, matrixProductBenchmarkValues(tt.cols*tt.inner)); err != nil {
				b.Fatal(err)
			}
			if bias, err = tensor.FromFloat32([]int64{int64(tt.cols)}, matrixProductBenchmarkValues(tt.cols)); err != nil {
				b.Fatal(err)
			}
			if output, err = tensor.New(tensor.Float32, []int64{int64(tt.rows), int64(tt.cols)}); err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err = tensor.GemmInto(left, right, bias, output, config); err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run(fmt.Sprintf("Prepared/%s", tt.name), func(b *testing.B) {
			var (
				left, bias, output *tensor.Tensor
				weight             *tensor.PreparedRightTranspose
				config             = tensor.GemmConfig{TransposeB: true, Alpha: 1, Beta: 1}
				err                error
			)

			if left, err = tensor.FromFloat32([]int64{int64(tt.rows), int64(tt.inner)}, matrixProductBenchmarkValues(tt.rows*tt.inner)); err != nil {
				b.Fatal(err)
			}
			if weight, err = tensor.PrepareRightTranspose([]int64{int64(tt.cols), int64(tt.inner)}, matrixProductBenchmarkValues(tt.cols*tt.inner)); err != nil {
				b.Fatal(err)
			}
			if bias, err = tensor.FromFloat32([]int64{int64(tt.cols)}, matrixProductBenchmarkValues(tt.cols)); err != nil {
				b.Fatal(err)
			}
			if output, err = tensor.New(tensor.Float32, []int64{int64(tt.rows), int64(tt.cols)}); err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err = weight.GemmInto(left, bias, output, config); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func matrixProductBenchmarkValues(size int) (values []float32) {
	values = make([]float32, size)
	for index := range values {
		values[index] = float32(index%127-63) / 127
	}
	return values
}
