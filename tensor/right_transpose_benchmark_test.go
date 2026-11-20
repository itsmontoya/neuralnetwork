package tensor_test

import (
	"fmt"
	"runtime"
	"sync/atomic"
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
		{name: "SmallSingleRow", rows: 1, inner: 2048, cols: 64},
		{name: "Irregular", rows: 17, inner: 33, cols: 19},
		{name: "BelowParallelThreshold", rows: 1, inner: 2048, cols: 511},
		{name: "SingleRowTail", rows: 1, inner: 2048, cols: 513},
		{name: "PartitionTail", rows: 1, inner: 2049, cols: 775},
		{name: "MultiRowTail", rows: 2, inner: 2048, cols: 513},
		{name: "MultiRowTail", rows: 3, inner: 2048, cols: 513},
		{name: "MultiRowTail", rows: 4, inner: 2048, cols: 513},
		{name: "MultiRowTail", rows: 5, inner: 2048, cols: 513},
		{name: "OddContraction", rows: 2, inner: 2049, cols: 513},
		{name: "QAndOutput", rows: 1, inner: 2048, cols: 2048},
		{name: "QAndOutput", rows: 6, inner: 2048, cols: 2048},
		{name: "KAndV", rows: 1, inner: 2048, cols: 512},
		{name: "KAndV", rows: 6, inner: 2048, cols: 512},
		{name: "FFNGateAndUp", rows: 1, inner: 2048, cols: 8192},
		{name: "FFNGateAndUp", rows: 3, inner: 2048, cols: 8192},
		{name: "FFNGateAndUp", rows: 6, inner: 2048, cols: 8192},
		{name: "FFNGateAndUp", rows: 24, inner: 2048, cols: 8192},
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
			b.SetBytes(int64(tt.inner) * int64(tt.cols) * 4)
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
	b.SetBytes(int64(inner) * int64(cols) * 4)
	for b.Loop() {
		if err = tensor.GemmInto(left, right, nil, output, config); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkRightTransposeConcurrent measures FFN projections sharing weights,
// with one caller per scheduler thread and disjoint destinations.
func BenchmarkRightTransposeConcurrent(b *testing.B) {
	const inner, cols = 2048, 8192
	var callers = runtime.GOMAXPROCS(0)
	b.Run("Kernel", func(b *testing.B) {
		var (
			left    = rightTransposeBenchmarkValues(inner)
			right   = rightTransposeBenchmarkValues(inner * cols)
			outputs = make([]float32, callers*cols)
			next    atomic.Int64
		)
		b.ReportAllocs()
		b.SetBytes(inner * cols * 4)
		b.ResetTimer()
		b.RunParallel(func(work *testing.PB) {
			var offset = int(next.Add(1)-1) * cols
			var output = outputs[offset : offset+cols]
			for work.Next() {
				float32product.RightTranspose(left, right, output, 1, inner, cols)
			}
		})
	})
	b.Run("GemmInto", func(b *testing.B) {
		var (
			left, right *tensor.Tensor
			config      = tensor.GemmConfig{TransposeB: true, Alpha: 1}
			outputs     = make([]*tensor.Tensor, callers)
			next        atomic.Int64
			err         error
		)
		if left, err = tensor.FromFloat32([]int64{1, inner}, rightTransposeBenchmarkValues(inner)); err != nil {
			b.Fatal(err)
		}
		if right, err = tensor.FromFloat32([]int64{cols, inner}, rightTransposeBenchmarkValues(inner*cols)); err != nil {
			b.Fatal(err)
		}
		for index := range outputs {
			if outputs[index], err = tensor.New(tensor.Float32, []int64{1, cols}); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportAllocs()
		b.SetBytes(inner * cols * 4)
		b.ResetTimer()
		b.RunParallel(func(work *testing.PB) {
			var (
				output    = outputs[next.Add(1)-1]
				workerErr error
			)
			for work.Next() {
				if workerErr = tensor.GemmInto(left, right, nil, output, config); workerErr != nil {
					b.Error(workerErr)
					return
				}
			}
		})
	})
}

func rightTransposeBenchmarkValues(size int) (values []float32) {
	values = make([]float32, size)
	for index := range values {
		values[index] = float32(index%127-63) / 127
	}
	return values
}
