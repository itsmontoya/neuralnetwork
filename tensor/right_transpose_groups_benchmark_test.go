package tensor_test

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/float32product"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

// BenchmarkRightTransposeGroups measures repeated and disjoint weight operands.
// Throughput uses the weight operand size so it can be compared across dispatchers.
func BenchmarkRightTransposeGroups(b *testing.B) {
	var shapes = []struct {
		name                   string
		inner, cols, rotations int
	}{
		{name: "KV", inner: 2048, cols: 512, rotations: 32},
		{name: "QO", inner: 2048, cols: 2048, rotations: 8},
		{name: "GateUp", inner: 2048, cols: 8192, rotations: 2},
		{name: "Down", inner: 8192, cols: 2048, rotations: 2},
		{name: "Vocabulary", inner: 2048, cols: 128256},
	}
	for _, shape := range shapes {
		var rowCounts = []int{1, 2, 3, 4, 5, 6, 27, 28, 29, 30}
		if shape.rotations == 0 {
			rowCounts = []int{1, 3, 4, 6}
		}
		for _, rows := range rowCounts {
			var name = fmt.Sprintf("%s/M%d", shape.name, rows)
			benchmarkRightTransposeGroupShape(b, "Reuse/"+name, rows, shape.inner, shape.cols, 1)
			if shape.rotations > 0 {
				benchmarkRightTransposeGroupShape(b, "Rotate/"+name, rows, shape.inner, shape.cols, shape.rotations)
			}
		}
	}
	for _, cols := range []int{0, 1, 2, 3, 4, 5, 63, 64, 127, 128, 255, 256, 511, 512, 513, 8193} {
		var inner = 2048
		if cols <= 5 {
			inner = 31
		}
		for _, rows := range []int{2, 3, 4, 5} {
			var name = fmt.Sprintf("Boundary/K%d/N%d/M%d", inner, cols, rows)
			benchmarkRightTransposeGroupShape(b, name, rows, inner, cols, 1)
		}
	}
	benchmarkRightTransposeGroupShape(b, "Boundary/K2049/N513/M7", 7, 2049, 513, 1)
}

func benchmarkRightTransposeGroupShape(b *testing.B, name string, rows, inner, cols, rotations int) {
	b.Run("Kernel/"+name, func(b *testing.B) {
		var (
			left    = rightTransposeBenchmarkValues(rows * inner)
			output  = make([]float32, rows*cols)
			weights = make([][]float32, rotations)
			next    int
		)
		for index := range weights {
			weights[index] = rightTransposeBenchmarkValues(inner * cols)
		}
		b.ReportAllocs()
		b.SetBytes(int64(inner) * int64(cols) * 4)
		for b.Loop() {
			float32product.RightTranspose(left, weights[next], output, rows, inner, cols)
			next++
			if next == rotations {
				next = 0
			}
		}
	})
	b.Run("GemmInto/"+name, func(b *testing.B) {
		var (
			left, output *tensor.Tensor
			weights      = make([]*tensor.Tensor, rotations)
			config       = tensor.GemmConfig{TransposeB: true, Alpha: 1}
			next         int
			err          error
		)
		if left, err = tensor.FromFloat32([]int64{int64(rows), int64(inner)}, rightTransposeBenchmarkValues(rows*inner)); err != nil {
			b.Fatal(err)
		}
		for index := range weights {
			if weights[index], err = tensor.FromFloat32([]int64{int64(cols), int64(inner)}, rightTransposeBenchmarkValues(inner*cols)); err != nil {
				b.Fatal(err)
			}
		}
		if output, err = tensor.New(tensor.Float32, []int64{int64(rows), int64(cols)}); err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		b.SetBytes(int64(inner) * int64(cols) * 4)
		for b.Loop() {
			if err = tensor.GemmInto(left, weights[next], nil, output, config); err != nil {
				b.Fatal(err)
			}
			next++
			if next == rotations {
				next = 0
			}
		}
	})
}

// BenchmarkRightTransposeGroupsConcurrent measures six-row FFN calls with
// shared immutable operands and one disjoint destination per caller.
func BenchmarkRightTransposeGroupsConcurrent(b *testing.B) {
	benchmarkRightTransposeGroupsConcurrent(b, 6)
}

// BenchmarkRightTransposeRemainderConcurrent measures shared weights and
// disjoint destinations when each caller has a two-row remainder.
func BenchmarkRightTransposeRemainderConcurrent(b *testing.B) {
	benchmarkRightTransposeGroupsConcurrent(b, 5)
}

func benchmarkRightTransposeGroupsConcurrent(b *testing.B, rows int) {
	const inner, cols = 2048, 8192
	var (
		callers = runtime.GOMAXPROCS(0)
		shape   = []int64{int64(rows), int64(inner)}
		result  = []int64{int64(rows), int64(cols)}
	)
	b.Run("Kernel", func(b *testing.B) {
		var (
			left    = rightTransposeBenchmarkValues(rows * inner)
			right   = rightTransposeBenchmarkValues(cols * inner)
			outputs = make([]float32, callers*rows*cols)
			next    atomic.Int64
		)
		b.ReportAllocs()
		b.SetBytes(inner * cols * 4)
		b.ResetTimer()
		b.RunParallel(func(work *testing.PB) {
			var offset = int(next.Add(1)-1) * rows * cols
			var output = outputs[offset : offset+rows*cols]
			for work.Next() {
				float32product.RightTranspose(left, right, output, rows, inner, cols)
			}
		})
	})
	b.Run("GemmInto", func(b *testing.B) {
		var (
			left, right *tensor.Tensor
			outputs     = make([]*tensor.Tensor, callers)
			config      = tensor.GemmConfig{TransposeB: true, Alpha: 1}
			next        atomic.Int64
			err         error
		)
		if left, err = tensor.FromFloat32(shape, rightTransposeBenchmarkValues(rows*inner)); err != nil {
			b.Fatal(err)
		}
		if right, err = tensor.FromFloat32([]int64{cols, inner}, rightTransposeBenchmarkValues(cols*inner)); err != nil {
			b.Fatal(err)
		}
		for index := range outputs {
			if outputs[index], err = tensor.New(tensor.Float32, result); err != nil {
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
