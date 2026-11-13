package tensor

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// BenchmarkMatrixAdapterArithmetic includes public validation and matrix dispatch.
func BenchmarkMatrixAdapterArithmetic(b *testing.B) {
	var operations = []struct {
		name string
		run  func(*Tensor, *Tensor, *Tensor) error
	}{
		{name: "Add", run: AddInto},
		{name: "Sub", run: SubInto},
		{name: "Mul", run: MulInto},
	}
	for _, operation := range operations {
		b.Run(operation.name, func(b *testing.B) {
			for _, length := range []int{0, 1, 10, 15, 16, 17, 4096, 65536} {
				b.Run(fmt.Sprint(length), func(b *testing.B) {
					var (
						left, right, output *Tensor
						err                 error
					)
					left = adapterBenchmarkTensor(b, []int64{1, int64(length)}, 3)
					right = adapterBenchmarkTensor(b, left.shape, 7)
					output = adapterBenchmarkTensor(b, left.shape, 0)
					b.ReportAllocs()
					for b.Loop() {
						if err = operation.run(left, right, output); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		})
	}
}

// BenchmarkMatrixAdapterScopes separates the adapter from allocation/copy costs.
func BenchmarkMatrixAdapterScopes(b *testing.B) {
	var (
		left, right, output             *Tensor
		leftMatrix, rightMatrix, result *matrix.Matrix
		values                          []float32
		err                             error
	)
	left = adapterBenchmarkTensor(b, []int64{1, 10}, 3)
	right = adapterBenchmarkTensor(b, left.shape, 7)
	output = adapterBenchmarkTensor(b, left.shape, 0)
	if leftMatrix, err = matrix.FromSlice(1, 10, left.floats); err != nil {
		b.Fatal(err)
	}
	if rightMatrix, err = matrix.FromSlice(1, 10, right.floats); err != nil {
		b.Fatal(err)
	}
	if result, err = matrix.New(1, 10); err != nil {
		b.Fatal(err)
	}
	b.Run("Adapter", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if err = matrixBinaryInto(left, right, output, (*matrix.Matrix).AddInto); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("FromSlice", func(b *testing.B) {
		var created *matrix.Matrix
		b.ReportAllocs()
		for b.Loop() {
			if created, err = matrix.FromSlice(1, 10, left.floats); err != nil {
				b.Fatal(err)
			}
		}
		runtime.KeepAlive(created)
	})
	b.Run("New", func(b *testing.B) {
		var created *matrix.Matrix
		b.ReportAllocs()
		for b.Loop() {
			if created, err = matrix.New(1, 10); err != nil {
				b.Fatal(err)
			}
		}
		runtime.KeepAlive(created)
	})
	b.Run("MatrixAdd", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if err = leftMatrix.AddInto(rightMatrix, result); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("Values", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if values, err = result.Values(); err != nil {
				b.Fatal(err)
			}
		}
		runtime.KeepAlive(values)
	})
	b.Run("ValuesInto", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if err = result.ValuesInto(output.floats); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("Copy", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			copy(output.floats, left.floats)
		}
	})
}

// BenchmarkMatMulAdapter includes planning and staging in the complete scope.
// Kernel excludes both and is only a diagnostic, not a public-call replacement.
func BenchmarkMatMulAdapter(b *testing.B) {
	var tests = []struct {
		name        string
		left, right []int64
	}{
		{name: "MNIST", left: []int64{1, 256}, right: []int64{256, 10}},
		{name: "Dot", left: []int64{256}, right: []int64{256}},
		{name: "Batch", left: []int64{2, 1, 4, 16}, right: []int64{3, 16, 10}},
	}
	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			var (
				left, right, output *Tensor
				plan                *matMulPlan
				leftShape           []int64
				rightShape          []int64
				err                 error
			)
			left = adapterBenchmarkTensor(b, tt.left, 3)
			right = adapterBenchmarkTensor(b, tt.right, 7)
			if plan, err = newMatMulPlan(tt.left, tt.right); err != nil {
				b.Fatal(err)
			}
			output = adapterBenchmarkTensor(b, plan.shape, 0)
			leftShape = []int64{plan.rows, plan.inner}
			rightShape = []int64{plan.inner, plan.cols}
			b.Run("Complete", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if err = MatMulInto(left, right, output); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("Plan", func(b *testing.B) {
				var created *matMulPlan
				b.ReportAllocs()
				for b.Loop() {
					if created, err = newMatMulPlan(tt.left, tt.right); err != nil {
						b.Fatal(err)
					}
				}
				runtime.KeepAlive(created)
			})
			b.Run("Staging", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if err = matMulBatchesInto(left, right, output, plan); err != nil {
						b.Fatal(err)
					}
				}
			})
			if tt.name == "Batch" {
				return
			}
			b.Run("Kernel", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if err = matrixProductInto(left.floats, right.floats, leftShape, rightShape, false, false, output.floats); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}

func adapterBenchmarkTensor(b *testing.B, shape []int64, offset int) (out *Tensor) {
	var err error
	b.Helper()
	if out, err = New(Float32, shape); err != nil {
		b.Fatal(err)
	}
	for index := range out.floats {
		out.floats[index] = float32((index+offset)%29-14) * 0.125
	}
	return out
}
