package tensor_test

import (
	"fmt"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

// BenchmarkPass4TensorArithmetic measures preallocated tensor arithmetic at
// Llama 3.2 1B decode and prefill sizes, with sigmoid as a comparison.
func BenchmarkPass4TensorArithmetic(b *testing.B) {
	var operations = []struct {
		name string
		dims []int64
		call func(left, right, destination *tensor.Tensor) error
	}{
		{name: "Add", dims: []int64{2048}, call: tensor.AddInto},
		{name: "AddRotaryQuery", dims: []int64{32, 64}, call: tensor.AddInto},
		{name: "AddRotaryKey", dims: []int64{8, 64}, call: tensor.AddInto},
		{name: "Subtract", dims: []int64{2048}, call: tensor.SubInto},
		{name: "Multiply", dims: []int64{2048}, call: tensor.MulInto},
		{name: "Multiply", dims: []int64{8192}, call: tensor.MulInto},
		{name: "Sigmoid", dims: []int64{8192}, call: func(left, _, destination *tensor.Tensor) (err error) {
			return tensor.SigmoidInto(left, destination)
		}},
	}
	for _, operation := range operations {
		for _, rows := range []int64{1, 6, 29} {
			var (
				shape = []int64{rows}
				name  = fmt.Sprintf("%s/%d", operation.name, rows)
			)
			for _, dimension := range operation.dims {
				shape = append(shape, dimension)
				name += fmt.Sprintf("x%d", dimension)
			}
			b.Run(name, func(b *testing.B) {
				var (
					left, right, destination *tensor.Tensor
					err                      error
				)
				left = pass4BenchmarkInput(b, shape, 3)
				right = pass4BenchmarkInput(b, shape, 5)
				if destination, err = tensor.New(tensor.Float32, shape); err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				for b.Loop() {
					if err = operation.call(left, right, destination); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func pass4BenchmarkInput(b *testing.B, shape []int64, multiplier int) (input *tensor.Tensor) {
	var (
		length int64 = 1
		values []float32
		err    error
	)
	b.Helper()
	for _, dimension := range shape {
		length *= dimension
	}
	values = make([]float32, length)
	for index := range values {
		values[index] = float32((index*multiplier)%127-63) / 8
	}
	if input, err = tensor.FromFloat32(shape, values); err != nil {
		b.Fatal(err)
	}
	return input
}
