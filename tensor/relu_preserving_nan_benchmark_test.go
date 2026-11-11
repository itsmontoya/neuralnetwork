package tensor_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func BenchmarkReLUPreservingNaNInto(b *testing.B) {
	for _, size := range []int{0, 1, 2, 3, 7, 14, 31, 32, 33, 42, 1023, 1024, 1025, 3136, 6272, 8193} {
		for _, pattern := range []string{"finite", "mixed", "allNaN"} {
			b.Run(fmt.Sprintf("%d/%s", size, pattern), func(operation *testing.B) {
				var (
					values        = make([]float32, size)
					shape         = []int64{int64(size)}
					input, output *tensor.Tensor
					err           error
				)

				for index := range values {
					values[index] = float32(index%17) - 8
					if pattern == "allNaN" || pattern == "mixed" && index%7 == 0 {
						values[index] = math.Float32frombits(0xff801234)
					}
				}
				if size == 1 {
					shape = nil
				}
				if input, err = tensor.FromFloat32(shape, values); err != nil {
					operation.Fatal(err)
				}
				if output, err = tensor.New(tensor.Float32, shape); err != nil {
					operation.Fatal(err)
				}
				operation.ReportAllocs()
				operation.ResetTimer()
				for operation.Loop() {
					if _, err = tensor.ReLUPreservingNaNInto(context.Background(), input, output); err != nil {
						operation.Fatal(err)
					}
				}
			})
		}
	}
}
