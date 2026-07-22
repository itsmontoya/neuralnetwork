//go:build darwin && cgo && metal && !purego

package matrix

import "testing"

var metalResidencyBenchmarkResult *Matrix
var metalResidencyBenchmarkValues []float32

func Benchmark_MetalMatrixResidency(b *testing.B) {
	var (
		left   *Matrix
		right  *Matrix
		result *Matrix
		values []float32
		index  int
		err    error
	)

	if !metalAvailable() {
		b.Skipf("Metal device unavailable: %s", metalLastError())
	}
	left = metalTestMatrix(b, 128, 256, 0.25)
	right = metalTestMatrix(b, 256, 128, -0.5)
	if result, err = New(128, 128); err != nil {
		b.Fatalf("New result returned error: %v", err)
	}
	values = make([]float32, 128*128)
	if err = left.MatMulInto(right, result); err != nil {
		b.Fatalf("warm-up MatMulInto returned error: %v", err)
	}

	b.Run("WarmedUnobserved", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for index = 0; index < b.N; index++ {
			if err = left.MatMulInto(right, result); err != nil {
				b.Fatalf("MatMulInto returned error: %v", err)
			}
		}
	})

	b.Run("WarmedObserved", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for index = 0; index < b.N; index++ {
			if err = left.MatMulInto(right, result); err != nil {
				b.Fatalf("MatMulInto returned error: %v", err)
			}
			if err = result.ValuesInto(values); err != nil {
				b.Fatalf("ValuesInto returned error: %v", err)
			}
		}
	})

	metalResidencyBenchmarkResult = result
	metalResidencyBenchmarkValues = values
}
