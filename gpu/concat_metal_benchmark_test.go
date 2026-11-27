//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"testing"
)

// Benchmark_ConcatMetal measures Scope.Concat on Apple Silicon Metal for
// realistic CNN channel concatenation (outer > 1 strided compute path) and
// batch / single-batch concatenation (outer == 1 blit fast path) across
// Float32, Float16, and BFloat16.
func Benchmark_ConcatMetal(b *testing.B) {
	type testcase struct {
		name   string
		shapes [][]int64
		axis   int
	}
	var cases = []testcase{
		{
			name:   "CNN_Channel_N8_64_128_64_56x56_axis1",
			shapes: [][]int64{{8, 64, 56, 56}, {8, 128, 56, 56}, {8, 64, 56, 56}},
			axis:   1,
		},
		{
			name:   "Batch_N4_4_256_56x56_axis0_blit",
			shapes: [][]int64{{4, 256, 56, 56}, {4, 256, 56, 56}},
			axis:   0,
		},
		{
			name:   "CNN_Channel_N1_64_128_64_56x56_axis1_blit",
			shapes: [][]int64{{1, 64, 56, 56}, {1, 128, 56, 56}, {1, 64, 56, 56}},
			axis:   1,
		},
		{
			name: "CNN_Channel_N8_8x32_56x56_axis1",
			shapes: [][]int64{
				{8, 32, 56, 56}, {8, 32, 56, 56}, {8, 32, 56, 56}, {8, 32, 56, 56},
				{8, 32, 56, 56}, {8, 32, 56, 56}, {8, 32, 56, 56}, {8, 32, 56, 56},
			},
			axis: 1,
		},
	}

	var dtypes = []DType{Float32, Float16, BFloat16}

	for _, tc := range cases {
		var dstShape = concatDestShape(tc.shapes, tc.axis)
		var totalElems = shapeProduct(dstShape)
		for _, dt := range dtypes {
			b.Run(fmt.Sprintf("%s/%s", dt, tc.name), func(b *testing.B) {
				var (
					dev    = requireMetalDevice(b)
					ctx    = context.Background()
					inputs = make([]*Tensor, len(tc.shapes))
					dst    = requireTensorWithDType(b, dev, dt, dstShape...)
				)
				for i, s := range tc.shapes {
					var (
						n      = shapeProduct(s)
						tensor = requireTensorWithDType(b, dev, dt, s...)
					)
					if dt == Float32 {
						requireError(b, tensor.Upload(make([]float32, n)), nil)
					} else {
						requireError(b, tensor.UploadBits(make([]uint16, n)), nil)
					}
					inputs[i] = tensor
				}

				// Warmup (lazy shader compile for outer > 1).
				var warmScope, err = dev.NewScope()
				requireError(b, err, nil)
				requireError(b, warmScope.Concat(inputs, dst, tc.axis), nil)
				requireError(b, warmScope.Submit(ctx), nil)
				requireError(b, warmScope.Wait(ctx), nil)
				requireError(b, warmScope.Close(), nil)

				// Effective memory traffic: read all inputs + write destination = 2 * dstBytes.
				var trafficBytes = 2.0 * float64(totalElems) * float64(dt.Bytes())
				b.ResetTimer()
				for range b.N {
					var scope, scopeErr = dev.NewScope()
					if scopeErr != nil {
						b.Fatal(scopeErr)
					}
					if scopeErr = scope.Concat(inputs, dst, tc.axis); scopeErr != nil {
						b.Fatal(scopeErr)
					}
					if scopeErr = scope.Submit(ctx); scopeErr != nil {
						b.Fatal(scopeErr)
					}
					if scopeErr = scope.Wait(ctx); scopeErr != nil {
						b.Fatal(scopeErr)
					}
					if scopeErr = scope.Close(); scopeErr != nil {
						b.Fatal(scopeErr)
					}
				}
				b.StopTimer()
				var elapsed = b.Elapsed().Seconds()
				if elapsed > 0 {
					b.ReportMetric((trafficBytes*float64(b.N)/elapsed)/1e9, "GB/s")
				}
			})
		}
	}
}
