//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// Benchmark_RMSNormMetal measures isolated single-dispatch and 33-dispatch
// batched RMSNorm latency across Llama 3.2 1B decode and prefill shapes.
func Benchmark_RMSNormMetal(b *testing.B) {
	type shape struct {
		name       string
		rows       int64
		cols       int64
		dispatches int
	}

	var shapes = []shape{
		{name: "M1_C2048_x1", rows: 1, cols: 2048, dispatches: 1},
		{name: "M1_C2048_x33", rows: 1, cols: 2048, dispatches: 33},
		{name: "M6_C2048_x33", rows: 6, cols: 2048, dispatches: 33},
		{name: "M27_C2048_x33", rows: 27, cols: 2048, dispatches: 33},
		{name: "M28_C2048_x33", rows: 28, cols: 2048, dispatches: 33},
		{name: "M29_C2048_x33", rows: 29, cols: 2048, dispatches: 33},
	}

	for _, tc := range shapes {
		for _, dtype := range []DType{Float32, Float16, BFloat16} {
			b.Run(fmt.Sprintf("%s/%s", tc.name, dtype), func(b *testing.B) {
				benchmarkRMSNormMetal(b, tc.rows, tc.cols, tc.dispatches, dtype)
			})
		}
	}
}

func benchmarkRMSNormMetal(b *testing.B, rows, cols int64, dispatches int, weightDType DType) {
	var (
		device     = requireMetalDevice(b)
		ctx        = context.Background()
		input      = requireTensor(b, device, rows, cols)
		output     = requireTensor(b, device, rows, cols)
		weight     = requireTensorWithDType(b, device, weightDType, cols)
		inputData  = attentionValues(rows*cols, 101, 1)
		weightData = attentionValues(cols, 102, 1)
	)
	requireError(b, input.Upload(inputData), nil)
	requireError(b, weight.UploadFloat32As(weightData), nil)

	runOnce := func() (elapsed time.Duration) {
		var (
			start      = time.Now()
			scope, err = device.NewScope()
		)
		requireError(b, err, nil)
		defer scope.Close()
		for range dispatches {
			requireError(b, scope.RMSNorm(input, weight, output, attentionEpsilon), nil)
		}
		requireError(b, scope.Submit(ctx), nil)
		requireError(b, scope.Wait(ctx), nil)
		requireError(b, scope.Close(), nil)
		return time.Since(start)
	}

	// Warmup.
	for range 3 {
		runOnce()
	}

	var total time.Duration
	b.ResetTimer()
	for b.Loop() {
		total += runOnce()
	}
	b.StopTimer()
	b.ReportMetric(float64(total.Nanoseconds())/float64(b.N)/1e6, "ms/op")
	b.ReportMetric(float64(dispatches), "dispatches/op")
}
