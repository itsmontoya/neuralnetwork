//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// Benchmark_LayerNormMetal measures isolated single-dispatch and 33-dispatch
// batched LayerNorm latency across BERT/ViT (768) and LLM (2048) widths.
func Benchmark_LayerNormMetal(b *testing.B) {
	type shape struct {
		name       string
		rows       int64
		cols       int64
		dispatches int
	}

	var shapes = []shape{
		{name: "M1_C768_x1", rows: 1, cols: 768, dispatches: 1},
		{name: "M32_C768_x1", rows: 32, cols: 768, dispatches: 1},
		{name: "M1_C768_x33", rows: 1, cols: 768, dispatches: 33},
		{name: "M32_C768_x33", rows: 32, cols: 768, dispatches: 33},
		{name: "M1_C2048_x1", rows: 1, cols: 2048, dispatches: 1},
		{name: "M32_C2048_x1", rows: 32, cols: 2048, dispatches: 1},
		{name: "M1_C2048_x33", rows: 1, cols: 2048, dispatches: 33},
		{name: "M32_C2048_x33", rows: 32, cols: 2048, dispatches: 33},
	}

	for _, tc := range shapes {
		for _, dtype := range []DType{Float32, Float16, BFloat16} {
			b.Run(fmt.Sprintf("%s/%s", tc.name, dtype), func(b *testing.B) {
				benchmarkLayerNormMetal(b, tc.rows, tc.cols, tc.dispatches, dtype, true)
			})
		}
		if tc.dispatches == 33 {
			b.Run(fmt.Sprintf("%s/Float32_NoBias", tc.name), func(b *testing.B) {
				benchmarkLayerNormMetal(b, tc.rows, tc.cols, tc.dispatches, Float32, false)
			})
		}
	}
}

func benchmarkLayerNormMetal(b *testing.B, rows, cols int64, dispatches int, paramDType DType, withBias bool) {
	var (
		dev        = requireMetalDevice(b)
		ctx        = context.Background()
		input      = requireTensor(b, dev, rows, cols)
		output     = requireTensor(b, dev, rows, cols)
		weight     = requireTensorWithDType(b, dev, paramDType, cols)
		bias       *Tensor
		inputData  = attentionValues(rows*cols, 201, 1)
		weightData = attentionValues(cols, 202, 1)
	)
	requireError(b, input.Upload(inputData), nil)
	requireError(b, weight.UploadFloat32As(weightData), nil)
	if withBias {
		bias = requireTensorWithDType(b, dev, paramDType, cols)
		requireError(b, bias.UploadFloat32As(attentionValues(cols, 203, 1)), nil)
	}

	runOnce := func() (elapsed time.Duration) {
		var (
			start      = time.Now()
			scope, err = dev.NewScope()
		)
		requireError(b, err, nil)
		defer scope.Close()
		for range dispatches {
			requireError(b, scope.LayerNorm(input, weight, bias, output, attentionEpsilon), nil)
		}
		requireError(b, scope.Submit(ctx), nil)
		requireError(b, scope.Wait(ctx), nil)
		requireError(b, scope.Close(), nil)
		return time.Since(start)
	}

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
