//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// Benchmark_RowBiasMetal compares native AddBroadcast ([M, N] + [N], 1 dispatch,
// 0 scratch bytes) against the ONNX-01 GatherRows + Add workaround (2 dispatches
// and an extra [M, N] FP32 scratch buffer) on Apple Silicon Metal.
func Benchmark_RowBiasMetal(b *testing.B) {
	type shape struct {
		name       string
		rows       int64
		cols       int64
		dispatches int
	}
	var shapes = []shape{
		{name: "1x4096", rows: 1, cols: 4096, dispatches: 1},
		{name: "32x4096", rows: 32, cols: 4096, dispatches: 1},
		{name: "32x4096_32disp", rows: 32, cols: 4096, dispatches: 32},
	}

	for _, tc := range shapes {
		b.Run(fmt.Sprintf("AddBroadcast/%s", tc.name), func(b *testing.B) {
			var (
				device = requireMetalDevice(b)
				ctx    = context.Background()
				input  = requireTensor(b, device, tc.rows, tc.cols)
				bias   = requireTensor(b, device, tc.cols)
				output = requireTensor(b, device, tc.rows, tc.cols)
			)
			requireError(b, input.Upload(attentionValues(tc.rows*tc.cols, 1501, 1)), nil)
			requireError(b, bias.Upload(attentionValues(tc.cols, 1502, 1)), nil)
			runOnce := func() (elapsed time.Duration) {
				var (
					start      = time.Now()
					scope, err = device.NewScope()
				)
				requireError(b, err, nil)
				defer scope.Close()
				for range tc.dispatches {
					requireError(b, scope.AddBroadcast(input, bias, output), nil)
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
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				total += runOnce()
			}
			b.StopTimer()
			b.ReportMetric(float64(total.Nanoseconds())/float64(b.N)/1e6, "ms/op")
			b.ReportMetric(0, "scratch-B")
		})

		b.Run(fmt.Sprintf("GatherRowsPlusAdd/%s", tc.name), func(b *testing.B) {
			var (
				device  = requireMetalDevice(b)
				ctx     = context.Background()
				input   = requireTensor(b, device, tc.rows, tc.cols)
				bias2D  = requireTensor(b, device, 1, tc.cols)
				scratch = requireTensor(b, device, tc.rows, tc.cols)
				output  = requireTensor(b, device, tc.rows, tc.cols)
				rowIdxs = make([]int64, tc.rows)
			)
			requireError(b, input.Upload(attentionValues(tc.rows*tc.cols, 1501, 1)), nil)
			requireError(b, bias2D.Upload(attentionValues(tc.cols, 1502, 1)), nil)
			runOnce := func() (elapsed time.Duration) {
				var (
					start      = time.Now()
					scope, err = device.NewScope()
				)
				requireError(b, err, nil)
				defer scope.Close()
				for range tc.dispatches {
					requireError(b, scope.GatherRows(bias2D, rowIdxs, scratch), nil)
					requireError(b, scope.Add(input, scratch, output), nil)
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
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				total += runOnce()
			}
			b.StopTimer()
			b.ReportMetric(float64(total.Nanoseconds())/float64(b.N)/1e6, "ms/op")
			b.ReportMetric(float64(scratch.Bytes()), "scratch-B")
		})
	}
}

// Benchmark_ArithmeticAndBroadcastMetal measures isolated Sub, Div, Scale,
// scalar broadcast, column broadcast, and 4D multidirectional broadcast on Metal.
func Benchmark_ArithmeticAndBroadcastMetal(b *testing.B) {
	b.Run("Sub/32x4096", func(b *testing.B) {
		benchmarkBinaryOpMetal(b, []int64{32, 4096}, []int64{32, 4096}, []int64{32, 4096}, (*Scope).Sub)
	})
	b.Run("Div/32x4096", func(b *testing.B) {
		benchmarkBinaryOpMetal(b, []int64{32, 4096}, []int64{32, 4096}, []int64{32, 4096}, (*Scope).Div)
	})
	b.Run("Scale/32x4096", func(b *testing.B) {
		var (
			device = requireMetalDevice(b)
			ctx    = context.Background()
			input  = requireTensor(b, device, 32, 4096)
			output = requireTensor(b, device, 32, 4096)
		)
		requireError(b, input.Upload(attentionValues(32*4096, 1510, 1)), nil)
		runOnce := func() (elapsed time.Duration) {
			var (
				start      = time.Now()
				scope, err = device.NewScope()
			)
			requireError(b, err, nil)
			defer scope.Close()
			requireError(b, scope.Scale(input, output, 0.125), nil)
			requireError(b, scope.Submit(ctx), nil)
			requireError(b, scope.Wait(ctx), nil)
			requireError(b, scope.Close(), nil)
			return time.Since(start)
		}
		for range 3 {
			runOnce()
		}
		var total time.Duration
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			total += runOnce()
		}
		b.StopTimer()
		b.ReportMetric(float64(total.Nanoseconds())/float64(b.N)/1e6, "ms/op")
	})
	b.Run("AddBroadcast_Scalar/32x4096", func(b *testing.B) {
		benchmarkBinaryOpMetal(b, []int64{32, 4096}, []int64{1}, []int64{32, 4096}, (*Scope).AddBroadcast)
	})
	b.Run("AddBroadcast_Column/32x4096", func(b *testing.B) {
		benchmarkBinaryOpMetal(b, []int64{32, 4096}, []int64{32, 1}, []int64{32, 4096}, (*Scope).AddBroadcast)
	})
	b.Run("AddBroadcast_4D/2x8x32x64", func(b *testing.B) {
		benchmarkBinaryOpMetal(b, []int64{2, 8, 32, 64}, []int64{1, 8, 1, 64}, []int64{2, 8, 32, 64}, (*Scope).AddBroadcast)
	})
}

func benchmarkBinaryOpMetal(
	b *testing.B,
	leftShape, rightShape, outShape []int64,
	encode func(*Scope, *Tensor, *Tensor, *Tensor) error,
) {
	b.Helper()
	var (
		device    = requireMetalDevice(b)
		ctx       = context.Background()
		left      = requireTensor(b, device, leftShape...)
		right     = requireTensor(b, device, rightShape...)
		output    = requireTensor(b, device, outShape...)
		rightVals = attentionValues(int64(elementCount(rightShape)), 1512, 1)
	)
	for i := range rightVals {
		if rightVals[i] == 0 {
			rightVals[i] = 1
		}
	}
	requireError(b, left.Upload(attentionValues(int64(elementCount(leftShape)), 1511, 1)), nil)
	requireError(b, right.Upload(rightVals), nil)
	runOnce := func() (elapsed time.Duration) {
		var (
			start      = time.Now()
			scope, err = device.NewScope()
		)
		requireError(b, err, nil)
		defer scope.Close()
		requireError(b, encode(scope, left, right, output), nil)
		requireError(b, scope.Submit(ctx), nil)
		requireError(b, scope.Wait(ctx), nil)
		requireError(b, scope.Close(), nil)
		return time.Since(start)
	}
	for range 3 {
		runOnce()
	}
	var total time.Duration
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		total += runOnce()
	}
	b.StopTimer()
	b.ReportMetric(float64(total.Nanoseconds())/float64(b.N)/1e6, "ms/op")
}
