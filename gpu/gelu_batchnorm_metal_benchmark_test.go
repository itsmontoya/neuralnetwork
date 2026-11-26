//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

// Benchmark_GELUMetal measures isolated Scope.GELU and Scope.GELUTanh latency
// on Apple Silicon Metal across [32, 4096] and [2, 64, 3072].
func Benchmark_GELUMetal(b *testing.B) {
	type shape struct {
		name string
		dims []int64
	}
	var shapes = []shape{
		{name: "32x4096", dims: []int64{32, 4096}},
		{name: "2x64x3072", dims: []int64{2, 64, 3072}},
	}
	var ops = []struct {
		name   string
		encode func(*Scope, *Tensor, *Tensor) error
	}{
		{name: "GELU", encode: (*Scope).GELU},
		{name: "GELUTanh", encode: (*Scope).GELUTanh},
	}
	for _, op := range ops {
		for _, tc := range shapes {
			b.Run(fmt.Sprintf("%s/%s", op.name, tc.name), func(b *testing.B) {
				var (
					dev    = requireMetalDevice(b)
					ctx    = context.Background()
					input  = requireTensor(b, dev, tc.dims...)
					output = requireTensor(b, dev, tc.dims...)
					count  = int64(input.state.storage.count)
					data   = attentionValues(count, 261, 1)
				)
				requireError(b, input.Upload(data), nil)
				runOnce := func() (elapsed time.Duration) {
					var (
						start      = time.Now()
						scope, err = dev.NewScope()
					)
					requireError(b, err, nil)
					defer scope.Close()
					requireError(b, op.encode(scope, input, output), nil)
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
				var ms = float64(total.Nanoseconds()) / float64(b.N) / 1e6
				var gbPerSec = (float64(count*4*2) / 1e9) / (ms / 1e3)
				b.ReportMetric(ms, "ms/op")
				b.ReportMetric(gbPerSec, "GB/s")
			})
		}
	}
}

// Benchmark_BatchNormMetal measures isolated Scope.BatchNorm latency on Apple
// Silicon Metal and paired CPU tensor.BatchNormalizationInto across
// [32, 64, 56, 56] and [1, 256, 14, 14].
func Benchmark_BatchNormMetal(b *testing.B) {
	type shape struct {
		name string
		dims []int64
	}
	var shapes = []shape{
		{name: "32x64x56x56", dims: []int64{32, 64, 56, 56}},
		{name: "1x256x14x14", dims: []int64{1, 256, 14, 14}},
	}
	for _, tc := range shapes {
		var (
			channels = tc.dims[1]
			count    = tc.dims[0] * tc.dims[1] * tc.dims[2] * tc.dims[3]
			xData    = attentionValues(count, 262, 1)
			sData    = attentionValues(channels, 263, 1)
			bData    = attentionValues(channels, 264, 1)
			mData    = attentionValues(channels, 265, 1)
			vData    = make([]float32, channels)
		)
		for i := range vData {
			vData[i] = 0.5 + float32(i%7)*0.25
		}

		b.Run("Metal/"+tc.name, func(b *testing.B) {
			var (
				dev      = requireMetalDevice(b)
				ctx      = context.Background()
				input    = requireTensor(b, dev, tc.dims...)
				output   = requireTensor(b, dev, tc.dims...)
				scale    = requireTensor(b, dev, channels)
				bias     = requireTensor(b, dev, channels)
				mean     = requireTensor(b, dev, channels)
				variance = requireTensor(b, dev, channels)
			)
			requireError(b, input.Upload(xData), nil)
			requireError(b, scale.Upload(sData), nil)
			requireError(b, bias.Upload(bData), nil)
			requireError(b, mean.Upload(mData), nil)
			requireError(b, variance.Upload(vData), nil)

			runOnce := func() (elapsed time.Duration) {
				var (
					start      = time.Now()
					scope, err = dev.NewScope()
				)
				requireError(b, err, nil)
				defer scope.Close()
				requireError(b, scope.BatchNorm(input, scale, bias, mean, variance, output, 1e-5), nil)
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
			var ms = float64(total.Nanoseconds()) / float64(b.N) / 1e6
			var gbPerSec = (float64(count*4*2) / 1e9) / (ms / 1e3)
			b.ReportMetric(ms, "ms/op")
			b.ReportMetric(gbPerSec, "GB/s")
		})

		b.Run("CPU/"+tc.name, func(b *testing.B) {
			var (
				cpuIn, e1  = tensor.FromFloat32(tc.dims, xData)
				cpuSc, e2  = tensor.FromFloat32([]int64{channels}, sData)
				cpuBi, e3  = tensor.FromFloat32([]int64{channels}, bData)
				cpuMn, e4  = tensor.FromFloat32([]int64{channels}, mData)
				cpuVr, e5  = tensor.FromFloat32([]int64{channels}, vData)
				cpuOut, e6 = tensor.FromFloat32(tc.dims, make([]float32, count))
			)
			for _, e := range []error{e1, e2, e3, e4, e5, e6} {
				requireError(b, e, nil)
			}
			var total time.Duration
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				var start = time.Now()
				requireError(b, tensor.BatchNormalizationInto(cpuIn, cpuSc, cpuBi, cpuMn, cpuVr, cpuOut, 1e-5), nil)
				total += time.Since(start)
			}
			b.StopTimer()
			var ms = float64(total.Nanoseconds()) / float64(b.N) / 1e6
			var gbPerSec = (float64(count*4*2) / 1e9) / (ms / 1e3)
			b.ReportMetric(ms, "ms/op")
			b.ReportMetric(gbPerSec, "GB/s")
		})
	}
}
