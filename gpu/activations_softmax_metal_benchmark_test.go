//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"time"
)

// Benchmark_ActivationsMetal measures isolated ReLU, Sigmoid, and Tanh latency
// on Apple Silicon Metal across [1, 4096], [32, 4096], and [1, 128256].
func Benchmark_ActivationsMetal(b *testing.B) {
	type shape struct {
		name string
		rows int64
		cols int64
	}
	var shapes = []shape{
		{name: "1x4096", rows: 1, cols: 4096},
		{name: "32x4096", rows: 32, cols: 4096},
		{name: "1x128256", rows: 1, cols: 128256},
	}
	var ops = []struct {
		name   string
		encode func(*Scope, *Tensor, *Tensor) error
	}{
		{name: "ReLU", encode: (*Scope).ReLU},
		{name: "Sigmoid", encode: (*Scope).Sigmoid},
		{name: "Tanh", encode: (*Scope).Tanh},
	}
	for _, op := range ops {
		for _, tc := range shapes {
			b.Run(fmt.Sprintf("%s/%s", op.name, tc.name), func(b *testing.B) {
				benchmarkUnaryOpMetal(b, tc.rows, tc.cols, op.encode)
			})
		}
	}
}

// Benchmark_SoftmaxMetal measures isolated contiguous last-axis Softmax across
// [1, 4096], [32, 4096], and [1, 128256], plus strided SoftmaxAxis on [32, 64, 64].
func Benchmark_SoftmaxMetal(b *testing.B) {
	type shape struct {
		name string
		rows int64
		cols int64
	}
	var shapes = []shape{
		{name: "1x4096", rows: 1, cols: 4096},
		{name: "32x4096", rows: 32, cols: 4096},
		{name: "1x128256", rows: 1, cols: 128256},
	}
	for _, tc := range shapes {
		b.Run(tc.name, func(b *testing.B) {
			benchmarkUnaryOpMetal(b, tc.rows, tc.cols, (*Scope).Softmax)
		})
	}
	b.Run("Strided_32x64x64_axis1", func(b *testing.B) {
		var (
			device = requireMetalDevice(b)
			ctx    = context.Background()
			input  = requireTensor(b, device, 32, 64, 64)
			output = requireTensor(b, device, 32, 64, 64)
			data   = attentionValues(32*64*64, 141, 1)
		)
		requireError(b, input.Upload(data), nil)
		runOnce := func() (elapsed time.Duration) {
			var (
				start      = time.Now()
				scope, err = device.NewScope()
			)
			requireError(b, err, nil)
			defer scope.Close()
			requireError(b, scope.SoftmaxAxis(input, output, 1), nil)
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
}

// Benchmark_ResidentMLPMetal measures a full single-scope resident
// MatMulRightTranspose -> ReLU -> MatMulRightTranspose -> Softmax MLP
// (hidden=1024, intermediate=2048, classes=1000) with zero intermediate host transfers.
func Benchmark_ResidentMLPMetal(b *testing.B) {
	for _, rows := range []int64{1, 32} {
		b.Run(fmt.Sprintf("M%d", rows), func(b *testing.B) {
			var (
				device   = requireMetalDevice(b)
				ctx      = context.Background()
				native   = device.state.backend.(metalBackend).runtime
				baseline = native.ResourceSnapshot()
				inDim    = int64(1024)
				hidden   = int64(2048)
				classes  = int64(1000)
				random   = rand.New(rand.NewPCG(0x14be11, uint64(rows)))
				xData    = make([]float32, rows*inDim)
				w1Data   = make([]float32, hidden*inDim)
				w2Data   = make([]float32, classes*hidden)
				outHost  = make([]float32, rows*classes)
			)
			for i := range xData {
				xData[i] = 2*random.Float32() - 1
			}
			for i := range w1Data {
				w1Data[i] = (2*random.Float32() - 1) / float32(math.Sqrt(float64(inDim)))
			}
			for i := range w2Data {
				w2Data[i] = (2*random.Float32() - 1) / float32(math.Sqrt(float64(hidden)))
			}
			var (
				x      = requireTensor(b, device, rows, inDim)
				w1     = requireTensor(b, device, hidden, inDim)
				w2     = requireTensor(b, device, classes, hidden)
				h1     = requireTensor(b, device, rows, hidden)
				h1Relu = requireTensor(b, device, rows, hidden)
				logits = requireTensor(b, device, rows, classes)
				probs  = requireTensor(b, device, rows, classes)
			)
			requireError(b, w1.Upload(w1Data), nil)
			requireError(b, w2.Upload(w2Data), nil)

			runOnce := func() (resident time.Duration) {
				requireError(b, x.Upload(xData), nil)
				var (
					start      = time.Now()
					scope, err = device.NewScope()
				)
				requireError(b, err, nil)
				defer scope.Close()
				requireError(b, scope.MatMulRightTranspose(x, w1, h1), nil)
				requireError(b, scope.ReLU(h1, h1Relu), nil)
				requireError(b, scope.MatMulRightTranspose(h1Relu, w2, logits), nil)
				requireError(b, scope.Softmax(logits, probs), nil)
				requireError(b, scope.Submit(ctx), nil)
				requireError(b, scope.Wait(ctx), nil)
				requireError(b, scope.Close(), nil)
				resident = time.Since(start)
				requireError(b, probs.DownloadInto(outHost), nil)
				return resident
			}

			for range 3 {
				runOnce()
			}
			var totalResident time.Duration
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				totalResident += runOnce()
			}
			b.StopTimer()
			b.ReportMetric(float64(totalResident.Nanoseconds())/float64(b.N)/1e6, "resident-ms/op")
			var (
				final = native.ResourceSnapshot()
				steps = uint64(b.N + 3)
			)
			if final.SubmittedCommands != baseline.SubmittedCommands+steps ||
				final.CompletedCommands != baseline.CompletedCommands+steps {
				b.Fatalf("expected 1 submit per MLP op: before=%+v after=%+v steps=%d", baseline, final, steps)
			}
		})
	}
}

func benchmarkUnaryOpMetal(b *testing.B, rows, cols int64, encode func(*Scope, *Tensor, *Tensor) error) {
	var (
		device = requireMetalDevice(b)
		ctx    = context.Background()
		input  = requireTensor(b, device, rows, cols)
		output = requireTensor(b, device, rows, cols)
		data   = attentionValues(rows*cols, 140, 1)
	)
	requireError(b, input.Upload(data), nil)

	runOnce := func() (elapsed time.Duration) {
		var (
			start      = time.Now()
			scope, err = device.NewScope()
		)
		requireError(b, err, nil)
		defer scope.Close()
		requireError(b, encode(scope, input, output), nil)
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
