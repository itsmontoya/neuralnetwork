//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

// Benchmark_Pool2DMetal measures isolated Scope.MaxPool2D and Scope.AvgPool2D
// latency on Apple Silicon Metal against paired CPU tensor.MaxPool2DInto and
// tensor.AveragePool2DInto across ResNet-50 and MNIST pooling shapes.
func Benchmark_Pool2DMetal(b *testing.B) {
	type testcase struct {
		name  string
		shape []int64
		cfg   Pool2DConfig
	}
	var cases = []testcase{
		{
			name:  "ResNet50Stem_32x64x112x112_k3s2p1",
			shape: []int64{32, 64, 112, 112},
			cfg:   Pool2DConfig{Kernel: [2]int64{3, 3}, Strides: [2]int64{2, 2}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}},
		},
		{
			name:  "ResNet50Global_32x512x7x7_k7",
			shape: []int64{32, 512, 7, 7},
			cfg:   Pool2DConfig{Kernel: [2]int64{7, 7}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}},
		},
		{
			name:  "MNIST_32x32x28x28_k2s2",
			shape: []int64{32, 32, 28, 28},
			cfg:   Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1}},
		},
	}

	type opCase struct {
		name    string
		gpuOp   func(*Scope, *Tensor, *Tensor, Pool2DConfig) error
		cpuOp   func(*tensor.Tensor, *tensor.Tensor, tensor.Pool2DConfig) error
		average bool
	}
	var ops = []opCase{
		{name: "MaxPool2D", gpuOp: (*Scope).MaxPool2D, cpuOp: tensor.MaxPool2DInto},
		{name: "AvgPool2D", gpuOp: (*Scope).AvgPool2D, cpuOp: tensor.AveragePool2DInto, average: true},
	}

	for _, op := range ops {
		for _, tc := range cases {
			var (
				outShape, shapeErr = Pool2DShape(tc.shape, tc.cfg)
				inCount            = tc.shape[0] * tc.shape[1] * tc.shape[2] * tc.shape[3]
				outCount           = outShape[0] * outShape[1] * outShape[2] * outShape[3]
				inData             = attentionValues(inCount, 290, 1)
				bytesMoved         = float64((inCount + outCount) * 4)
			)
			requireError(b, shapeErr, nil)

			b.Run(fmt.Sprintf("%s/Metal/%s", op.name, tc.name), func(b *testing.B) {
				var (
					dev    = requireMetalDevice(b)
					ctx    = context.Background()
					input  = requireTensor(b, dev, tc.shape...)
					output = requireTensor(b, dev, outShape...)
				)
				requireError(b, input.Upload(inData), nil)
				runOnce := func() (elapsed time.Duration) {
					var (
						start      = time.Now()
						scope, err = dev.NewScope()
					)
					requireError(b, err, nil)
					defer scope.Close()
					requireError(b, op.gpuOp(scope, input, output, tc.cfg), nil)
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
				var gbPerSec = (bytesMoved / 1e9) / (ms / 1e3)
				b.ReportMetric(ms, "ms/op")
				b.ReportMetric(gbPerSec, "GB/s")
			})

			b.Run(fmt.Sprintf("%s/CPU/%s", op.name, tc.name), func(b *testing.B) {
				var (
					cpuIn, e1  = tensor.FromFloat32(tc.shape, inData)
					cpuOut, e2 = tensor.FromFloat32(outShape, make([]float32, outCount))
					cpuCfg     = tensor.Pool2DConfig{
						Kernel:          tc.cfg.Kernel,
						Strides:         tc.cfg.Strides,
						Dilations:       tc.cfg.Dilations,
						Pads:            tc.cfg.Pads,
						CeilMode:        tc.cfg.CeilMode,
						CountIncludePad: tc.cfg.CountIncludePad,
					}
				)
				requireError(b, e1, nil)
				requireError(b, e2, nil)
				var total time.Duration
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					var start = time.Now()
					requireError(b, op.cpuOp(cpuIn, cpuOut, cpuCfg), nil)
					total += time.Since(start)
				}
				b.StopTimer()
				var ms = float64(total.Nanoseconds()) / float64(b.N) / 1e6
				var gbPerSec = (bytesMoved / 1e9) / (ms / 1e3)
				b.ReportMetric(ms, "ms/op")
				b.ReportMetric(gbPerSec, "GB/s")
			})
		}
	}
}
