//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

// Benchmark_Conv2DMetal measures isolated Scope.Conv2D across Float32, Float16,
// and BFloat16 weights on Apple Silicon Metal against paired CPU
// tensor.Conv2DInto for ResNet-50, ViT-patch, and MNIST shapes.
func Benchmark_Conv2DMetal(b *testing.B) {
	type testcase struct {
		name    string
		inShape []int64
		wShape  []int64
		cfg     Conv2DConfig
	}
	var cases = []testcase{
		{
			name:    "ResNet50_3x3_N8_64x56x56_64",
			inShape: []int64{8, 64, 56, 56},
			wShape:  []int64{64, 64, 3, 3},
			cfg:     Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 1},
		},
		{
			name:    "ResNet50_1x1_N8_256x56x56_64",
			inShape: []int64{8, 256, 56, 56},
			wShape:  []int64{64, 256, 1, 1},
			cfg:     Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1},
		},
		{
			name:    "ViTPatch_N8_3x224x224_768_k16s16",
			inShape: []int64{8, 3, 224, 224},
			wShape:  []int64{768, 3, 16, 16},
			cfg:     Conv2DConfig{Strides: [2]int64{16, 16}, Dilations: [2]int64{1, 1}, Groups: 1},
		},
		{
			name:    "MNIST_N32_1x28x28_32_k5p2",
			inShape: []int64{32, 1, 28, 28},
			wShape:  []int64{32, 1, 5, 5},
			cfg:     Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{2, 2, 2, 2}, Dilations: [2]int64{1, 1}, Groups: 1},
		},
	}

	var dtypes = []DType{Float32, Float16, BFloat16}

	for _, tc := range cases {
		var (
			outShape, shapeErr = Conv2DShape(tc.inShape, tc.wShape, tc.cfg)
			inCount            = tc.inShape[0] * tc.inShape[1] * tc.inShape[2] * tc.inShape[3]
			wCount             = tc.wShape[0] * tc.wShape[1] * tc.wShape[2] * tc.wShape[3]
			outCount           = outShape[0] * outShape[1] * outShape[2] * outShape[3]
			inData             = attentionValues(inCount, 601, 1)
			wData              = attentionValues(wCount, 607, 2)
			bData              = attentionValues(tc.wShape[0], 613, 3)
			flops              = 2.0 * float64(outCount) * float64(tc.wShape[1]*tc.wShape[2]*tc.wShape[3])
		)
		requireError(b, shapeErr, nil)

		for _, dt := range dtypes {
			b.Run(fmt.Sprintf("Metal/%s/%s", dt, tc.name), func(b *testing.B) {
				var (
					dev  = requireMetalDevice(b)
					ctx  = context.Background()
					inT  = requireTensor(b, dev, tc.inShape...)
					wT   = requireTensorWithDType(b, dev, dt, tc.wShape...)
					bT   = requireTensorWithDType(b, dev, dt, tc.wShape[0])
					outT = requireTensor(b, dev, outShape...)
				)
				requireError(b, inT.Upload(inData), nil)
				if dt == Float32 {
					requireError(b, wT.Upload(wData), nil)
					requireError(b, bT.Upload(bData), nil)
				} else {
					requireError(b, wT.UploadFloat32As(wData), nil)
					requireError(b, bT.UploadFloat32As(bData), nil)
				}
				runOnce := func() (elapsed time.Duration) {
					var (
						start      = time.Now()
						scope, err = dev.NewScope()
					)
					requireError(b, err, nil)
					defer scope.Close()
					requireError(b, scope.Conv2D(inT, wT, bT, outT, tc.cfg), nil)
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
				var gflops = (flops / 1e9) / (ms / 1e3)
				b.ReportMetric(ms, "ms/op")
				b.ReportMetric(gflops, "GFLOP/s")
			})
		}

		b.Run(fmt.Sprintf("CPU/float32/%s", tc.name), func(b *testing.B) {
			var (
				cpuIn, e1  = tensor.FromFloat32(tc.inShape, inData)
				cpuW, e2   = tensor.FromFloat32(tc.wShape, wData)
				cpuB, e3   = tensor.FromFloat32([]int64{tc.wShape[0]}, bData)
				cpuOut, e4 = tensor.FromFloat32(outShape, make([]float32, outCount))
				cpuCfg     = tensor.Conv2DConfig{
					Strides:   tc.cfg.Strides,
					Dilations: tc.cfg.Dilations,
					Pads:      tc.cfg.Pads,
					Groups:    tc.cfg.Groups,
				}
			)
			requireError(b, e1, nil)
			requireError(b, e2, nil)
			requireError(b, e3, nil)
			requireError(b, e4, nil)
			var total time.Duration
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				var start = time.Now()
				requireError(b, tensor.Conv2DInto(cpuIn, cpuW, cpuB, cpuOut, cpuCfg), nil)
				total += time.Since(start)
			}
			b.StopTimer()
			var ms = float64(total.Nanoseconds()) / float64(b.N) / 1e6
			var gflops = (flops / 1e9) / (ms / 1e3)
			b.ReportMetric(ms, "ms/op")
			b.ReportMetric(gflops, "GFLOP/s")
		})
	}
}

// Benchmark_ResNetBottleneckMetal benchmarks a complete 11-op ResNet-50
// bottleneck (Conv2D 1x1 -> BatchNorm -> ReLU -> Conv2D 3x3 -> BatchNorm ->
// ReLU -> Conv2D 1x1 -> BatchNorm -> Add -> ReLU -> MaxPool2D) in a single
// gpu.Scope against paired CPU tensor execution.
func Benchmark_ResNetBottleneckMetal(b *testing.B) {
	var (
		n     = int64(8)
		cIn   = int64(256)
		cMid  = int64(64)
		h, w  = int64(56), int64(56)
		eps   = float32(1e-5)
		cfg1  = Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		cfg3  = Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		pCfg  = Pool2DConfig{Kernel: [2]int64{3, 3}, Strides: [2]int64{2, 2}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}}
		inDat = attentionValues(n*cIn*h*w, 701, 1)
		w1Dat = attentionValues(cMid*cIn*1*1, 709, 2)
		w2Dat = attentionValues(cMid*cMid*3*3, 719, 3)
		w3Dat = attentionValues(cIn*cMid*1*1, 727, 4)
	)
	makeBN := func(ch int64, seed uint64) (sc, bi, mn, va []float32) {
		sc = attentionValues(ch, seed, 1)
		bi = attentionValues(ch, seed+1, 2)
		mn = attentionValues(ch, seed+2, 3)
		va = make([]float32, ch)
		for i := range va {
			va[i] = 0.5 + float32(i%7)*0.1
		}
		return sc, bi, mn, va
	}
	var (
		sc1, bi1, mn1, va1 = makeBN(cMid, 801)
		sc2, bi2, mn2, va2 = makeBN(cMid, 811)
		sc3, bi3, mn3, va3 = makeBN(cIn, 821)
	)

	for _, dt := range []DType{Float32, Float16, BFloat16} {
		b.Run(fmt.Sprintf("Metal/%s", dt), func(b *testing.B) {
			var (
				dev    = requireMetalDevice(b)
				ctx    = context.Background()
				inT    = requireTensor(b, dev, n, cIn, h, w)
				w1T    = requireTensorWithDType(b, dev, dt, cMid, cIn, 1, 1)
				w2T    = requireTensorWithDType(b, dev, dt, cMid, cMid, 3, 3)
				w3T    = requireTensorWithDType(b, dev, dt, cIn, cMid, 1, 1)
				sc1T   = requireTensor(b, dev, cMid)
				bi1T   = requireTensor(b, dev, cMid)
				mn1T   = requireTensor(b, dev, cMid)
				va1T   = requireTensor(b, dev, cMid)
				sc2T   = requireTensor(b, dev, cMid)
				bi2T   = requireTensor(b, dev, cMid)
				mn2T   = requireTensor(b, dev, cMid)
				va2T   = requireTensor(b, dev, cMid)
				sc3T   = requireTensor(b, dev, cIn)
				bi3T   = requireTensor(b, dev, cIn)
				mn3T   = requireTensor(b, dev, cIn)
				va3T   = requireTensor(b, dev, cIn)
				c1Out  = requireTensor(b, dev, n, cMid, h, w)
				bn1Out = requireTensor(b, dev, n, cMid, h, w)
				r1Out  = requireTensor(b, dev, n, cMid, h, w)
				c2Out  = requireTensor(b, dev, n, cMid, h, w)
				bn2Out = requireTensor(b, dev, n, cMid, h, w)
				r2Out  = requireTensor(b, dev, n, cMid, h, w)
				c3Out  = requireTensor(b, dev, n, cIn, h, w)
				bn3Out = requireTensor(b, dev, n, cIn, h, w)
				addOut = requireTensor(b, dev, n, cIn, h, w)
				r3Out  = requireTensor(b, dev, n, cIn, h, w)
				pOut   = requireTensor(b, dev, n, cIn, h/2, w/2)
			)
			requireError(b, inT.Upload(inDat), nil)
			if dt == Float32 {
				requireError(b, w1T.Upload(w1Dat), nil)
				requireError(b, w2T.Upload(w2Dat), nil)
				requireError(b, w3T.Upload(w3Dat), nil)
			} else {
				requireError(b, w1T.UploadFloat32As(w1Dat), nil)
				requireError(b, w2T.UploadFloat32As(w2Dat), nil)
				requireError(b, w3T.UploadFloat32As(w3Dat), nil)
			}
			requireError(b, sc1T.Upload(sc1), nil)
			requireError(b, bi1T.Upload(bi1), nil)
			requireError(b, mn1T.Upload(mn1), nil)
			requireError(b, va1T.Upload(va1), nil)
			requireError(b, sc2T.Upload(sc2), nil)
			requireError(b, bi2T.Upload(bi2), nil)
			requireError(b, mn2T.Upload(mn2), nil)
			requireError(b, va2T.Upload(va2), nil)
			requireError(b, sc3T.Upload(sc3), nil)
			requireError(b, bi3T.Upload(bi3), nil)
			requireError(b, mn3T.Upload(mn3), nil)
			requireError(b, va3T.Upload(va3), nil)

			runOnce := func() (elapsed time.Duration) {
				var (
					start      = time.Now()
					scope, err = dev.NewScope()
				)
				requireError(b, err, nil)
				defer scope.Close()
				requireError(b, scope.Conv2D(inT, w1T, nil, c1Out, cfg1), nil)
				requireError(b, scope.BatchNorm(c1Out, sc1T, bi1T, mn1T, va1T, bn1Out, eps), nil)
				requireError(b, scope.ReLU(bn1Out, r1Out), nil)
				requireError(b, scope.Conv2D(r1Out, w2T, nil, c2Out, cfg3), nil)
				requireError(b, scope.BatchNorm(c2Out, sc2T, bi2T, mn2T, va2T, bn2Out, eps), nil)
				requireError(b, scope.ReLU(bn2Out, r2Out), nil)
				requireError(b, scope.Conv2D(r2Out, w3T, nil, c3Out, cfg1), nil)
				requireError(b, scope.BatchNorm(c3Out, sc3T, bi3T, mn3T, va3T, bn3Out, eps), nil)
				requireError(b, scope.Add(bn3Out, inT, addOut), nil)
				requireError(b, scope.ReLU(addOut, r3Out), nil)
				requireError(b, scope.MaxPool2D(r3Out, pOut, pCfg), nil)
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
			b.ReportMetric(ms, "ms/op")
		})
	}

	b.Run("CPU/float32", func(b *testing.B) {
		mustTensor := func(shape []int64, data []float32) *tensor.Tensor {
			var out, e = tensor.FromFloat32(shape, data)
			requireError(b, e, nil)
			return out
		}
		var (
			cInT  = mustTensor([]int64{n, cIn, h, w}, inDat)
			cW1   = mustTensor([]int64{cMid, cIn, 1, 1}, w1Dat)
			cW2   = mustTensor([]int64{cMid, cMid, 3, 3}, w2Dat)
			cW3   = mustTensor([]int64{cIn, cMid, 1, 1}, w3Dat)
			cSc1  = mustTensor([]int64{cMid}, sc1)
			cBi1  = mustTensor([]int64{cMid}, bi1)
			cMn1  = mustTensor([]int64{cMid}, mn1)
			cVa1  = mustTensor([]int64{cMid}, va1)
			cSc2  = mustTensor([]int64{cMid}, sc2)
			cBi2  = mustTensor([]int64{cMid}, bi2)
			cMn2  = mustTensor([]int64{cMid}, mn2)
			cVa2  = mustTensor([]int64{cMid}, va2)
			cSc3  = mustTensor([]int64{cIn}, sc3)
			cBi3  = mustTensor([]int64{cIn}, bi3)
			cMn3  = mustTensor([]int64{cIn}, mn3)
			cVa3  = mustTensor([]int64{cIn}, va3)
			cTmp1 = mustTensor([]int64{n, cMid, h, w}, make([]float32, n*cMid*h*w))
			cTmp2 = mustTensor([]int64{n, cMid, h, w}, make([]float32, n*cMid*h*w))
			cTmp3 = mustTensor([]int64{n, cIn, h, w}, make([]float32, n*cIn*h*w))
			cTmp4 = mustTensor([]int64{n, cIn, h, w}, make([]float32, n*cIn*h*w))
			cPool = mustTensor([]int64{n, cIn, h / 2, w / 2}, make([]float32, n*cIn*(h/2)*(w/2)))
			tCfg1 = tensor.Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
			tCfg3 = tensor.Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
			tPCfg = tensor.Pool2DConfig{Kernel: [2]int64{3, 3}, Strides: [2]int64{2, 2}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}}
		)
		var total time.Duration
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			var start = time.Now()
			requireError(b, tensor.Conv2DInto(cInT, cW1, nil, cTmp1, tCfg1), nil)
			requireError(b, tensor.BatchNormalizationInto(cTmp1, cSc1, cBi1, cMn1, cVa1, cTmp2, eps), nil)
			requireError(b, tensor.ReLUInto(cTmp2, cTmp1), nil)
			requireError(b, tensor.Conv2DInto(cTmp1, cW2, nil, cTmp2, tCfg3), nil)
			requireError(b, tensor.BatchNormalizationInto(cTmp2, cSc2, cBi2, cMn2, cVa2, cTmp1, eps), nil)
			requireError(b, tensor.ReLUInto(cTmp1, cTmp2), nil)
			requireError(b, tensor.Conv2DInto(cTmp2, cW3, nil, cTmp3, tCfg1), nil)
			requireError(b, tensor.BatchNormalizationInto(cTmp3, cSc3, cBi3, cMn3, cVa3, cTmp4, eps), nil)
			requireError(b, tensor.AddInto(cTmp4, cInT, cTmp3), nil)
			requireError(b, tensor.ReLUInto(cTmp3, cTmp4), nil)
			requireError(b, tensor.MaxPool2DInto(cTmp4, cPool, tPCfg), nil)
			total += time.Since(start)
		}
		b.StopTimer()
		var ms = float64(total.Nanoseconds()) / float64(b.N) / 1e6
		b.ReportMetric(ms, "ms/op")
	})
}
