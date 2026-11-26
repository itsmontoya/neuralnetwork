//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"math"
	"testing"
	"time"
)

func Benchmark_TransposeMetal(b *testing.B) {
	type benchCase struct {
		name  string
		shape []int64
		dst   []int64
		perm  []int
	}
	var cases = []benchCase{
		{name: "2D_1024x1024", shape: []int64{1024, 1024}, dst: []int64{1024, 1024}, perm: []int{1, 0}},
		{name: "4D_2x64x12x64_BSHD_to_BHSD", shape: []int64{2, 64, 12, 64}, dst: []int64{2, 12, 64, 64}, perm: []int{0, 2, 1, 3}},
	}
	for _, tc := range cases {
		for _, dt := range []DType{Float32, Float16} {
			b.Run(fmt.Sprintf("%s/%s", tc.name, dt), func(b *testing.B) {
				var (
					dev = requireMetalDevice(b)
					ctx = context.Background()
					src = requireTensorWithDType(b, dev, dt, tc.shape...)
					dst = requireTensorWithDType(b, dev, dt, tc.dst...)
				)
				requireError(b, src.UploadFloat32As(attentionValues(int64(src.state.storage.count), 301, 1)), nil)
				runOnce := func() (elapsed time.Duration) {
					var (
						start      = time.Now()
						scope, err = dev.NewScope()
					)
					requireError(b, err, nil)
					defer scope.Close()
					requireError(b, scope.Transpose(src, dst, tc.perm), nil)
					requireError(b, scope.Submit(ctx), nil)
					requireError(b, scope.Wait(ctx), nil)
					requireError(b, scope.Close(), nil)
					return time.Since(start)
				}
				_ = runOnce()
				b.ResetTimer()
				var total time.Duration
				for range b.N {
					total += runOnce()
				}
				b.StopTimer()
				b.ReportMetric(float64(total.Nanoseconds())/float64(b.N)/1e6, "ms/op")
			})
		}
	}
}

func Benchmark_MatMulLeftTransposeMetal(b *testing.B) {
	for _, dt := range []DType{Float32, Float16, BFloat16} {
		b.Run(fmt.Sprintf("K768_M32_N768/%s", dt), func(b *testing.B) {
			var (
				dev   = requireMetalDevice(b)
				ctx   = context.Background()
				left  = requireTensor(b, dev, 768, 32)
				right = requireTensorWithDType(b, dev, dt, 768, 768)
				dst   = requireTensor(b, dev, 32, 768)
			)
			requireError(b, left.Upload(attentionValues(768*32, 302, 1)), nil)
			requireError(b, right.UploadFloat32As(attentionValues(768*768, 303, 1)), nil)
			runOnce := func() (elapsed time.Duration) {
				var (
					start      = time.Now()
					scope, err = dev.NewScope()
				)
				requireError(b, err, nil)
				defer scope.Close()
				requireError(b, scope.MatMulLeftTranspose(left, right, dst), nil)
				requireError(b, scope.Submit(ctx), nil)
				requireError(b, scope.Wait(ctx), nil)
				requireError(b, scope.Close(), nil)
				return time.Since(start)
			}
			_ = runOnce()
			b.ResetTimer()
			var total time.Duration
			for range b.N {
				total += runOnce()
			}
			b.StopTimer()
			b.ReportMetric(float64(total.Nanoseconds())/float64(b.N)/1e6, "ms/op")
		})
	}
}

func Benchmark_BatchedMatMulMetal(b *testing.B) {
	type benchCase struct {
		name       string
		leftShape  []int64
		rightShape []int64
		dstShape   []int64
		rightT     bool
		dtypes     []DType
	}
	var cases = []benchCase{
		{
			name:       "AttnScores_2x12x64x64_RT",
			leftShape:  []int64{2, 12, 64, 64},
			rightShape: []int64{2, 12, 64, 64},
			dstShape:   []int64{2, 12, 64, 64},
			rightT:     true,
			dtypes:     []DType{Float32},
		},
		{
			name:       "AttnContext_2x12x64x64",
			leftShape:  []int64{2, 12, 64, 64},
			rightShape: []int64{2, 12, 64, 64},
			dstShape:   []int64{2, 12, 64, 64},
			rightT:     false,
			dtypes:     []DType{Float32},
		},
		{
			name:       "SharedProj_2x64x768_x_768x768",
			leftShape:  []int64{2, 64, 768},
			rightShape: []int64{768, 768},
			dstShape:   []int64{2, 64, 768},
			rightT:     false,
			dtypes:     []DType{Float32, Float16, BFloat16},
		},
	}
	for _, tc := range cases {
		for _, dt := range tc.dtypes {
			b.Run(fmt.Sprintf("%s/%s", tc.name, dt), func(b *testing.B) {
				var (
					dev   = requireMetalDevice(b)
					ctx   = context.Background()
					left  = requireTensor(b, dev, tc.leftShape...)
					right = requireTensorWithDType(b, dev, dt, tc.rightShape...)
					dst   = requireTensor(b, dev, tc.dstShape...)
				)
				requireError(b, left.Upload(attentionValues(int64(left.state.storage.count), 304, 1)), nil)
				requireError(b, right.UploadFloat32As(attentionValues(int64(right.state.storage.count), 305, 1)), nil)
				runOnce := func() (elapsed time.Duration) {
					var (
						start      = time.Now()
						scope, err = dev.NewScope()
					)
					requireError(b, err, nil)
					defer scope.Close()
					if tc.rightT {
						requireError(b, scope.BatchedMatMulRightTranspose(left, right, dst), nil)
					} else {
						requireError(b, scope.BatchedMatMul(left, right, dst), nil)
					}
					requireError(b, scope.Submit(ctx), nil)
					requireError(b, scope.Wait(ctx), nil)
					requireError(b, scope.Close(), nil)
					return time.Since(start)
				}
				_ = runOnce()
				b.ResetTimer()
				var total time.Duration
				for range b.N {
					total += runOnce()
				}
				b.StopTimer()
				b.ReportMetric(float64(total.Nanoseconds())/float64(b.N)/1e6, "ms/op")
			})
		}
	}
}

// Benchmark_ONNXTransformerEncoderBlockMetal measures a complete synthetic
// BERT-base-sized ONNX Transformer encoder block (B=2, S=64, H=12, D=64,
// E=768, FFN=3072) executing in one resident gpu.Scope with zero host copies.
func Benchmark_ONNXTransformerEncoderBlockMetal(b *testing.B) {
	const (
		B   = int64(2)
		S   = int64(64)
		H   = int64(12)
		D   = int64(64)
		E   = H * D
		FFN = E * 4
		eps = float32(1e-5)
	)
	var scale = float32(1.0 / math.Sqrt(float64(D)))

	for _, wt := range []DType{Float32, Float16, BFloat16} {
		b.Run(fmt.Sprintf("BERTBase_B2_S64_E768/%s", wt), func(b *testing.B) {
			var (
				dev     = requireMetalDevice(b)
				ctx     = context.Background()
				x       = requireTensor(b, dev, B, S, E)
				ln1Out  = requireTensor(b, dev, B, S, E)
				ln1W    = requireTensorWithDType(b, dev, wt, E)
				ln1B    = requireTensorWithDType(b, dev, wt, E)
				wq      = requireTensorWithDType(b, dev, wt, E, E)
				wk      = requireTensorWithDType(b, dev, wt, E, E)
				wv      = requireTensorWithDType(b, dev, wt, E, E)
				wo      = requireTensorWithDType(b, dev, wt, E, E)
				bo      = requireTensor(b, dev, E)
				ln2W    = requireTensorWithDType(b, dev, wt, E)
				ln2B    = requireTensorWithDType(b, dev, wt, E)
				w1      = requireTensorWithDType(b, dev, wt, E, FFN)
				b1      = requireTensor(b, dev, FFN)
				w2      = requireTensorWithDType(b, dev, wt, FFN, E)
				b2      = requireTensor(b, dev, E)
				qProj   = requireTensor(b, dev, B, S, E)
				kProj   = requireTensor(b, dev, B, S, E)
				vProj   = requireTensor(b, dev, B, S, E)
				qHeads  = requireTensor(b, dev, B, H, S, D)
				kHeads  = requireTensor(b, dev, B, H, S, D)
				vHeads  = requireTensor(b, dev, B, H, S, D)
				scores  = requireTensor(b, dev, B, H, S, S)
				scaled  = requireTensor(b, dev, B, H, S, S)
				probs   = requireTensor(b, dev, B, H, S, S)
				ctxH    = requireTensor(b, dev, B, H, S, D)
				ctxS    = requireTensor(b, dev, B, S, H, D)
				attnOut = requireTensor(b, dev, B, S, E)
				attnB   = requireTensor(b, dev, B, S, E)
				res1    = requireTensor(b, dev, B, S, E)
				ln2Out  = requireTensor(b, dev, B, S, E)
				ffn1    = requireTensor(b, dev, B, S, FFN)
				ffn1B   = requireTensor(b, dev, B, S, FFN)
				ffnAct  = requireTensor(b, dev, B, S, FFN)
				ffn2    = requireTensor(b, dev, B, S, E)
				ffn2B   = requireTensor(b, dev, B, S, E)
				finalY  = requireTensor(b, dev, B, S, E)
			)
			var (
				q4d, err1   = qProj.View([]int64{B, S, H, D})
				k4d, err2   = kProj.View([]int64{B, S, H, D})
				v4d, err3   = vProj.View([]int64{B, S, H, D})
				ctx3d, err4 = ctxS.View([]int64{B, S, E})
			)
			requireError(b, err1, nil)
			requireError(b, err2, nil)
			requireError(b, err3, nil)
			requireError(b, err4, nil)
			b.Cleanup(func() {
				_ = q4d.Close()
				_ = k4d.Close()
				_ = v4d.Close()
				_ = ctx3d.Close()
			})

			requireError(b, x.Upload(attentionValues(B*S*E, 310, 1)), nil)
			requireError(b, ln1W.UploadFloat32As(attentionValues(E, 311, 1)), nil)
			requireError(b, ln1B.UploadFloat32As(attentionValues(E, 312, 1)), nil)
			requireError(b, wq.UploadFloat32As(attentionValues(E*E, 313, 1)), nil)
			requireError(b, wk.UploadFloat32As(attentionValues(E*E, 314, 1)), nil)
			requireError(b, wv.UploadFloat32As(attentionValues(E*E, 315, 1)), nil)
			requireError(b, wo.UploadFloat32As(attentionValues(E*E, 316, 1)), nil)
			requireError(b, bo.Upload(attentionValues(E, 317, 1)), nil)
			requireError(b, ln2W.UploadFloat32As(attentionValues(E, 318, 1)), nil)
			requireError(b, ln2B.UploadFloat32As(attentionValues(E, 319, 1)), nil)
			requireError(b, w1.UploadFloat32As(attentionValues(E*FFN, 320, 1)), nil)
			requireError(b, b1.Upload(attentionValues(FFN, 321, 1)), nil)
			requireError(b, w2.UploadFloat32As(attentionValues(FFN*E, 322, 1)), nil)
			requireError(b, b2.Upload(attentionValues(E, 323, 1)), nil)

			runOnce := func() (elapsed time.Duration) {
				var (
					start      = time.Now()
					scope, err = dev.NewScope()
				)
				requireError(b, err, nil)
				defer scope.Close()
				requireError(b, scope.LayerNorm(x, ln1W, ln1B, ln1Out, eps), nil)
				requireError(b, scope.BatchedMatMul(ln1Out, wq, qProj), nil)
				requireError(b, scope.BatchedMatMul(ln1Out, wk, kProj), nil)
				requireError(b, scope.BatchedMatMul(ln1Out, wv, vProj), nil)
				requireError(b, scope.Transpose(q4d, qHeads, []int{0, 2, 1, 3}), nil)
				requireError(b, scope.Transpose(k4d, kHeads, []int{0, 2, 1, 3}), nil)
				requireError(b, scope.Transpose(v4d, vHeads, []int{0, 2, 1, 3}), nil)
				requireError(b, scope.BatchedMatMulRightTranspose(qHeads, kHeads, scores), nil)
				requireError(b, scope.Scale(scores, scaled, scale), nil)
				requireError(b, scope.Softmax(scaled, probs), nil)
				requireError(b, scope.BatchedMatMul(probs, vHeads, ctxH), nil)
				requireError(b, scope.Transpose(ctxH, ctxS, []int{0, 2, 1, 3}), nil)
				requireError(b, scope.BatchedMatMul(ctx3d, wo, attnOut), nil)
				requireError(b, scope.AddBroadcast(attnOut, bo, attnB), nil)
				requireError(b, scope.Add(x, attnB, res1), nil)
				requireError(b, scope.LayerNorm(res1, ln2W, ln2B, ln2Out, eps), nil)
				requireError(b, scope.BatchedMatMul(ln2Out, w1, ffn1), nil)
				requireError(b, scope.AddBroadcast(ffn1, b1, ffn1B), nil)
				requireError(b, scope.ReLU(ffn1B, ffnAct), nil)
				requireError(b, scope.BatchedMatMul(ffnAct, w2, ffn2), nil)
				requireError(b, scope.AddBroadcast(ffn2, b2, ffn2B), nil)
				requireError(b, scope.Add(res1, ffn2B, finalY), nil)
				requireError(b, scope.Submit(ctx), nil)
				requireError(b, scope.Wait(ctx), nil)
				requireError(b, scope.Close(), nil)
				return time.Since(start)
			}
			_ = runOnce()
			b.ResetTimer()
			var total time.Duration
			for range b.N {
				total += runOnce()
			}
			b.StopTimer()
			b.ReportMetric(float64(total.Nanoseconds())/float64(b.N)/1e6, "ms/op")
		})
	}
}

func Benchmark_BatchedMatMulTiledSweepMetal(b *testing.B) {
	type sweepCase struct {
		name       string
		leftShape  []int64
		rightShape []int64
		dstShape   []int64
		rightT     bool
		leftT      bool
		flops      float64
		dtypes     []DType
	}
	var cases = []sweepCase{
		{
			name:       "LeftT_K768_M128_N3072",
			leftShape:  []int64{768, 128},
			rightShape: []int64{768, 3072},
			dstShape:   []int64{128, 3072},
			leftT:      true,
			flops:      2 * 128 * 768 * 3072,
			dtypes:     []DType{Float32, Float16, BFloat16},
		},
		{
			name:       "SharedProj_2x64x768_x_768x3072",
			leftShape:  []int64{2, 64, 768},
			rightShape: []int64{768, 3072},
			dstShape:   []int64{2, 64, 3072},
			flops:      2 * 2 * 64 * 768 * 3072,
			dtypes:     []DType{Float32, Float16, BFloat16},
		},
		{
			name:       "SharedProj_2x64x3072_x_768x3072_RT",
			leftShape:  []int64{2, 64, 3072},
			rightShape: []int64{768, 3072},
			dstShape:   []int64{2, 64, 768},
			rightT:     true,
			flops:      2 * 2 * 64 * 3072 * 768,
			dtypes:     []DType{Float32, Float16, BFloat16},
		},
		{
			name:       "AttnScores_2x12x256x64_RT",
			leftShape:  []int64{2, 12, 256, 64},
			rightShape: []int64{2, 12, 256, 64},
			dstShape:   []int64{2, 12, 256, 256},
			rightT:     true,
			flops:      2 * 2 * 12 * 256 * 64 * 256,
			dtypes:     []DType{Float32},
		},
		{
			name:       "AttnContext_2x12x256x256",
			leftShape:  []int64{2, 12, 256, 256},
			rightShape: []int64{2, 12, 256, 64},
			dstShape:   []int64{2, 12, 256, 64},
			flops:      2 * 2 * 12 * 256 * 256 * 64,
			dtypes:     []DType{Float32},
		},
	}

	for _, tc := range cases {
		for _, dt := range tc.dtypes {
			b.Run(fmt.Sprintf("%s/%s", tc.name, dt), func(b *testing.B) {
				var (
					dev   = requireMetalDevice(b)
					ctx   = context.Background()
					left  = requireTensor(b, dev, tc.leftShape...)
					right = requireTensorWithDType(b, dev, dt, tc.rightShape...)
					dst   = requireTensor(b, dev, tc.dstShape...)
				)
				requireError(b, left.Upload(attentionValues(int64(left.state.storage.count), 330, 1)), nil)
				requireError(b, right.UploadFloat32As(attentionValues(int64(right.state.storage.count), 331, 1)), nil)
				runOnce := func() (elapsed time.Duration) {
					var (
						start      = time.Now()
						scope, err = dev.NewScope()
					)
					requireError(b, err, nil)
					defer scope.Close()
					switch {
					case tc.leftT:
						requireError(b, scope.MatMulLeftTranspose(left, right, dst), nil)
					case tc.rightT:
						requireError(b, scope.BatchedMatMulRightTranspose(left, right, dst), nil)
					default:
						requireError(b, scope.BatchedMatMul(left, right, dst), nil)
					}
					requireError(b, scope.Submit(ctx), nil)
					requireError(b, scope.Wait(ctx), nil)
					requireError(b, scope.Close(), nil)
					return time.Since(start)
				}
				_ = runOnce()
				b.ResetTimer()
				var total time.Duration
				for range b.N {
					total += runOnce()
				}
				b.StopTimer()
				var secPerOp = float64(total.Nanoseconds()) / float64(b.N) / 1e9
				b.ReportMetric(secPerOp*1e3, "ms/op")
				b.ReportMetric(tc.flops/secPerOp/1e9, "GFLOP/s")
			})
		}
	}
}
