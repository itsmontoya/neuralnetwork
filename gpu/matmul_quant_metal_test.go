//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
	"testing"
)

func Test_MetalTensorMatMulQuant(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
	)
	type shapeCase struct {
		name              string
		rows, inner, cols int64
		zero              bool
	}
	var cases = []shapeCase{
		{name: "zero_m1_k32_n1", rows: 1, inner: 32, cols: 1, zero: true},
		{name: "zero_m3_k64_n7", rows: 3, inner: 64, cols: 7, zero: true},
		{name: "row_rt_k32_n1", rows: 1, inner: 32, cols: 1},
		{name: "row_rt_k32_n3", rows: 1, inner: 32, cols: 3},
		{name: "row_rt_k64_n9", rows: 1, inner: 64, cols: 9},
		{name: "row_rt_k96_n15", rows: 1, inner: 96, cols: 15},
		{name: "row_rt_k160_n17", rows: 1, inner: 160, cols: 17},
		{name: "row_rt_k256_n33", rows: 1, inner: 256, cols: 33},
		{name: "multi_rt_m2_k32_n17", rows: 2, inner: 32, cols: 17},
		{name: "multi_rt_m3_k96_n17", rows: 3, inner: 96, cols: 17},
		{name: "multi_rt_m4_k64_n33", rows: 4, inner: 64, cols: 33},
		{name: "tiled_partial_m8_k32_n9", rows: 8, inner: 32, cols: 9},
		{name: "tiled_m15_k64_n17", rows: 15, inner: 64, cols: 17},
		{name: "tiled_m16_k256_n33", rows: 16, inner: 256, cols: 33},
		{name: "multi_rt_m17_k160_n33", rows: 17, inner: 160, cols: 33},
		{name: "tiled_m31_k64_n65", rows: 31, inner: 64, cols: 65},
		{name: "tiled_exact_m32_k256_n64", rows: 32, inner: 256, cols: 64},
		{name: "tiled_edge_m33_k128_n65", rows: 33, inner: 128, cols: 65},
		{name: "tiled_m64_k256_n512", rows: 64, inner: 256, cols: 512},
		{name: "tiled_m128_k256_n128", rows: 128, inner: 256, cols: 128},
		{name: "tiled_small_n_m32_k64_n3", rows: 32, inner: 64, cols: 3},
		{name: "projection_attn_kv_decode", rows: 1, inner: 2048, cols: 512},
		{name: "projection_attn_qo_decode", rows: 1, inner: 2048, cols: 2048},
		{name: "projection_gate_up_decode", rows: 1, inner: 2048, cols: 8192},
		{name: "projection_down_decode", rows: 1, inner: 8192, cols: 2048},
		{name: "projection_classifier_decode", rows: 1, inner: 2048, cols: 128256},
		{name: "projection_prefill_6", rows: 6, inner: 2048, cols: 8192},
		{name: "projection_prefill_27", rows: 27, inner: 2048, cols: 8192},
		{name: "projection_prefill_28", rows: 28, inner: 2048, cols: 8192},
		{name: "projection_prefill_29", rows: 29, inner: 2048, cols: 8192},
		{name: "projection_down_prefill_29", rows: 29, inner: 8192, cols: 2048},
	}

	for _, dtype := range []DType{Q8_0, Q4_0, Q4_1} {
		t.Run(dtype.String(), func(t *testing.T) {
			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					var (
						left   = make([]float32, tc.rows*tc.inner)
						weight = make([]float32, tc.cols*tc.inner)
						random = rand.New(rand.NewPCG(0x6e6e12, uint64(tc.rows*1000003+tc.inner*1009+tc.cols)))
					)
					if !tc.zero {
						for i := range left {
							left[i] = 2*random.Float32() - 1
						}
						for i := range weight {
							weight[i] = 2*random.Float32() - 1
						}
					}
					var (
						dequant = dequantizeWeightsSlice(t, weight, dtype)
						got     = runMetalMatMulQuant(t, gpu, dtype, tc.rows, tc.inner, tc.cols, left, weight)
					)
					checkMetalMatMul(t, tc.inner, tc.cols, left, dequant, got)
				})
			}
		})
	}

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("quantized matmul leaked native resources: before %+v; after %+v", initial, after)
	}
}

func Test_MetalTensorMatMulQuantSpecialValues(t *testing.T) {
	var (
		gpu = requireMetalDevice(t)
		inf = float32(math.Inf(1))
		nan = float32(math.NaN())
	)

	for _, dtype := range []DType{Q8_0, Q4_0, Q4_1} {
		t.Run(dtype.String(), func(t *testing.T) {
			var blockBytes = int(dtype.BlockBytes())
			makeBlock := func(scaleBits, minBits uint16, qVal int) (block []byte) {
				block = make([]byte, blockBytes)
				binary.LittleEndian.PutUint16(block[0:2], scaleBits)
				switch dtype {
				case Q8_0:
					for i := range QuantBlockSize {
						block[2+i] = byte(int8(qVal))
					}
				case Q4_0:
					var (
						u      = uint8(qVal + 8)
						packed = u | (u << 4)
					)
					for j := range 16 {
						block[2+j] = packed
					}
				case Q4_1:
					binary.LittleEndian.PutUint16(block[2:4], minBits)
					var (
						u      = uint8(qVal)
						packed = u | (u << 4)
					)
					for j := range 16 {
						block[4+j] = packed
					}
				}
				return block
			}

			type specialCase struct {
				name      string
				scaleBits uint16
				minBits   uint16
				qVal      int
				leftVal   float32
				want      float32
			}
			var cases []specialCase
			if dtype == Q4_1 {
				cases = []specialCase{
					{name: "zero_scale_zero_min_finite_left", scaleBits: 0x0000, minBits: 0x0000, qVal: 3, leftVal: 2.0, want: 0.0},
					{name: "zero_scale_nonzero_min", scaleBits: 0x0000, minBits: Float32ToFloat16(1.5), qVal: 5, leftVal: 2.0, want: 32 * 2.0 * 1.5},
					{name: "zero_scale_negative_min", scaleBits: 0x0000, minBits: Float32ToFloat16(-2.5), qVal: 0, leftVal: 1.5, want: 32 * 1.5 * (-2.5)},
					{name: "zero_scale_zero_min_inf_left", scaleBits: 0x0000, minBits: 0x0000, qVal: 3, leftVal: inf, want: nan},
					{name: "positive_scale_negative_min", scaleBits: Float32ToFloat16(0.5), minBits: Float32ToFloat16(-3.0), qVal: 4, leftVal: 2.0, want: 32 * 2.0 * (0.5*4.0 - 3.0)},
					{name: "extreme_max_fp16_scale", scaleBits: 0x7bff, minBits: 0x0000, qVal: 1, leftVal: 1.0, want: 32 * 65504.0},
					{name: "min_subnormal_scale", scaleBits: 0x0001, minBits: 0x0000, qVal: 1, leftVal: 1.0, want: 32 * Float16ToFloat32(0x0001)},
					{name: "nan_scale", scaleBits: 0x7e00, minBits: 0x0000, qVal: 1, leftVal: 1.0, want: nan},
					{name: "nan_min", scaleBits: Float32ToFloat16(1.0), minBits: 0x7e00, qVal: 1, leftVal: 1.0, want: nan},
					{name: "inf_scale_positive", scaleBits: 0x7c00, minBits: 0x0000, qVal: 1, leftVal: 1.0, want: inf},
					{name: "inf_scale_zero_left", scaleBits: 0x7c00, minBits: 0x0000, qVal: 1, leftVal: 0.0, want: nan},
					{name: "nan_left", scaleBits: Float32ToFloat16(1.0), minBits: Float32ToFloat16(-1.0), qVal: 2, leftVal: nan, want: nan},
					{name: "inf_left_neg_weight", scaleBits: Float32ToFloat16(0.5), minBits: Float32ToFloat16(-2.0), qVal: 1, leftVal: inf, want: -inf},
				}
			} else {
				cases = []specialCase{
					{name: "zero_scale_finite_left", scaleBits: 0x0000, qVal: 3, leftVal: 2.0, want: 0.0},
					{name: "zero_scale_inf_left", scaleBits: 0x0000, qVal: 3, leftVal: inf, want: nan},
					{name: "negative_scale", scaleBits: Float32ToFloat16(-2.5), qVal: 2, leftVal: 1.5, want: 32 * 1.5 * (-2.5 * 2)},
					{name: "extreme_max_fp16_scale", scaleBits: 0x7bff, qVal: 1, leftVal: 1.0, want: 32 * 65504.0},
					{name: "min_subnormal_scale", scaleBits: 0x0001, qVal: 1, leftVal: 1.0, want: 32 * Float16ToFloat32(0x0001)},
					{name: "nan_scale", scaleBits: 0x7e00, qVal: 1, leftVal: 1.0, want: nan},
					{name: "inf_scale_positive", scaleBits: 0x7c00, qVal: 1, leftVal: 1.0, want: inf},
					{name: "inf_scale_zero_left", scaleBits: 0x7c00, qVal: 1, leftVal: 0.0, want: nan},
					{name: "nan_left", scaleBits: Float32ToFloat16(1.0), qVal: 1, leftVal: nan, want: nan},
					{name: "inf_left", scaleBits: Float32ToFloat16(1.0), qVal: -1, leftVal: inf, want: -inf},
				}
				if dtype == Q8_0 {
					cases = append(cases, specialCase{
						name:      "q8_min_int8_minus_128",
						scaleBits: Float32ToFloat16(0.25),
						qVal:      -128,
						leftVal:   1.0,
						want:      32 * (0.25 * -128),
					})
				}
			}

			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					var (
						block = makeBlock(tc.scaleBits, tc.minBits, tc.qVal)
						left1 = make([]float32, 32)
						left2 = make([]float32, 64)
					)
					for i := range left1 {
						left1[i] = tc.leftVal
					}
					for i := range left2 {
						left2[i] = tc.leftVal
					}

					// M=1, N=1 (single-row tail without row1).
					var got1 = runMetalMatMulQuantRaw(t, gpu, dtype, 1, 32, 1, left1, block)
					if !sameSpecialFloat32(got1[0], tc.want) {
						t.Fatalf("M=1 N=1: got %g, want %g", got1[0], tc.want)
					}

					// M=1, N=2 (paired rows in SIMD group).
					var twoBlocks = append(slices.Clone(block), block...)
					var got2 = runMetalMatMulQuantRaw(t, gpu, dtype, 1, 32, 2, left1, twoBlocks)
					for col, v := range got2 {
						if !sameSpecialFloat32(v, tc.want) {
							t.Fatalf("M=1 N=2 col %d: got %g, want %g", col, v, tc.want)
						}
					}

					// M=2, N=2 (multi-row fallback kernel).
					var gotM2 = runMetalMatMulQuantRaw(t, gpu, dtype, 2, 32, 2, left2, twoBlocks)
					for idx, v := range gotM2 {
						if !sameSpecialFloat32(v, tc.want) {
							t.Fatalf("M=2 N=2 idx %d: got %g, want %g", idx, v, tc.want)
						}
					}

					// M=8, N=9 (tiled kernel partial edge tile).
					var (
						left8   = make([]float32, 8*32)
						blocks9 = make([]byte, 0, 9*blockBytes)
					)
					for i := range left8 {
						left8[i] = tc.leftVal
					}
					for range 9 {
						blocks9 = append(blocks9, block...)
					}
					var gotM8 = runMetalMatMulQuantRaw(t, gpu, dtype, 8, 32, 9, left8, blocks9)
					for idx, v := range gotM8 {
						if !sameSpecialFloat32(v, tc.want) {
							t.Fatalf("M=8 N=9 idx %d: got %g, want %g", idx, v, tc.want)
						}
					}

					// M=33, N=65 (tiled kernel interior + boundary tiles).
					var (
						left33   = make([]float32, 33*32)
						blocks65 = make([]byte, 0, 65*blockBytes)
					)
					for i := range left33 {
						left33[i] = tc.leftVal
					}
					for range 65 {
						blocks65 = append(blocks65, block...)
					}
					var gotM33 = runMetalMatMulQuantRaw(t, gpu, dtype, 33, 32, 65, left33, blocks65)
					for idx, v := range gotM33 {
						if !sameSpecialFloat32(v, tc.want) {
							t.Fatalf("M=33 N=65 idx %d: got %g, want %g", idx, v, tc.want)
						}
					}
				})
			}
		})
	}
}

func Test_MetalTensorMatMulQuantCopyAndConcurrency(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		ctx     = context.Background()
	)
	for _, dtype := range []DType{Q8_0, Q4_0, Q4_1} {
		t.Run(dtype.String(), func(t *testing.T) {
			var (
				weight       = requireTensorWithDType(t, gpu, dtype, 2, 32)
				weightCopy   = requireTensorWithDType(t, gpu, dtype, 2, 32)
				firstOutput  = requireTensor(t, gpu, 1, 2)
				secondOutput = requireTensor(t, gpu, 1, 2)
				rawValues    = make([]float32, 64)
			)
			for i := range 32 {
				rawValues[i] = 1.0
				rawValues[32+i] = -2.0
			}
			requireError(t, weight.UploadFloat32As(rawValues), nil)
			var dequant = dequantizeWeightsSlice(t, rawValues, dtype)
			var wantFirst, wantSecond [2]float32
			for i := range 32 {
				wantFirst[0] += 2.0 * dequant[i]
				wantFirst[1] += 2.0 * dequant[32+i]
				wantSecond[0] += -1.0 * dequant[i]
				wantSecond[1] += -1.0 * dequant[32+i]
			}

			// Verify Metal Copy between quantized tensors bit-for-bit.
			var (
				copyScope = requireScope(t, gpu)
				wantBytes = make([]byte, weight.Bytes())
				gotBytes  = make([]byte, weightCopy.Bytes())
			)
			requireError(t, weight.DownloadBytesInto(wantBytes), nil)
			requireError(t, copyScope.Copy(weight, weightCopy), nil)
			requireError(t, copyScope.Submit(ctx), nil)
			requireError(t, copyScope.Wait(ctx), nil)
			requireError(t, copyScope.Close(), nil)
			requireError(t, weightCopy.DownloadBytesInto(gotBytes), nil)
			if !slices.Equal(gotBytes, wantBytes) {
				t.Fatalf("%s Metal copy bytes mismatch", dtype)
			}

			// Concurrent scopes sharing immutable quantized weight + deferred Close.
			for iter := range 3 {
				var (
					first                     = requireScope(t, gpu)
					second                    = requireScope(t, gpu)
					firstInput                = requireTensor(t, gpu, 1, 32)
					secondInput               = requireTensor(t, gpu, 1, 32)
					firstResult, secondResult [2]float32
					results                   = make(chan error, 2)
				)
				requireError(t, first.Fill(firstInput, 2.0), nil)
				requireError(t, first.MatMulRightTranspose(firstInput, weight, firstOutput), nil)
				requireError(t, second.Fill(secondInput, -1.0), nil)
				requireError(t, second.MatMulRightTranspose(secondInput, weight, secondOutput), nil)
				requireError(t, weight.UploadBytes(wantBytes), ErrBusy)
				requireError(t, firstOutput.DownloadInto(firstResult[:]), ErrBusy)
				requireError(t, first.Submit(ctx), nil)
				requireError(t, second.Submit(ctx), nil)
				requireError(t, firstInput.Close(), nil)
				requireError(t, secondInput.Close(), nil)
				go func() { results <- first.Wait(ctx) }()
				go func() { results <- second.Wait(ctx) }()
				requireError(t, <-results, nil)
				requireError(t, <-results, nil)
				requireError(t, first.Close(), nil)
				requireError(t, second.Close(), nil)
				requireError(t, firstOutput.DownloadInto(firstResult[:]), nil)
				requireError(t, secondOutput.DownloadInto(secondResult[:]), nil)
				if math.Abs(float64(firstResult[0]-wantFirst[0])) > 1e-4 || math.Abs(float64(firstResult[1]-wantFirst[1])) > 1e-4 ||
					math.Abs(float64(secondResult[0]-wantSecond[0])) > 1e-4 || math.Abs(float64(secondResult[1]-wantSecond[1])) > 1e-4 {
					t.Fatalf("iteration %d: results = %v and %v, want %v and %v", iter, firstResult, secondResult, wantFirst, wantSecond)
				}
			}

			// Cancellation at Submit discards cleanly and leaves weight readable.
			var (
				canceledCtx, cancel = context.WithCancel(ctx)
				cancelScope         = requireScope(t, gpu)
				cancelInput         = requireTensor(t, gpu, 1, 32)
			)
			cancel()
			requireError(t, cancelScope.Fill(cancelInput, 1.0), nil)
			requireError(t, cancelScope.MatMulRightTranspose(cancelInput, weight, firstOutput), nil)
			if err := cancelScope.Submit(canceledCtx); !errors.Is(err, context.Canceled) {
				t.Fatalf("Submit(canceled) = %v, want context.Canceled", err)
			}
			requireError(t, cancelScope.Close(), nil)
			requireError(t, cancelInput.Close(), nil)
		})
	}
	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("shared quantized matmul leaked native resources: before %+v; after %+v", initial, after)
	}
}

func Test_MetalTensorMatMulQuantTiledConcurrentStress(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		ctx     = context.Background()
		rows    = int64(33)
		inner   = int64(128)
		cols    = int64(65)
		workers = 8
	)
	for _, dtype := range []DType{Q8_0, Q4_0, Q4_1} {
		t.Run(dtype.String(), func(t *testing.T) {
			var (
				random  = rand.New(rand.NewPCG(0x6e6e22, uint64(dtype)*19))
				leftVal = make([]float32, rows*inner)
				weightV = make([]float32, cols*inner)
			)
			for i := range leftVal {
				leftVal[i] = 2*random.Float32() - 1
			}
			for i := range weightV {
				weightV[i] = 2*random.Float32() - 1
			}
			var (
				dequant = dequantizeWeightsSlice(t, weightV, dtype)
				weight  = requireTensorWithDType(t, gpu, dtype, cols, inner)
			)
			requireError(t, weight.UploadFloat32As(weightV), nil)

			var (
				wg   sync.WaitGroup
				errs = make(chan error, workers)
			)
			for worker := range workers {
				wg.Add(1)
				go func(workerID int) {
					defer wg.Done()
					var (
						left, errL = gpu.NewFloat32([]int64{rows, inner})
						out, errO  = gpu.NewFloat32([]int64{rows, cols})
					)
					if errL != nil {
						errs <- errL
						return
					}
					defer left.Close()
					if errO != nil {
						errs <- errO
						return
					}
					defer out.Close()
					if err := left.Upload(leftVal); err != nil {
						errs <- err
						return
					}
					for iter := range 4 {
						var scope, errS = gpu.NewScope()
						if errS != nil {
							errs <- errS
							return
						}
						if err := scope.MatMulRightTranspose(left, weight, out); err != nil {
							_ = scope.Close()
							errs <- err
							return
						}
						if err := scope.Submit(ctx); err != nil {
							_ = scope.Close()
							errs <- err
							return
						}
						if err := scope.Wait(ctx); err != nil {
							_ = scope.Close()
							errs <- err
							return
						}
						if err := scope.Close(); err != nil {
							errs <- err
							return
						}
						if iter == 3 {
							var got = make([]float32, rows*cols)
							if err := out.DownloadInto(got); err != nil {
								errs <- err
								return
							}
							for row := range rows {
								for col := range cols {
									var want float64
									for k := range inner {
										want += float64(leftVal[row*inner+k]) * float64(dequant[col*inner+k])
									}
									var diff = math.Abs(float64(got[row*cols+col]) - want)
									if diff > 1e-3 {
										errs <- fmt.Errorf("worker %d (%d,%d): got %g, want %g (diff %g)", workerID, row, col, got[row*cols+col], want, diff)
										return
									}
								}
							}
						}
					}
				}(worker)
			}
			wg.Wait()
			close(errs)
			for err := range errs {
				t.Fatal(err)
			}
		})
	}
	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("concurrent tiled quantized matmul leaked native resources: before %+v; after %+v", initial, after)
	}
}

func Test_MetalFFNQuantized(t *testing.T) {
	var (
		gpu          = requireMetalDevice(t)
		native       = gpu.state.backend.(metalBackend).runtime
		initial      = native.ResourceSnapshot()
		weightValues = ffnWeightValues()
	)
	var (
		sqnrM1     = make(map[DType]float64)
		fp32Inputs = make(map[int64][]float32, len(ffnRowCounts))
		fp32Refs   = make(map[int64][]float64, len(ffnRowCounts))
	)
	for _, rows := range ffnRowCounts {
		var input = ffnInputs(rows)
		fp32Inputs[rows] = input
		fp32Refs[rows] = ffnReference(input, weightValues, ffnHidden, ffnIntermediate)
	}
	for _, dtype := range []DType{Q8_0, Q4_0, Q4_1} {
		t.Run(dtype.String(), func(t *testing.T) {
			var (
				weights        = newFFNWeightsWithDType(t, gpu, dtype, ffnHidden, ffnIntermediate)
				dequantWeights [3][]float32
				totalWeightB   uint64
			)
			for i, w := range weights {
				requireError(t, w.UploadFloat32As(weightValues[i]), nil)
				totalWeightB += w.Bytes()
				dequantWeights[i] = dequantizeWeightsSlice(t, weightValues[i], dtype)
			}
			var wantWeightBytes = uint64(3*ffnHidden*ffnIntermediate/QuantBlockSize) * dtype.BlockBytes()
			if totalWeightB != wantWeightBytes {
				t.Fatalf("FFN %s weight bytes = %d, want %d", dtype, totalWeightB, wantWeightBytes)
			}

			for _, rows := range ffnRowCounts {
				t.Run(fmt.Sprintf("M%d", rows), func(t *testing.T) {
					var (
						input      = fp32Inputs[rows]
						wantDequan = ffnReference(input, dequantWeights, ffnHidden, ffnIntermediate)
						wantFP32   = fp32Refs[rows]
						scratch    = newFFNScratch(t, gpu, rows, ffnHidden, ffnIntermediate)
						output     = make([]float32, len(input))
						scope      = requireScope(t, gpu)
						ctx        = context.Background()
					)
					requireError(t, scratch.input.Upload(input), nil)
					requireError(t, scratch.encode(scope, weights), nil)
					requireError(t, scope.Submit(ctx), nil)
					requireError(t, scope.Wait(ctx), nil)
					requireError(t, scope.Close(), nil)
					requireError(t, scratch.output.DownloadInto(output), nil)
					checkFFN(t, output, wantDequan)
					logFFNError(t, output, wantDequan)
					var sqnr, maxQuantErr = computeSQNR(wantFP32, output)
					if rows == 1 {
						sqnrM1[dtype] = sqnr
					}
					t.Logf("FFN %s M=%d vs unquantized FP32: SQNR=%.2f dB, maxAbsErr=%g", dtype, rows, sqnr, maxQuantErr)
				})
			}
		})
	}
	if sqnrM1[Q4_1] <= sqnrM1[Q4_0] {
		t.Fatalf("expected Q4_1 FFN M=1 SQNR (%.2f dB) > Q4_0 (%.2f dB)", sqnrM1[Q4_1], sqnrM1[Q4_0])
	}
	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("quantized FFN leaked native resources: before %+v; after %+v", initial, after)
	}
}

func runMetalMatMulQuant(
	t *testing.T,
	gpu *Device,
	dtype DType,
	rows, inner, cols int64,
	values, weight []float32,
) (got []float32) {
	t.Helper()
	var (
		left        = requireTensor(t, gpu, rows, inner)
		right       = requireTensorWithDType(t, gpu, dtype, cols, inner)
		destination = requireTensor(t, gpu, rows, cols)
		scope       = requireScope(t, gpu)
		ctx         = context.Background()
		wantBytes   = uint64(cols*(inner/QuantBlockSize)) * dtype.BlockBytes()
	)
	if right.Bytes() != wantBytes {
		t.Fatalf("right bytes = %d, want %d", right.Bytes(), wantBytes)
	}
	requireError(t, left.Upload(values), nil)
	requireError(t, right.UploadFloat32As(weight), nil)
	requireError(t, scope.MatMulRightTranspose(left, right, destination), nil)
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)
	got = make([]float32, rows*cols)
	requireError(t, destination.DownloadInto(got), nil)
	return got
}

func runMetalMatMulQuantRaw(
	t *testing.T,
	gpu *Device,
	dtype DType,
	rows, inner, cols int64,
	values []float32,
	rawRight []byte,
) (got []float32) {
	t.Helper()
	var (
		left        = requireTensor(t, gpu, rows, inner)
		right       = requireTensorWithDType(t, gpu, dtype, cols, inner)
		destination = requireTensor(t, gpu, rows, cols)
		scope       = requireScope(t, gpu)
		ctx         = context.Background()
	)
	requireError(t, left.Upload(values), nil)
	requireError(t, right.UploadBytes(rawRight), nil)
	requireError(t, scope.MatMulRightTranspose(left, right, destination), nil)
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)
	got = make([]float32, rows*cols)
	requireError(t, destination.DownloadInto(got), nil)
	return got
}

func dequantizeWeightsSlice(t testing.TB, values []float32, dtype DType) (out []float32) {
	t.Helper()
	switch dtype {
	case Float32:
		return values
	case Float16, BFloat16:
		out = make([]float32, len(values))
		for i, v := range values {
			out[i] = roundTripDType(v, dtype)
		}
		return out
	case Q8_0:
		var encoded, err = QuantizeQ8_0(values)
		requireError(t, err, nil)
		out, err = DequantizeQ8_0(encoded)
		requireError(t, err, nil)
		return out
	case Q4_0:
		var encoded, err = QuantizeQ4_0(values)
		requireError(t, err, nil)
		out, err = DequantizeQ4_0(encoded)
		requireError(t, err, nil)
		return out
	case Q4_1:
		var encoded, err = QuantizeQ4_1(values)
		requireError(t, err, nil)
		out, err = DequantizeQ4_1(encoded)
		requireError(t, err, nil)
		return out
	default:
		t.Fatalf("unsupported dtype %v", dtype)
		return nil
	}
}

func computeSQNR(reference []float64, quantized []float32) (sqnrDB, maxAbs float64) {
	var (
		signalPower float64
		noisePower  float64
	)
	for i, ref := range reference {
		var diff = math.Abs(float64(quantized[i]) - ref)
		signalPower += ref * ref
		noisePower += diff * diff
		if diff > maxAbs {
			maxAbs = diff
		}
	}
	if noisePower == 0 {
		return math.Inf(1), 0
	}
	return 10 * math.Log10(signalPower/noisePower), maxAbs
}
