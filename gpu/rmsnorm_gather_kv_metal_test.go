//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"math"
	"slices"
	"testing"
)

func Test_MetalRMSNormParallelAndMixedWeights(t *testing.T) {
	var (
		device = requireMetalDevice(t)
		widths = []int64{1, 3, 6, 17, 32, 33, 64, 65, 128, 129, 256, 257, 2048, 8192}
		rows   = []int64{1, 6, 29}
		dtypes = []DType{Float32, Float16, BFloat16}
	)

	for _, w := range widths {
		for _, m := range rows {
			if w == 8192 && m > 6 {
				continue
			}
			for _, dtype := range dtypes {
				t.Run(fmt.Sprintf("M%d_C%d_%s", m, w, dtype), func(t *testing.T) {
					var (
						input      = requireTensor(t, device, m, w)
						output     = requireTensor(t, device, m, w)
						weight     = requireTensorWithDType(t, device, dtype, w)
						inputData  = attentionValues(m*w, uint64(w*31+m), 1)
						weightData = attentionValues(w, uint64(w*37+m), 1)
						scope      = requireScope(t, device)
					)
					requireError(t, input.Upload(inputData), nil)
					requireError(t, weight.UploadFloat32As(weightData), nil)
					requireError(t, scope.RMSNorm(input, weight, output, attentionEpsilon), nil)
					completeAttentionScope(t, scope)

					var got = make([]float32, m*w)
					requireError(t, output.DownloadInto(got), nil)

					var effWeight = dequantizeForRef(weightData, dtype)
					var want = referenceNorm(inputData, effWeight)
					var p50, p95, p99, maxErr = fp64ErrorStats(got, want)
					if maxErr > 1e-5 {
						t.Fatalf("M=%d C=%d %s max abs error = %g (p50=%g p95=%g p99=%g)", m, w, dtype, maxErr, p50, p95, p99)
					}
					if w == 2048 && (m == 1 || m == 29) {
						t.Logf("RMSNorm M=%d C=%d %s FP64 abs err: p50=%g p95=%g p99=%g max=%g", m, w, dtype, p50, p95, p99, maxErr)
					}
				})
			}
		}
	}

	// Special values: zeros, MaxFloat32 overflow guard, NaN, and +Inf.
	t.Run("special_values", func(t *testing.T) {
		var (
			input  = requireTensor(t, device, 4, 8)
			output = requireTensor(t, device, 4, 8)
			weight = requireTensorWithDType(t, device, Float16, 8)
			data   = make([]float32, 32)
			wData  = []float32{1, -1, 0.5, 2, -0.5, 1.5, 0.25, -2}
			scope  = requireScope(t, device)
		)
		// Row 0: all zeros.
		// Row 1: MaxFloat32 / -MaxFloat32 overflow guard.
		data[8], data[9], data[10] = math.MaxFloat32, -math.MaxFloat32, 1e30
		// Row 2: contains NaN.
		data[16] = 1
		data[17] = float32(math.NaN())
		// Row 3: contains +Inf.
		data[24] = float32(math.Inf(1))
		data[25] = 2

		requireError(t, input.Upload(data), nil)
		requireError(t, weight.UploadFloat32As(wData), nil)
		requireError(t, scope.RMSNorm(input, weight, output, attentionEpsilon), nil)
		completeAttentionScope(t, scope)

		var got = make([]float32, 32)
		requireError(t, output.DownloadInto(got), nil)
		var effW = dequantizeForRef(wData, Float16)
		// Row 0: all zeros.
		for i := range 8 {
			if got[i] != 0 {
				t.Fatalf("zero row[%d] = %v, want 0", i, got[i])
			}
		}
		// Row 1: finite normalized values matching FP64 reference.
		var wantRow1 = referenceNorm(data[8:16], effW)
		compareAttention(t, got[8:16], wantRow1, 1e-6)
		// Row 2: NaN propagates across the row.
		for i := 16; i < 24; i++ {
			if !math.IsNaN(float64(got[i])) {
				t.Fatalf("NaN row[%d] = %v, want NaN", i-16, got[i])
			}
		}
		// Row 3: +Inf produces NaN across the row.
		for i := 24; i < 32; i++ {
			if !math.IsNaN(float64(got[i])) {
				t.Fatalf("+Inf row[%d] = %v, want NaN", i-24, got[i])
			}
		}
	})
}

func Test_MetalGatherRowsReducedPrecision(t *testing.T) {
	var device = requireMetalDevice(t)

	for _, dtype := range []DType{Float16, BFloat16} {
		t.Run(fmt.Sprintf("%s_same_and_mixed_odd", dtype), func(t *testing.T) {
			for _, cols := range []int64{1, 3, 5, 17, 2048} {
				var (
					srcRows  int64 = 7
					ids            = []int64{6, 0, 6, 3, 1}
					src            = requireTensorWithDType(t, device, dtype, srcRows, cols)
					sameDst        = requireTensorWithDType(t, device, dtype, int64(len(ids)), cols)
					mixedDst       = requireTensor(t, device, int64(len(ids)), cols)
					rawBits        = make([]uint16, srcRows*cols)
					scope          = requireScope(t, device)
				)
				for i := range rawBits {
					rawBits[i] = uint16((i*37 + 13) & 0x7bff)
				}
				if cols >= 5 {
					// Include special bit patterns: +0, -0, subnormal, -1, +Inf.
					rawBits[0] = 0x0000
					rawBits[1] = 0x8000
					rawBits[2] = 0x0001
					if dtype == Float16 {
						rawBits[3] = 0xbc00
						rawBits[4] = 0x7c00
					} else {
						rawBits[3] = 0xbf80
						rawBits[4] = 0x7f80
					}
				}
				requireError(t, src.UploadBits(rawBits), nil)
				requireError(t, scope.GatherRows(src, ids, sameDst), nil)
				requireError(t, scope.GatherRows(src, ids, mixedDst), nil)
				completeAttentionScope(t, scope)

				var gotBits = make([]uint16, int64(len(ids))*cols)
				requireError(t, sameDst.DownloadBitsInto(gotBits), nil)
				var gotF32 = make([]float32, int64(len(ids))*cols)
				requireError(t, mixedDst.DownloadInto(gotF32), nil)

				for outRow, srcRow := range ids {
					for col := range cols {
						var wantBit = rawBits[srcRow*cols+col]
						var gotBit = gotBits[int64(outRow)*cols+col]
						if gotBit != wantBit {
							t.Fatalf("%s cols=%d same-dtype [%d,%d]: got 0x%04x, want 0x%04x", dtype, cols, outRow, col, gotBit, wantBit)
						}
						var wantF32 float32
						if dtype == Float16 {
							wantF32 = Float16ToFloat32(wantBit)
						} else {
							wantF32 = BFloat16ToFloat32(wantBit)
						}
						var actual = gotF32[int64(outRow)*cols+col]
						if math.Float32bits(actual) != math.Float32bits(wantF32) {
							t.Fatalf("%s cols=%d mixed [%d,%d]: got %v (0x%08x), want %v (0x%08x)",
								dtype, cols, outRow, col, actual, math.Float32bits(actual), wantF32, math.Float32bits(wantF32))
						}
					}
				}
			}
		})

		// >512 rows exercises multi-chunk setBytes dispatch in nn_metal_scope_encode_gather_rows_mixed.
		t.Run(fmt.Sprintf("%s_multichunk_600_rows", dtype), func(t *testing.T) {
			const (
				srcRows int64 = 16
				cols    int64 = 7
				outRows int64 = 600
			)
			var (
				src     = requireTensorWithDType(t, device, dtype, srcRows, cols)
				dst     = requireTensor(t, device, outRows, cols)
				srcData = attentionValues(srcRows*cols, 555, 2)
				ids     = make([]int64, outRows)
				scope   = requireScope(t, device)
			)
			for i := range ids {
				ids[i] = int64((i * 7) % int(srcRows))
			}
			requireError(t, src.UploadFloat32As(srcData), nil)
			requireError(t, scope.GatherRows(src, ids, dst), nil)
			completeAttentionScope(t, scope)

			var (
				got    = make([]float32, outRows*cols)
				effSrc = dequantizeForRef(srcData, dtype)
			)
			requireError(t, dst.DownloadInto(got), nil)
			for outRow, srcRow := range ids {
				if !slices.Equal(got[int64(outRow)*cols:int64(outRow+1)*cols], effSrc[srcRow*cols:(srcRow+1)*cols]) {
					t.Fatalf("%s multichunk row %d mismatch", dtype, outRow)
				}
			}
		})

		// Tied [128256, 2048] embedding/classifier in one halved allocation.
		t.Run(fmt.Sprintf("%s_tied_llama_128256x2048", dtype), func(t *testing.T) {
			const vocab, width int64 = 128256, 2048
			var (
				native = device.state.backend.(metalBackend).runtime
				before = native.ResourceSnapshot()
			)
			t.Run("exec", func(t *testing.T) {
				var (
					weight = requireTensorWithDType(t, device, dtype, vocab, width)
					embed  = requireTensor(t, device, 1, width)
					logits = requireTensor(t, device, 1, vocab)
					scope  = requireScope(t, device)
					raw    = make([]uint16, vocab*width)
				)
				if weight.Bytes() != uint64(vocab*width*2) {
					t.Fatalf("weight bytes = %d, want %d", weight.Bytes(), vocab*width*2)
				}
				var oneBit uint16 = Float32ToFloat16(1.0)
				if dtype == BFloat16 {
					oneBit = Float32ToBFloat16(1.0)
				}
				for col := range width {
					raw[42*width+col] = oneBit
					raw[128000*width+col] = oneBit
				}
				requireError(t, weight.UploadBits(raw), nil)
				requireError(t, scope.GatherRows(weight, []int64{42}, embed), nil)
				requireError(t, scope.MatMulRightTranspose(embed, weight, logits), nil)
				completeAttentionScope(t, scope)

				var gotEmbed = make([]float32, width)
				requireError(t, embed.DownloadInto(gotEmbed), nil)
				for col := range width {
					if gotEmbed[col] != 1.0 {
						t.Fatalf("embed[%d] = %v, want 1", col, gotEmbed[col])
					}
				}
				var gotLogits = make([]float32, vocab)
				requireError(t, logits.DownloadInto(gotLogits), nil)
				if gotLogits[42] != float32(width) || gotLogits[128000] != float32(width) || gotLogits[0] != 0 {
					t.Fatalf("tied logits: [0]=%v [42]=%v [128000]=%v", gotLogits[0], gotLogits[42], gotLogits[128000])
				}
			})
			var after = native.ResourceSnapshot()
			if after.LiveBuffers != before.LiveBuffers || after.LiveBufferBytes != before.LiveBufferBytes ||
				after.CreatedBuffers-before.CreatedBuffers != 3 || after.SubmittedCommands-before.SubmittedCommands != 1 {
				t.Fatalf("unexpected native resources: before %+v after %+v", before, after)
			}
		})
	}
}

func Test_MetalKVCacheFloat16AndAttention(t *testing.T) {
	var device = requireMetalDevice(t)
	var cases = []struct {
		name                   string
		rows, heads, kv, width int64
	}{
		{"MHA_27rows", 27, 3, 3, 6},
		{"GQA_28rows", 28, 6, 3, 10},
		{"MQA_29rows", 29, 5, 1, 8},
		{"Llama_32to8_6rows", 6, 32, 8, 64},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var (
				native   = device.state.backend.(metalBackend).runtime
				baseline = native.ResourceSnapshot()
			)
			t.Run("prefill_and_decode", func(t *testing.T) {
				var (
					capRows  = tc.rows + 1
					cache, _ = NewKVCacheWithDType(device, capRows, tc.kv, tc.width, Float16)
					qPrefill = requireTensor(t, device, tc.rows, tc.heads, tc.width)
					kPrefill = requireTensor(t, device, tc.rows, tc.kv, tc.width)
					vPrefill = requireTensor(t, device, tc.rows, tc.kv, tc.width)
					oPrefill = requireTensor(t, device, tc.rows, tc.heads, tc.width)
					qDecode  = requireTensor(t, device, 1, tc.heads, tc.width)
					kDecode  = requireTensor(t, device, 1, tc.kv, tc.width)
					vDecode  = requireTensor(t, device, 1, tc.kv, tc.width)
					oDecode  = requireTensor(t, device, 1, tc.heads, tc.width)
					qPreData = attentionValues(tc.rows*tc.heads*tc.width, 11, 0.5)
					kPreData = attentionValues(tc.rows*tc.kv*tc.width, 12, 0.5)
					vPreData = attentionValues(tc.rows*tc.kv*tc.width, 13, 0.5)
					qDecData = attentionValues(tc.heads*tc.width, 14, 0.5)
					kDecData = attentionValues(tc.kv*tc.width, 15, 0.5)
					vDecData = attentionValues(tc.kv*tc.width, 16, 0.5)
				)
				t.Cleanup(func() { requireError(t, cache.Close(), nil) })

				// Verify exact 50% KV slot byte footprint.
				var wantSlotBytes = uint64(2 * capRows * tc.kv * tc.width * 2)
				if cache.state.slots[0].Bytes() != wantSlotBytes || cache.state.slots[1].Bytes() != wantSlotBytes {
					t.Fatalf("FP16 KV slot bytes = %d, want %d", cache.state.slots[0].Bytes(), wantSlotBytes)
				}

				// Empty cache yields zeros.
				var emptyScope = requireScope(t, device)
				requireError(t, qDecode.Upload(qDecData), nil)
				requireError(t, emptyScope.CausalAttention(qDecode, cache, oDecode, 0), nil)
				completeAttentionScope(t, emptyScope)
				var zeroOut = make([]float32, len(qDecData))
				requireError(t, oDecode.DownloadInto(zeroOut), nil)
				compareAttention(t, zeroOut, make([]float64, len(zeroOut)), 0)

				// Prefill step.
				requireError(t, qPrefill.Upload(qPreData), nil)
				requireError(t, kPrefill.Upload(kPreData), nil)
				requireError(t, vPrefill.Upload(vPreData), nil)
				var preScope = requireScope(t, device)
				var preUpdate = stageKV(t, preScope, cache, kPrefill, vPrefill, 0)
				requireError(t, preScope.CausalAttention(qPrefill, cache, oPrefill, 0), nil)
				completeAttentionScope(t, preScope)
				var gotPre = make([]float32, len(qPreData))
				requireError(t, oPrefill.DownloadInto(gotPre), nil)

				var (
					kPreF16 = dequantizeForRef(kPreData, Float16)
					vPreF16 = dequantizeForRef(vPreData, Float16)
					wantPre = referenceAttention(testFP64(qPreData), testFP64(kPreF16), testFP64(vPreF16), tc.heads, tc.kv, tc.width, 0)
				)
				compareAttention(t, gotPre, wantPre, 1e-5)
				if cache.Len() != 0 {
					t.Fatal("completion published FP16 KV before Publish")
				}
				requireError(t, preUpdate.Publish(), nil)
				if cache.Len() != tc.rows {
					t.Fatalf("published FP16 KV len = %d, want %d", cache.Len(), tc.rows)
				}

				// Decode step.
				requireError(t, kDecode.Upload(kDecData), nil)
				requireError(t, vDecode.Upload(vDecData), nil)
				var decScope = requireScope(t, device)
				var decUpdate = stageKV(t, decScope, cache, kDecode, vDecode, tc.rows)
				requireError(t, decScope.CausalAttention(qDecode, cache, oDecode, tc.rows), nil)
				completeAttentionScope(t, decScope)
				var gotDec = make([]float32, len(qDecData))
				requireError(t, oDecode.DownloadInto(gotDec), nil)

				var (
					kAll    = slices.Concat(kPreF16, dequantizeForRef(kDecData, Float16))
					vAll    = slices.Concat(vPreF16, dequantizeForRef(vDecData, Float16))
					wantDec = referenceAttention(testFP64(qDecData), testFP64(kAll), testFP64(vAll), tc.heads, tc.kv, tc.width, tc.rows)
				)
				compareAttention(t, gotDec, wantDec, 1e-5)
				requireError(t, decUpdate.Publish(), nil)
			})
			var after = native.ResourceSnapshot()
			if after.LiveBuffers != baseline.LiveBuffers || after.LiveBufferBytes != baseline.LiveBufferBytes || after.LiveScopes != baseline.LiveScopes {
				t.Fatalf("FP16 KV test leaked resources: before %+v after %+v", baseline, after)
			}
		})
	}
}

func Test_MetalTransformerBlockMixedPrecision(t *testing.T) {
	const (
		hidden   int64 = 128
		heads    int64 = 4
		kvHeads  int64 = 2
		headDim  int64 = 32
		ffnDim   int64 = 256
		vocab    int64 = 64
		capacity int64 = 16
	)
	type mode struct {
		name        string
		weightDType DType
		kvDType     DType
	}
	var modes = []mode{
		{name: "FP32_weights_FP32_KV", weightDType: Float32, kvDType: Float32},
		{name: "FP16_weights_FP32_KV", weightDType: Float16, kvDType: Float32},
		{name: "FP16_weights_FP16_KV", weightDType: Float16, kvDType: Float16},
		{name: "BF16_weights_FP32_KV", weightDType: BFloat16, kvDType: Float32},
		{name: "BF16_weights_FP16_KV", weightDType: BFloat16, kvDType: Float16},
	}

	var (
		device = requireMetalDevice(t)
		ctx    = context.Background()
		kvDim  = kvHeads * headDim
		c, s   = attentionCoefficients(capacity, headDim)
		embW   = attentionValues(vocab*hidden, 201, 0.25)
		norm1W = attentionValues(hidden, 202, 1.0)
		wqData = attentionValues(hidden*hidden, 203, 0.1)
		wkData = attentionValues(kvDim*hidden, 204, 0.1)
		wvData = attentionValues(kvDim*hidden, 205, 0.1)
		woData = attentionValues(hidden*hidden, 206, 0.1)
		norm2W = attentionValues(hidden, 207, 1.0)
		wgData = attentionValues(ffnDim*hidden, 208, 0.1)
		wuData = attentionValues(ffnDim*hidden, 209, 0.1)
		wdData = attentionValues(hidden*ffnDim, 210, 0.1)
		prompt = []int64{3, 17, 42, 7, 19, 55}
		nextID = []int64{11}
	)

	for _, m := range modes {
		t.Run(m.name, func(t *testing.T) {
			var (
				native   = device.state.backend.(metalBackend).runtime
				baseline = native.ResourceSnapshot()
			)
			t.Run("prefill_and_decode", func(t *testing.T) {
				var (
					wEmb     = requireTensorWithDType(t, device, m.weightDType, vocab, hidden)
					wNorm1   = requireTensorWithDType(t, device, m.weightDType, hidden)
					wq       = requireTensorWithDType(t, device, m.weightDType, hidden, hidden)
					wk       = requireTensorWithDType(t, device, m.weightDType, kvDim, hidden)
					wv       = requireTensorWithDType(t, device, m.weightDType, kvDim, hidden)
					wo       = requireTensorWithDType(t, device, m.weightDType, hidden, hidden)
					wNorm2   = requireTensorWithDType(t, device, m.weightDType, hidden)
					wg       = requireTensorWithDType(t, device, m.weightDType, ffnDim, hidden)
					wu       = requireTensorWithDType(t, device, m.weightDType, ffnDim, hidden)
					wd       = requireTensorWithDType(t, device, m.weightDType, hidden, ffnDim)
					cosT     = requireTensor(t, device, capacity, headDim/2)
					sinT     = requireTensor(t, device, capacity, headDim/2)
					cache, _ = NewKVCacheWithDType(device, capacity, kvHeads, headDim, m.kvDType)
				)
				t.Cleanup(func() { requireError(t, cache.Close(), nil) })

				requireError(t, wEmb.UploadFloat32As(embW), nil)
				requireError(t, wNorm1.UploadFloat32As(norm1W), nil)
				requireError(t, wq.UploadFloat32As(wqData), nil)
				requireError(t, wk.UploadFloat32As(wkData), nil)
				requireError(t, wv.UploadFloat32As(wvData), nil)
				requireError(t, wo.UploadFloat32As(woData), nil)
				requireError(t, wNorm2.UploadFloat32As(norm2W), nil)
				requireError(t, wg.UploadFloat32As(wgData), nil)
				requireError(t, wu.UploadFloat32As(wuData), nil)
				requireError(t, wd.UploadFloat32As(wdData), nil)
				requireError(t, cosT.Upload(c), nil)
				requireError(t, sinT.Upload(s), nil)

				// Effective weights dequantized to FP32 for exact FP64 reference.
				var (
					effEmb  = dequantizeForRef(embW, m.weightDType)
					effN1   = dequantizeForRef(norm1W, m.weightDType)
					effQ    = dequantizeForRef(wqData, m.weightDType)
					effK    = dequantizeForRef(wkData, m.weightDType)
					effV    = dequantizeForRef(wvData, m.weightDType)
					effO    = dequantizeForRef(woData, m.weightDType)
					effN2   = dequantizeForRef(norm2W, m.weightDType)
					effG    = dequantizeForRef(wgData, m.weightDType)
					effU    = dequantizeForRef(wuData, m.weightDType)
					effD    = dequantizeForRef(wdData, m.weightDType)
					refKeys []float64
					refVals []float64
				)

				runStep := func(ids []int64, pos int64) (gotLogits []float32, wantLogits []float64) {
					var (
						rows   = int64(len(ids))
						x      = requireTensor(t, device, rows, hidden)
						n1     = requireTensor(t, device, rows, hidden)
						q      = requireTensor(t, device, rows, hidden)
						k      = requireTensor(t, device, rows, kvDim)
						v      = requireTensor(t, device, rows, kvDim)
						qRot   = requireTensor(t, device, rows, hidden)
						kRot   = requireTensor(t, device, rows, kvDim)
						attn   = requireTensor(t, device, rows, hidden)
						oProj  = requireTensor(t, device, rows, hidden)
						res1   = requireTensor(t, device, rows, hidden)
						n2     = requireTensor(t, device, rows, hidden)
						gate   = requireTensor(t, device, rows, ffnDim)
						up     = requireTensor(t, device, rows, ffnDim)
						swiglu = requireTensor(t, device, rows, ffnDim)
						down   = requireTensor(t, device, rows, hidden)
						res2   = requireTensor(t, device, rows, hidden)
						logits = requireTensor(t, device, rows, vocab)
						scope  = requireScope(t, device)
					)
					var beforeCmd = native.ResourceSnapshot().SubmittedCommands
					requireError(t, scope.GatherRows(wEmb, ids, x), nil)
					requireError(t, scope.RMSNorm(x, wNorm1, n1, attentionEpsilon), nil)
					requireError(t, scope.MatMulRightTranspose(n1, wq, q), nil)
					requireError(t, scope.MatMulRightTranspose(n1, wk, k), nil)
					requireError(t, scope.MatMulRightTranspose(n1, wv, v), nil)
					requireError(t, scope.RotatePairs(q, cosT, sinT, qRot, headDim, pos, 0), nil)
					requireError(t, scope.RotatePairs(k, cosT, sinT, kRot, headDim, pos, 0), nil)
					var update = stageKV(t, scope, cache, kRot, v, pos)
					requireError(t, scope.CausalAttention(qRot, cache, attn, pos), nil)
					requireError(t, scope.MatMulRightTranspose(attn, wo, oProj), nil)
					requireError(t, scope.Add(x, oProj, res1), nil)
					requireError(t, scope.RMSNorm(res1, wNorm2, n2, attentionEpsilon), nil)
					requireError(t, scope.MatMulRightTranspose(n2, wg, gate), nil)
					requireError(t, scope.MatMulRightTranspose(n2, wu, up), nil)
					requireError(t, scope.SwiGLU(gate, up, swiglu), nil)
					requireError(t, scope.MatMulRightTranspose(swiglu, wd, down), nil)
					requireError(t, scope.Add(res1, down, res2), nil)
					requireError(t, scope.MatMulRightTranspose(res2, wEmb, logits), nil)
					requireError(t, scope.Submit(ctx), nil)
					requireError(t, scope.Wait(ctx), nil)
					requireError(t, scope.Close(), nil)
					requireError(t, update.Publish(), nil)

					var afterCmd = native.ResourceSnapshot().SubmittedCommands
					if afterCmd != beforeCmd+1 {
						t.Fatalf("expected 1 command submission, got %d", afterCmd-beforeCmd)
					}

					gotLogits = make([]float32, rows*vocab)
					requireError(t, logits.DownloadInto(gotLogits), nil)

					// Independent FP64 reference.
					var xRef = make([]float32, rows*hidden)
					for r, id := range ids {
						copy(xRef[int64(r)*hidden:int64(r+1)*hidden], effEmb[id*hidden:(id+1)*hidden])
					}
					var (
						n1Ref = referenceNorm(xRef, effN1)
						qRef  = referenceProjection(n1Ref, effQ, hidden)
						kRef  = referenceProjection(n1Ref, effK, hidden)
						vRef  = referenceProjection(n1Ref, effV, hidden)
					)
					referenceRotate(qRef, c, s, pos, heads, headDim)
					referenceRotate(kRef, c, s, pos, kvHeads, headDim)
					if m.kvDType == Float16 {
						for i, val := range kRef {
							kRef[i] = float64(Float16ToFloat32(Float32ToFloat16(float32(val))))
						}
						for i, val := range vRef {
							vRef[i] = float64(Float16ToFloat32(Float32ToFloat16(float32(val))))
						}
					}
					refKeys = append(refKeys, kRef...)
					refVals = append(refVals, vRef...)
					var (
						attnRef = referenceAttention(qRef, refKeys, refVals, heads, kvHeads, headDim, pos)
						oRef    = referenceProjection(attnRef, effO, hidden)
						res1Ref = make([]float32, rows*hidden)
					)
					for i := range res1Ref {
						res1Ref[i] = float32(float64(xRef[i]) + oRef[i])
					}
					var (
						n2Ref  = referenceNorm(res1Ref, effN2)
						gRef   = referenceProjection(n2Ref, effG, hidden)
						uRef   = referenceProjection(n2Ref, effU, hidden)
						swiRef = make([]float64, len(gRef))
					)
					for i, gv := range gRef {
						swiRef[i] = (gv / (1 + math.Exp(-gv))) * uRef[i]
					}
					var (
						dRef    = referenceProjection(swiRef, effD, ffnDim)
						res2Ref = make([]float64, rows*hidden)
					)
					for i := range res2Ref {
						res2Ref[i] = float64(res1Ref[i]) + dRef[i]
					}
					wantLogits = referenceProjection(res2Ref, effEmb, hidden)
					return gotLogits, wantLogits
				}

				var preGot, preWant = runStep(prompt, 0)
				var p50, p95, p99, maxErr = fp64ErrorStats(preGot, preWant)
				if maxErr > 1e-4 {
					t.Fatalf("%s prefill max err = %g (p50=%g p95=%g p99=%g)", m.name, maxErr, p50, p95, p99)
				}
				t.Logf("%s prefill (M=%d) FP64 abs err: p50=%g p95=%g p99=%g max=%g", m.name, len(prompt), p50, p95, p99, maxErr)

				var decGot, decWant = runStep(nextID, int64(len(prompt)))
				p50, p95, p99, maxErr = fp64ErrorStats(decGot, decWant)
				if maxErr > 1e-4 {
					t.Fatalf("%s decode max err = %g (p50=%g p95=%g p99=%g)", m.name, maxErr, p50, p95, p99)
				}
				t.Logf("%s decode  (M=1) FP64 abs err: p50=%g p95=%g p99=%g max=%g", m.name, p50, p95, p99, maxErr)
			})
			var after = native.ResourceSnapshot()
			if after.LiveBuffers != baseline.LiveBuffers || after.LiveBufferBytes != baseline.LiveBufferBytes || after.LiveScopes != baseline.LiveScopes {
				t.Fatalf("%s leaked resources: before %+v after %+v", m.name, baseline, after)
			}
		})
	}
}

func dequantizeForRef(values []float32, dtype DType) (out []float32) {
	out = make([]float32, len(values))
	switch dtype {
	case Float16:
		for i, v := range values {
			out[i] = Float16ToFloat32(Float32ToFloat16(v))
		}
	case BFloat16:
		for i, v := range values {
			out[i] = BFloat16ToFloat32(Float32ToBFloat16(v))
		}
	default:
		copy(out, values)
	}
	return out
}

func fp64ErrorStats(got []float32, want []float64) (p50, p95, p99, maxErr float64) {
	var errs = make([]float64, len(got))
	for i := range got {
		var d = math.Abs(float64(got[i]) - want[i])
		errs[i] = d
		if d > maxErr {
			maxErr = d
		}
	}
	slices.Sort(errs)
	if len(errs) == 0 {
		return 0, 0, 0, 0
	}
	p50 = errs[(len(errs)-1)*50/100]
	p95 = errs[(len(errs)-1)*95/100]
	p99 = errs[(len(errs)-1)*99/100]
	return p50, p95, p99, maxErr
}
