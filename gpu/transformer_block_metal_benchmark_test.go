//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// Benchmark_TransformerBlockMixedMetal benchmarks a complete single-scope
// Llama 3.2 1B-shaped transformer block (RMSNorm -> QKV -> RotatePairs ->
// AppendKV -> CausalAttention -> O -> Residual -> RMSNorm -> Gate/Up ->
// SwiGLU -> Down -> Residual) across FP32, FP16, and BF16 weights with FP32
// and FP16 KV caches.
func Benchmark_TransformerBlockMixedMetal(b *testing.B) {
	type mode struct {
		name        string
		weightDType DType
		kvDType     DType
	}
	var modes = []mode{
		{name: "FP32_KV32", weightDType: Float32, kvDType: Float32},
		{name: "FP16_KV32", weightDType: Float16, kvDType: Float32},
		{name: "FP16_KV16", weightDType: Float16, kvDType: Float16},
		{name: "BF16_KV32", weightDType: BFloat16, kvDType: Float32},
		{name: "BF16_KV16", weightDType: BFloat16, kvDType: Float16},
	}

	for _, shape := range [][2]int64{{1, 511}, {29, 0}} {
		for _, m := range modes {
			b.Run(fmt.Sprintf("M%d_H%d/%s", shape[0], shape[1], m.name), func(b *testing.B) {
				benchmarkTransformerBlockMetal(b, shape[0], shape[1], m.weightDType, m.kvDType)
			})
		}
	}
}

func benchmarkTransformerBlockMetal(b *testing.B, rows, position int64, weightDType, kvDType DType) {
	const (
		hidden   int64 = 2048
		heads    int64 = 32
		kvHeads  int64 = 8
		headDim  int64 = 64
		ffnDim   int64 = 8192
		capacity int64 = 512
	)
	var (
		kvDim      = kvHeads * headDim
		device     = requireMetalDevice(b)
		ctx        = context.Background()
		wNorm1     = requireTensorWithDType(b, device, weightDType, hidden)
		wq         = requireTensorWithDType(b, device, weightDType, hidden, hidden)
		wk         = requireTensorWithDType(b, device, weightDType, kvDim, hidden)
		wv         = requireTensorWithDType(b, device, weightDType, kvDim, hidden)
		wo         = requireTensorWithDType(b, device, weightDType, hidden, hidden)
		wNorm2     = requireTensorWithDType(b, device, weightDType, hidden)
		wg         = requireTensorWithDType(b, device, weightDType, ffnDim, hidden)
		wu         = requireTensorWithDType(b, device, weightDType, ffnDim, hidden)
		wd         = requireTensorWithDType(b, device, weightDType, hidden, ffnDim)
		cosT       = requireTensor(b, device, capacity, headDim/2)
		sinT       = requireTensor(b, device, capacity, headDim/2)
		cache, err = NewKVCacheWithDType(device, capacity, kvHeads, headDim, kvDType)
	)
	requireError(b, err, nil)
	b.Cleanup(func() { requireError(b, cache.Close(), nil) })

	var c, s = attentionCoefficients(capacity, headDim)
	requireError(b, wNorm1.UploadFloat32As(attentionValues(hidden, 301, 1)), nil)
	requireError(b, wq.UploadFloat32As(attentionValues(hidden*hidden, 302, 0.05)), nil)
	requireError(b, wk.UploadFloat32As(attentionValues(kvDim*hidden, 303, 0.05)), nil)
	requireError(b, wv.UploadFloat32As(attentionValues(kvDim*hidden, 304, 0.05)), nil)
	requireError(b, wo.UploadFloat32As(attentionValues(hidden*hidden, 305, 0.05)), nil)
	requireError(b, wNorm2.UploadFloat32As(attentionValues(hidden, 306, 1)), nil)
	requireError(b, wg.UploadFloat32As(attentionValues(ffnDim*hidden, 307, 0.05)), nil)
	requireError(b, wu.UploadFloat32As(attentionValues(ffnDim*hidden, 308, 0.05)), nil)
	requireError(b, wd.UploadFloat32As(attentionValues(hidden*ffnDim, 309, 0.05)), nil)
	requireError(b, cosT.Upload(c), nil)
	requireError(b, sinT.Upload(s), nil)

	if position > 0 {
		var (
			kSeed    = requireTensor(b, device, position, kvHeads, headDim)
			vSeed    = requireTensor(b, device, position, kvHeads, headDim)
			seedS, e = device.NewScope()
		)
		requireError(b, e, nil)
		requireError(b, kSeed.Upload(attentionValues(position*kvDim, 310, 0.5)), nil)
		requireError(b, vSeed.Upload(attentionValues(position*kvDim, 311, 0.5)), nil)
		var upd = stageKV(b, seedS, cache, kSeed, vSeed, 0)
		completeAttentionScope(b, seedS)
		requireError(b, upd.Publish(), nil)
		requireError(b, kSeed.Close(), nil)
		requireError(b, vSeed.Close(), nil)
	}

	var (
		x      = requireTensor(b, device, rows, hidden)
		n1     = requireTensor(b, device, rows, hidden)
		q      = requireTensor(b, device, rows, hidden)
		k      = requireTensor(b, device, rows, kvDim)
		v      = requireTensor(b, device, rows, kvDim)
		qRot   = requireTensor(b, device, rows, hidden)
		kRot   = requireTensor(b, device, rows, kvDim)
		attn   = requireTensor(b, device, rows, hidden)
		oProj  = requireTensor(b, device, rows, hidden)
		res1   = requireTensor(b, device, rows, hidden)
		n2     = requireTensor(b, device, rows, hidden)
		gate   = requireTensor(b, device, rows, ffnDim)
		up     = requireTensor(b, device, rows, ffnDim)
		swiglu = requireTensor(b, device, rows, ffnDim)
		down   = requireTensor(b, device, rows, hidden)
		res2   = requireTensor(b, device, rows, hidden)
	)
	requireError(b, x.Upload(attentionValues(rows*hidden, 312, 0.5)), nil)

	runOnce := func() (elapsed time.Duration) {
		var (
			start    = time.Now()
			scope, e = device.NewScope()
		)
		requireError(b, e, nil)
		defer scope.Close()
		requireError(b, scope.RMSNorm(x, wNorm1, n1, attentionEpsilon), nil)
		requireError(b, scope.MatMulRightTranspose(n1, wq, q), nil)
		requireError(b, scope.MatMulRightTranspose(n1, wk, k), nil)
		requireError(b, scope.MatMulRightTranspose(n1, wv, v), nil)
		requireError(b, scope.RotatePairs(q, cosT, sinT, qRot, headDim, position, 0), nil)
		requireError(b, scope.RotatePairs(k, cosT, sinT, kRot, headDim, position, 0), nil)
		var update *KVAppend
		update, e = scope.AppendKV(cache, kRot, v, position)
		requireError(b, e, nil)
		defer update.Close()
		requireError(b, scope.CausalAttention(qRot, cache, attn, position), nil)
		requireError(b, scope.MatMulRightTranspose(attn, wo, oProj), nil)
		requireError(b, scope.Add(x, oProj, res1), nil)
		requireError(b, scope.RMSNorm(res1, wNorm2, n2, attentionEpsilon), nil)
		requireError(b, scope.MatMulRightTranspose(n2, wg, gate), nil)
		requireError(b, scope.MatMulRightTranspose(n2, wu, up), nil)
		requireError(b, scope.SwiGLU(gate, up, swiglu), nil)
		requireError(b, scope.MatMulRightTranspose(swiglu, wd, down), nil)
		requireError(b, scope.Add(res1, down, res2), nil)
		requireError(b, scope.Submit(ctx), nil)
		requireError(b, scope.Wait(ctx), nil)
		requireError(b, scope.Close(), nil)
		requireError(b, update.Close(), nil)
		return time.Since(start)
	}

	for range 2 {
		runOnce()
	}

	var total time.Duration
	b.ResetTimer()
	for b.Loop() {
		total += runOnce()
	}
	b.StopTimer()

	var (
		weightBytes = wNorm1.Bytes() + wq.Bytes() + wk.Bytes() + wv.Bytes() + wo.Bytes() +
			wNorm2.Bytes() + wg.Bytes() + wu.Bytes() + wd.Bytes()
		kvBytes = cache.state.slot.Bytes()
	)
	b.ReportMetric(float64(total.Nanoseconds())/float64(b.N)/1e6, "ms/op")
	b.ReportMetric(float64(weightBytes), "weight-B")
	b.ReportMetric(float64(kvBytes), "kv-B")
	b.ReportMetric(1, "commands/op")
}
