//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// Benchmark_AppendKVMetal measures 16-layer AppendKV latency (matching Llama 3.2
// 1B's 16 decoder layers in one scope submission) across decode context lengths
// H = 38, 512, 2048, and 4096 for both Float32 and Float16 KV caches.
func Benchmark_AppendKVMetal(b *testing.B) {
	var histories = []int64{38, 512, 2048, 4096}
	for _, history := range histories {
		for _, dtype := range []DType{Float32, Float16} {
			b.Run(fmt.Sprintf("M1_H%d_x16/%s", history, dtype), func(b *testing.B) {
				benchmarkAppendKVMetal(b, history, 16, dtype)
			})
		}
	}
}

func benchmarkAppendKVMetal(b *testing.B, position int64, layers int, kvDType DType) {
	const (
		kvHeads int64 = 8
		headDim int64 = 64
	)
	var (
		capacity = position + 1
		device   = requireMetalDevice(b)
		native   = device.state.backend.(metalBackend).runtime
		ctx      = context.Background()
		keys     = requireTensor(b, device, 1, kvHeads, headDim)
		values   = requireTensor(b, device, 1, kvHeads, headDim)
		beforeKV = native.ResourceSnapshot()
		caches   = make([]*KVCache, layers)
		err      error
	)
	for i := range caches {
		if caches[i], err = NewKVCacheWithDType(device, capacity, kvHeads, headDim, kvDType); err != nil {
			b.Fatalf("NewKVCacheWithDType layer %d: %v", i, err)
		}
		var cache = caches[i]
		b.Cleanup(func() { requireError(b, cache.Close(), nil) })
	}
	var (
		afterKV   = native.ResourceSnapshot()
		kvBytes   = afterKV.LiveBufferBytes - beforeKV.LiveBufferBytes
		kvBuffers = afterKV.LiveBuffers - beforeKV.LiveBuffers
	)

	requireError(b, keys.Upload(attentionValues(kvHeads*headDim, 601, 0.5)), nil)
	requireError(b, values.Upload(attentionValues(kvHeads*headDim, 602, 0.5)), nil)

	if position > 0 {
		var (
			kSeed     = requireTensor(b, device, position, kvHeads, headDim)
			vSeed     = requireTensor(b, device, position, kvHeads, headDim)
			seedScope *Scope
			updates   = make([]*KVAppend, layers)
		)
		requireError(b, kSeed.Upload(attentionValues(position*kvHeads*headDim, 603, 0.5)), nil)
		requireError(b, vSeed.Upload(attentionValues(position*kvHeads*headDim, 604, 0.5)), nil)
		seedScope, err = device.NewScope()
		requireError(b, err, nil)
		for i, cache := range caches {
			updates[i] = stageKV(b, seedScope, cache, kSeed, vSeed, 0)
		}
		completeAttentionScope(b, seedScope)
		for _, upd := range updates {
			requireError(b, upd.Publish(), nil)
		}
		requireError(b, kSeed.Close(), nil)
		requireError(b, vSeed.Close(), nil)
	}

	var updates = make([]*KVAppend, layers)
	runOnce := func() (elapsed time.Duration) {
		var (
			start    = time.Now()
			scope, e = device.NewScope()
		)
		requireError(b, e, nil)
		defer scope.Close()
		for i, cache := range caches {
			if updates[i], e = scope.AppendKV(cache, keys, values, position); e != nil {
				b.Fatalf("AppendKV layer %d: %v", i, e)
			}
		}
		requireError(b, scope.Submit(ctx), nil)
		requireError(b, scope.Wait(ctx), nil)
		requireError(b, scope.Close(), nil)
		for i, upd := range updates {
			requireError(b, upd.Close(), nil)
			updates[i] = nil
		}
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
	b.ReportMetric(float64(kvBytes), "kv-device-B")
	b.ReportMetric(float64(kvBuffers), "kv-buffers")
	b.ReportMetric(float64(layers), "dispatches/op")
}
