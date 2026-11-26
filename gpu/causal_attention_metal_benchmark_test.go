//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// Benchmark_CausalAttentionMetal measures 16-dispatch CausalAttention latency
// across Llama 3.2 1B decode context lengths (M=1, H=38, 128, 512, 1024) and
// prefill (M=29, H=29) for both Float32 and Float16 KV caches.
func Benchmark_CausalAttentionMetal(b *testing.B) {
	type shape struct {
		name       string
		rows       int64
		history    int64
		position   int64
		dispatches int
	}

	var shapes = []shape{
		{name: "M1_H38_x16", rows: 1, history: 38, position: 37, dispatches: 16},
		{name: "M1_H128_x16", rows: 1, history: 128, position: 127, dispatches: 16},
		{name: "M1_H512_x16", rows: 1, history: 512, position: 511, dispatches: 16},
		{name: "M1_H1024_x16", rows: 1, history: 1024, position: 1023, dispatches: 16},
		{name: "M29_H29_x16", rows: 29, history: 29, position: 0, dispatches: 16},
	}

	for _, tc := range shapes {
		for _, dtype := range []DType{Float32, Float16} {
			b.Run(fmt.Sprintf("%s/%s", tc.name, dtype), func(b *testing.B) {
				benchmarkCausalAttentionMetal(b, tc.rows, tc.history, tc.position, tc.dispatches, dtype)
			})
		}
	}
}

func benchmarkCausalAttentionMetal(b *testing.B, rows, history, position int64, dispatches int, kvDType DType) {
	const (
		heads   int64 = 32
		kvHeads int64 = 8
		headDim int64 = 64
	)
	var (
		device     = requireMetalDevice(b)
		ctx        = context.Background()
		queries    = requireTensor(b, device, rows, heads, headDim)
		output     = requireTensor(b, device, rows, heads, headDim)
		cache, err = NewKVCacheWithDType(device, history, kvHeads, headDim, kvDType)
	)
	requireError(b, err, nil)
	b.Cleanup(func() { requireError(b, cache.Close(), nil) })

	requireError(b, queries.Upload(attentionValues(rows*heads*headDim, 401, 0.5)), nil)

	var (
		kSeed      = requireTensor(b, device, history, kvHeads, headDim)
		vSeed      = requireTensor(b, device, history, kvHeads, headDim)
		seedScope  *Scope
		seedUpdate *KVAppend
	)
	seedScope, err = device.NewScope()
	requireError(b, err, nil)
	requireError(b, kSeed.Upload(attentionValues(history*kvHeads*headDim, 402, 0.5)), nil)
	requireError(b, vSeed.Upload(attentionValues(history*kvHeads*headDim, 403, 0.5)), nil)
	seedUpdate = stageKV(b, seedScope, cache, kSeed, vSeed, 0)
	completeAttentionScope(b, seedScope)
	requireError(b, seedUpdate.Publish(), nil)
	requireError(b, kSeed.Close(), nil)
	requireError(b, vSeed.Close(), nil)

	runOnce := func() (elapsed time.Duration) {
		var (
			start    = time.Now()
			scope, e = device.NewScope()
		)
		requireError(b, e, nil)
		defer scope.Close()
		for range dispatches {
			requireError(b, scope.CausalAttention(queries, cache, output, position), nil)
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
	b.ReportMetric(float64(dispatches), "dispatches/op")
}
