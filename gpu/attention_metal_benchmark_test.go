//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// Benchmark_AttentionMetal runs a full 32:8 attention layer with fixed
// synthetic weights and either a populated decode cache or a fresh prefill.
func Benchmark_AttentionMetal(b *testing.B) {
	for _, shape := range [][2]int64{{1, 511}, {29, 0}} {
		b.Run(fmt.Sprintf("M%d-H%d", shape[0], shape[1]), func(b *testing.B) {
			benchmarkAttentionMetal(b, shape[0], shape[1])
		})
	}
}

func benchmarkAttentionMetal(b *testing.B, rows, position int64) {
	const (
		heads    int64 = 32
		kvHeads  int64 = 8
		width    int64 = 64
		capacity int64 = 512
	)
	var (
		start    = time.Now()
		device   = requireMetalDevice(b)
		native   = device.state.backend.(metalBackend).runtime
		open     = time.Since(start)
		baseline = native.ResourceSnapshot()
		ctx      = context.Background()
	)
	start = time.Now()
	var (
		fixture    = newAttentionFixture(b, device, heads, kvHeads, width, capacity)
		scratch    = newAttentionScratch(b, device, fixture, rows)
		cache      = requireKVCache(b, device, capacity, kvHeads, width)
		allocation = time.Since(start)
	)
	fixture.prepare()
	start = time.Now()
	fixture.upload(b)
	var (
		upload     = time.Since(start)
		input      = attentionValues(rows*fixture.hidden, 49, 1)
		output     = make([]float32, len(input))
		keys       = attentionValues(position*kvHeads*width, 51, 1)
		values     = attentionValues(position*kvHeads*width, 53, 1)
		want, _, _ = fixture.reference(input, position, testFP64(keys), testFP64(values))
	)
	if position > 0 {
		var (
			k          = requireTensor(b, device, position, kvHeads, width)
			v          = requireTensor(b, device, position, kvHeads, width)
			scope, err = device.NewScope()
		)
		requireError(b, err, nil)
		defer scope.Close()
		requireError(b, k.Upload(keys), nil)
		requireError(b, v.Upload(values), nil)
		var update = stageKV(b, scope, cache, k, v, 0)
		completeAttentionScope(b, scope)
		requireError(b, update.Publish(), nil)
		requireError(b, k.Close(), nil)
		requireError(b, v.Close(), nil)
	}

	var (
		first = timeAttentionMetal(b, ctx, device, fixture, scratch, cache, position, input, output)
		warm  = native.ResourceSnapshot()
	)
	compareAttention(b, output, want, 1e-5)
	for range 3 {
		timeAttentionMetal(b, ctx, device, fixture, scratch, cache, position, input, output)
		compareAttention(b, output, want, 1e-5)
	}

	var sums [8]time.Duration
	b.ReportAllocs()
	for b.Loop() {
		var phases = timeAttentionMetal(b, ctx, device, fixture, scratch, cache, position, input, output)
		b.StopTimer()
		compareAttention(b, output, want, 1e-5)
		for index, elapsed := range phases {
			sums[index] += elapsed
		}
		b.StartTimer()
	}

	var (
		final        = native.ResourceSnapshot()
		scratchBytes = 4 * rows * (7*fixture.hidden + 3*fixture.kvWidth)
		slotBytes    = 2 * capacity * kvHeads * width * 4
		modelBytes   = 4 * (2*fixture.hidden*fixture.hidden + 2*fixture.hidden*fixture.kvWidth + fixture.hidden)
		tableBytes   = 4 * capacity * width
	)
	if final.LiveBuffers != baseline.LiveBuffers+19 || final.LiveScopes != baseline.LiveScopes ||
		final.LiveBufferBytes != baseline.LiveBufferBytes+uint64(scratchBytes+2*slotBytes+modelBytes+tableBytes) ||
		final.CreatedBuffers != warm.CreatedBuffers || final.SubmittedCommands != warm.SubmittedCommands+uint64(b.N)+3 ||
		final.CompletedCommands != warm.CompletedCommands+uint64(b.N)+3 {
		b.Fatalf("unexpected resource use: baseline %+v warm %+v final %+v", baseline, warm, final)
	}
	b.ReportMetric(float64(open), "open-ns")
	b.ReportMetric(float64(allocation), "alloc-device-ns")
	b.ReportMetric(float64(upload), "upload-shared-ns")
	b.ReportMetric(float64(first[2]), "first-encode-ns")
	b.ReportMetric(float64(first[7]), "first-wall-ns")
	for index, label := range [...]string{"upload-ns/op", "scope-ns/op", "encode-ns/op", "submit-ns/op",
		"wait-ns/op", "close-ns/op", "readback-ns/op", "wall-ns/op"} {
		b.ReportMetric(float64(sums[index])/float64(b.N), label)
	}
	b.ReportMetric(float64(scratchBytes), "scratch-device-B")
	b.ReportMetric(float64(slotBytes), "active-kv-device-B")
	b.ReportMetric(float64(slotBytes), "staged-kv-device-B")
	b.ReportMetric(float64(modelBytes+tableBytes), "shared-device-B")
	b.ReportMetric(float64(4*len(input)+4*len(output)), "activation-host-B")
	b.ReportMetric(float64(2*position*kvHeads*width*4), "prefix-copy-B/op")
	b.ReportMetric(float64(2*rows*kvHeads*width*4), "append-copy-B/op")
	b.ReportMetric(1, "commands/op")
}

func timeAttentionMetal(b *testing.B, ctx context.Context, device *Device, f *attentionFixture, scratch *attentionScratch,
	cache *KVCache, position int64, input, output []float32) (phases [8]time.Duration) {
	b.Helper()
	var start = time.Now()
	requireError(b, scratch.input.Upload(input), nil)
	var uploaded = time.Now()
	var scope, err = device.NewScope()
	requireError(b, err, nil)
	defer scope.Close()
	var opened = time.Now()
	var update *KVAppend
	update, err = scratch.encode(scope, cache, f, position)
	requireError(b, err, nil)
	defer update.Close()
	var encoded = time.Now()
	requireError(b, scope.Submit(ctx), nil)
	var submitted = time.Now()
	requireError(b, scope.Wait(ctx), nil)
	var waited = time.Now()
	requireError(b, scope.Close(), nil)
	requireError(b, update.Close(), nil)
	var closed = time.Now()
	requireError(b, scratch.output.DownloadInto(output), nil)
	var finished = time.Now()
	return [8]time.Duration{
		uploaded.Sub(start), opened.Sub(uploaded), encoded.Sub(opened), submitted.Sub(encoded),
		waited.Sub(submitted), closed.Sub(waited), finished.Sub(closed), finished.Sub(start),
	}
}
