//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
)

func Test_MetalAttentionComposition(t *testing.T) {
	var device = requireMetalDevice(t)
	var cases = []struct {
		name                   string
		rows, heads, kv, width int64
	}{
		{"MHA odd heads, 27 rows", 27, 3, 3, 6},
		{"GQA uneven head count, 28 rows", 28, 6, 3, 10},
		{"MQA tail, 29 rows", 29, 5, 1, 8},
		{"32 to 8 GQA, 6 rows", 6, 32, 8, 64},
	}
	for _, shape := range cases {
		t.Run(shape.name, func(t *testing.T) {
			var (
				native   = device.state.backend.(metalBackend).runtime
				baseline = native.ResourceSnapshot()
			)
			t.Run("prefill and decode", func(t *testing.T) {
				var (
					fixture            = newAttentionFixture(t, device, shape.heads, shape.kv, shape.width, shape.rows+1)
					cache              = requireKVCache(t, device, shape.rows+1, shape.kv, shape.width)
					prefill            = newAttentionScratch(t, device, fixture, shape.rows)
					decode             = newAttentionScratch(t, device, fixture, 1)
					resources          = native.ResourceSnapshot()
					prompt             = attentionValues(shape.rows*fixture.hidden, 123, 1)
					token              = attentionValues(fixture.hidden, 456, 1)
					want, keys, values []float64
				)
				fixture.prepare()
				fixture.upload(t)
				want, keys, values = fixture.reference(prompt, 0, nil, nil)
				t.Run("prefill", func(t *testing.T) {
					runMetalAttentionStep(t, device, fixture, prefill, cache, prompt, want, 0)
				})
				want, _, _ = fixture.reference(token, shape.rows, keys, values)
				t.Run("decode", func(t *testing.T) {
					runMetalAttentionStep(t, device, fixture, decode, cache, token, want, shape.rows)
				})
				var final = native.ResourceSnapshot()
				if final.CreatedBuffers != resources.CreatedBuffers || final.LiveBufferBytes != resources.LiveBufferBytes ||
					final.SubmittedCommands != resources.SubmittedCommands+2 || final.CompletedCommands != resources.CompletedCommands+2 ||
					final.LiveScopes != resources.LiveScopes {
					t.Fatalf("unexpected resource use: before %+v after %+v", resources, final)
				}
			})
			var final = native.ResourceSnapshot()
			if final.LiveBuffers != baseline.LiveBuffers || final.LiveBufferBytes != baseline.LiveBufferBytes || final.LiveScopes != baseline.LiveScopes {
				t.Fatalf("composition leaked resources: before %+v after %+v", baseline, final)
			}
		})
	}
}

func runMetalAttentionStep(t *testing.T, device *Device, f *attentionFixture, scratch *attentionScratch, cache *KVCache, input []float32, want []float64, position int64) {
	t.Helper()
	var (
		scope = requireScope(t, device)
		got   = make([]float32, len(input))
	)
	requireError(t, scratch.input.Upload(input), nil)
	var update, err = scratch.encode(scope, cache, f, position)
	requireError(t, err, nil)
	defer update.Close()
	requireError(t, scratch.output.DownloadInto(got), ErrBusy)
	completeAttentionScope(t, scope)
	requireError(t, scratch.output.DownloadInto(got), nil)
	compareAttention(t, got, want, 1e-5)
	if cache.Len() != position {
		t.Fatal("completion published the cache before validation")
	}
	requireError(t, update.Publish(), nil)
	if cache.Len() != position+scratch.input.state.shape[0] {
		t.Fatal("publication did not expose new rows")
	}
}

func Test_MetalAttentionStableMaskAndViews(t *testing.T) {
	var (
		device = requireMetalDevice(t)
		cache  = requireKVCache(t, device, 7, 1, 3)
		query  = requireTensor(t, device, 3, 3, 3)
		keys   = requireTensor(t, device, 4, 1, 3)
		values = requireTensor(t, device, 4, 3)
		output = requireTensor(t, device, 3, 3, 3)
		result = make([]float32, 27)
	)
	var keyData = []float32{100, 0, 0, 99, 0, 0, 200, 0, 0, 300, 0, 0}
	var valueData = []float32{1, 3, 5, -1, -3, -5, 2, 4, 6, 8, 9, 10}
	var queryData = []float32{100, 0, 0, -100, 0, 0, 0, 0, 0}
	queryData = slices.Repeat(queryData, 3)
	requireError(t, query.Upload(queryData), nil)
	requireError(t, keys.Upload(keyData), nil)
	requireError(t, values.Upload(valueData), nil)

	var empty = requireScope(t, device)
	requireError(t, empty.CausalAttention(query, cache, output, 0), nil)
	completeAttentionScope(t, empty)
	requireError(t, output.DownloadInto(result), nil)
	compareAttention(t, result, make([]float64, 27), 0)

	var write = requireScope(t, device)
	var update = stageKV(t, write, cache, keys, values, 0)
	requireError(t, write.CausalAttention(query, cache, output, 1), nil)
	completeAttentionScope(t, write)
	requireError(t, output.DownloadInto(result), nil)
	compareAttention(t, result, referenceAttention(testFP64(queryData), testFP64(keyData), testFP64(valueData), 3, 1, 3, 1), 1e-6)

	var committed = requireScope(t, device)
	requireError(t, committed.CausalAttention(query, cache, output, 0), nil)
	completeAttentionScope(t, committed)
	requireError(t, output.DownloadInto(result), nil)
	compareAttention(t, result, make([]float64, 27), 0)
	requireError(t, update.Publish(), nil)
	var read = requireScope(t, device)
	requireError(t, read.CausalAttention(query, cache, output, math.MaxInt64-2), nil)
	completeAttentionScope(t, read)
	requireError(t, output.DownloadInto(result), nil)
	compareAttention(t, result, referenceAttention(testFP64(queryData), testFP64(keyData), testFP64(valueData), 3, 1, 3, 4), 1e-6)
}

func Test_MetalAttentionRotaryAndRMSNorm(t *testing.T) {
	var (
		device  = requireMetalDevice(t)
		input   = requireTensor(t, device, 3, 2, 6)
		rotated = requireTensor(t, device, 3, 2, 6)
		norm    = requireTensor(t, device, 3, 2, 6)
		weight  = requireTensor(t, device, 6)
		cosine  = requireTensor(t, device, 5, 3)
		sine    = requireTensor(t, device, 5, 3)
		scope   = requireScope(t, device)
		data    = attentionValues(36, 77, 1)
		weights = attentionValues(6, 78, 1)
		c, s    = attentionCoefficients(24, 6)
		result  = make([]float32, 36)
	)
	clear(data[:6])
	data[6], data[7], data[8] = math.MaxFloat32, -math.MaxFloat32, 1e30
	requireError(t, input.Upload(data), nil)
	requireError(t, cosine.Upload(c[19*3:]), nil)
	requireError(t, sine.Upload(s[19*3:]), nil)
	requireError(t, weight.Upload(weights), nil)
	requireError(t, scope.RotatePairs(input, cosine, sine, rotated, 6, 18, 19), ErrInvalid)
	requireError(t, scope.RotatePairs(input, cosine, sine, rotated, 6, 23, 19), ErrInvalid)
	requireError(t, scope.RotatePairs(input, cosine, sine, rotated, 6, math.MaxInt64, math.MaxInt64), ErrInvalid)
	requireError(t, scope.RotatePairs(input, cosine, sine, rotated, 3, 19, 19), ErrInvalid)
	requireError(t, scope.RMSNorm(input, weight, norm, float32(math.NaN())), ErrInvalid)
	requireError(t, scope.RMSNorm(input, weight, norm, attentionEpsilon), nil)
	requireError(t, scope.RotatePairs(norm, cosine, sine, rotated, 6, 21, 19), nil)
	completeAttentionScope(t, scope)
	requireError(t, norm.DownloadInto(result), nil)
	var want = referenceNorm(data, weights)
	compareAttention(t, result, want, 1e-6)
	referenceRotate(want, c, s, 21, 2, 6)
	requireError(t, rotated.DownloadInto(result), nil)
	compareAttention(t, result, want, 1e-6)
}

type failedAttentionCommand struct {
	commandBuffer
	fault error
}

func (c failedAttentionCommand) Wait() (err error) {
	if err = c.commandBuffer.Wait(); err != nil {
		return err
	}
	return c.fault
}

func Test_MetalKVFailureAndRetention(t *testing.T) {
	var (
		device  = requireMetalDevice(t)
		cache   = requireKVCache(t, device, 3, 1, 2)
		input   = requireTensor(t, device, 1, 2)
		output  = requireTensor(t, device, 1, 2)
		prefill = requireScope(t, device)
		result  [2]float32
	)
	requireError(t, input.Upload([]float32{1, 3}), nil)
	var initial = stageKV(t, prefill, cache, input, input, 0)
	completeAttentionScope(t, prefill)
	requireError(t, initial.Publish(), nil)

	var canceled = requireScope(t, device)
	var abandoned = stageKV(t, canceled, cache, input, input, 1)
	requireError(t, canceled.Submit(canceledContext()), context.Canceled)
	requireError(t, canceled.Close(), nil)
	requireError(t, abandoned.Publish(), ErrInvalid)
	var drained = requireScope(t, device)
	var unwaited = stageKV(t, drained, cache, input, input, 1)
	requireError(t, drained.Submit(context.Background()), nil)
	requireError(t, drained.Wait(canceledContext()), context.Canceled)
	requireError(t, drained.Close(), nil)
	requireError(t, unwaited.Publish(), ErrInvalid)
	if cache.Len() != 1 {
		t.Fatal("canceled work advanced the cache")
	}

	var (
		oldReader = requireScope(t, device)
		failed    = requireScope(t, device)
		next      = requireTensor(t, device, 1, 2)
		fault     = errors.New("injected terminal error after real GPU writes")
		ctx       = context.Background()
	)
	requireError(t, next.Upload([]float32{2, 4}), nil)
	requireError(t, oldReader.CausalAttention(input, cache, output, 1), nil)
	var update = stageKV(t, failed, cache, next, next, 1)
	failed.state.commands = failedAttentionCommand{commandBuffer: failed.state.commands, fault: fault}
	requireError(t, failed.Submit(ctx), nil)
	requireError(t, failed.Wait(ctx), fault)
	requireError(t, update.Publish(), ErrInvalid)
	requireError(t, failed.Close(), nil)
	if cache.Len() != 1 {
		t.Fatal("failed command published staged rows")
	}
	requireError(t, next.Upload([]float32{0, 1}), nil)
	var retry = requireScope(t, device)
	var retried = stageKV(t, retry, cache, next, next, 1)
	completeAttentionScope(t, retry)
	requireError(t, retried.Publish(), nil)
	var busy = requireScope(t, device)
	var _, err = busy.AppendKV(cache, next, next, 2)
	requireError(t, err, ErrBusy)
	requireError(t, busy.Close(), nil)
	completeAttentionScope(t, oldReader)
	requireError(t, output.DownloadInto(result[:]), nil)
	compareAttention(t, result[:], []float64{1, 3}, 0)
	var read = requireScope(t, device)
	requireError(t, read.CausalAttention(input, cache, output, 1), nil)
	requireError(t, read.Submit(ctx), nil)
	requireError(t, cache.Close(), nil)
	requireError(t, input.Close(), nil)
	requireError(t, read.Wait(ctx), nil)
	requireError(t, read.Close(), nil)
	requireError(t, output.DownloadInto(result[:]), nil)
	compareAttention(t, result[:], referenceAttention([]float64{1, 3}, []float64{1, 3, 0, 1},
		[]float64{1, 3, 0, 1}, 1, 1, 2, 1), 1e-6)
}

func Test_MetalAttentionConcurrentSessions(t *testing.T) {
	var (
		device  = requireMetalDevice(t)
		f       = newAttentionFixture(t, device, 6, 2, 8, 9)
		caches  = [2]*KVCache{requireKVCache(t, device, 9, 2, 8), requireKVCache(t, device, 9, 2, 8)}
		scratch = [2]*attentionScratch{newAttentionScratch(t, device, f, 7), newAttentionScratch(t, device, f, 7)}
		scopes  = [2]*Scope{requireScope(t, device), requireScope(t, device)}
		updates [2]*KVAppend
		wants   [2][]float64
		ctx     = context.Background()
		results = make(chan error, 2)
		start   = make(chan struct{})
		err     error
	)
	f.prepare()
	f.upload(t)
	for index := range scopes {
		var input = attentionValues(7*f.hidden, uint64(index+711), 1)
		wants[index], _, _ = f.reference(input, 0, nil, nil)
		requireError(t, scratch[index].input.Upload(input), nil)
		updates[index], err = scratch[index].encode(scopes[index], caches[index], f, 0)
		requireError(t, err, nil)
		defer updates[index].Close()
	}
	if f.q.state.readers != 2 || f.norm.state.readers != 2 {
		t.Fatal("independent recordings did not share read-only weights")
	}
	for _, scope := range scopes {
		go func() {
			<-start
			var commandErr = scope.Submit(ctx)
			if commandErr == nil {
				commandErr = scope.Wait(ctx)
			}
			results <- commandErr
		}()
	}
	close(start)
	for range scopes {
		requireError(t, <-results, nil)
	}
	var got = make([]float32, 7*f.hidden)
	for index := range scopes {
		requireError(t, scopes[index].Close(), nil)
		requireError(t, scratch[index].output.DownloadInto(got), nil)
		compareAttention(t, got, wants[index], 1e-5)
		requireError(t, updates[index].Publish(), nil)
	}
	requireError(t, caches[0].Reset(), nil)
	if caches[0].Len() != 0 || caches[1].Len() != 7 {
		t.Fatal("one cache changed the other session's length")
	}
}

func testFP64(input []float32) (out []float64) {
	out = make([]float64, len(input))
	for index, value := range input {
		out[index] = float64(value)
	}
	return out
}
