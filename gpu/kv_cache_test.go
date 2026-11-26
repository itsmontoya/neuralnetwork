package gpu

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
)

func requireKVCache(t testing.TB, device *Device, capacity, heads, width int64) (cache *KVCache) {
	t.Helper()
	var err error
	cache, err = NewKVCache(device, capacity, heads, width)
	requireError(t, err, nil)
	t.Cleanup(func() { requireError(t, cache.Close(), nil) })
	return cache
}

func stageKV(t testing.TB, scope *Scope, cache *KVCache, keys, values *Tensor, position int64) (update *KVAppend) {
	t.Helper()
	var err error
	update, err = scope.AppendKV(cache, keys, values, position)
	requireError(t, err, nil)
	t.Cleanup(func() { requireError(t, update.Close(), nil) })
	return update
}

func Test_KVCacheDiscardAndRetry(t *testing.T) {
	var (
		ctx        = context.Background()
		canceled   = canceledContext()
		fault      = errors.New("injected KV fault")
		finishStep = func(t *testing.T, scope *Scope) {
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)
		}
	)
	var tests = []string{
		"partial encoding", "append encoder failure", "cancel before submit", "commit error",
		"GPU error", "canceled wait", "abandon completed update", "abandon recording",
	}
	for _, name := range tests {
		t.Run(name, func(t *testing.T) {
			var (
				backend fakeBackend
				device  = newDevice(&backend, 128)
				cache   = requireKVCache(t, device, 3, 1, 2)
				key     = requireTensor(t, device, 1, 2)
				value   = requireTensor(t, device, 1, 1, 2)
				initial = requireScope(t, device)
			)
			defer func() {
				for _, tensor := range []*Tensor{key, value} {
					requireError(t, tensor.Close(), nil)
				}
				requireError(t, cache.Close(), nil)
				requireError(t, device.Close(), nil)
				if backend.liveBuffers.Load() != 0 || backend.liveScopes.Load() != 0 {
					t.Fatal("KV storage leaked")
				}
			}()
			requireError(t, key.Upload([]float32{1, 2}), nil)
			requireError(t, value.Upload([]float32{3, 4}), nil)
			var first = stageKV(t, initial, cache, key, value, 0)
			requireError(t, first.Publish(), ErrBusy)
			finishStep(t, initial)
			if cache.Len() != 0 {
				t.Fatal("Wait published initial KV")
			}
			requireError(t, first.Publish(), nil)
			requireError(t, first.Publish(), ErrInvalid)
			var (
				active       = cache.state.slot.state.storage.buffer.(*fakeAllocation)
				beforeKeys   = slices.Clone(active.values[:2])
				beforeValues = slices.Clone(active.values[6:8])
				failed       = requireScope(t, device)
				native       = failed.state.commands.(*fakeCommandBuffer)
				update       *KVAppend
				err          error
			)
			requireError(t, key.Upload([]float32{5, 6}), nil)
			requireError(t, value.Upload([]float32{7, 8}), nil)
			if name == "append encoder failure" {
				native.encodeErr = fault
				update, err = failed.AppendKV(cache, key, value, 1)
				requireError(t, err, fault)
			} else {
				update = stageKV(t, failed, cache, key, value, 1)
				requireError(t, update.Publish(), ErrBusy)
			}
			requireError(t, cache.Reset(), ErrBusy)
			switch name {
			case "partial encoding":
				requireError(t, failed.Copy(key, nil), ErrInvalid)
				requireError(t, failed.Submit(ctx), ErrInvalid)
			case "append encoder failure":
				requireError(t, failed.Submit(ctx), ErrInvalid)
			case "cancel before submit":
				requireError(t, failed.Submit(canceled), context.Canceled)
			case "commit error":
				native.commitErr = fault
				requireError(t, failed.Submit(ctx), fault)
			case "GPU error":
				// The fake executes the writes before reporting the error.
				native.waitErr = fault
				requireError(t, failed.Submit(ctx), nil)
				requireError(t, failed.Wait(ctx), fault)
			case "canceled wait":
				requireError(t, failed.Submit(ctx), nil)
				requireError(t, failed.Wait(canceled), context.Canceled)
				requireError(t, update.Publish(), ErrBusy)
			case "abandon completed update":
				requireError(t, failed.Submit(ctx), nil)
				requireError(t, failed.Wait(ctx), nil)
				requireError(t, update.Close(), nil)
			case "abandon recording":
				requireError(t, update.Close(), nil)
				var competing = requireScope(t, device)
				_, err = competing.AppendKV(cache, key, value, 1)
				requireError(t, err, ErrBusy)
				requireError(t, competing.Close(), nil)
			}
			requireError(t, failed.Close(), nil)
			if update != nil {
				requireError(t, update.Publish(), ErrInvalid)
			}
			if cache.Len() != 1 || !slices.Equal(beforeKeys, active.values[:2]) || !slices.Equal(beforeValues, active.values[6:8]) {
				t.Fatalf("failed append changed committed history: length %d values %v", cache.Len(), active.values)
			}
			requireError(t, key.Upload([]float32{9, 10}), nil)
			requireError(t, value.Upload([]float32{11, 12}), nil)
			var (
				retry = requireScope(t, device)
				next  = stageKV(t, retry, cache, key, value, 1)
			)
			finishStep(t, retry)
			requireError(t, next.Publish(), nil)
			var (
				result = cache.state.slot.state.storage.buffer.(*fakeAllocation)
				want   = []float32{1, 2, 9, 10, 0, 0, 3, 4, 11, 12, 0, 0}
			)
			if cache.Len() != 2 || !slices.Equal(result.values, want) {
				t.Fatalf("retry published incorrect values: length %d values %v", cache.Len(), result.values)
			}
		})
	}
}

func Test_KVCacheCanceledWaitAndCloseRetainsStorage(t *testing.T) {
	var (
		backend     fakeBackend
		device      = newDevice(&backend, 128)
		cache       = requireKVCache(t, device, 2, 1, 2)
		ctx         = context.Background()
		stop, end   = context.WithCancel(ctx)
		value       = requireTensor(t, device, 1, 2)
		scope       = requireScope(t, device)
		native      = scope.state.commands.(*fakeCommandBuffer)
		gate        = make(chan struct{})
		waitResult  = make(chan error, 1)
		closeResult = make(chan error, 1)
	)
	defer end()
	var unblock = unblockAtCleanup(t, gate)
	native.waitStarted = make(chan struct{})
	native.waitGate = gate
	requireError(t, value.Upload([]float32{1, 2}), nil)
	var update = stageKV(t, scope, cache, value, value, 0)
	requireError(t, scope.Submit(ctx), nil)
	go func() { waitResult <- scope.Wait(stop) }()
	waitSignal(t, native.waitStarted)
	end()
	requireError(t, <-waitResult, context.Canceled)
	go func() { closeResult <- scope.Close() }()
	requireError(t, cache.Close(), nil)
	requireError(t, value.Close(), nil)
	requireError(t, update.Publish(), ErrClosed)
	requireError(t, device.Close(), ErrBusy)
	if backend.liveBuffers.Load() != 2 {
		t.Fatalf("want two leased allocations, got %d", backend.liveBuffers.Load())
	}
	unblock()
	requireError(t, <-closeResult, nil)
	requireError(t, device.Close(), nil)
	if backend.liveBuffers.Load() != 0 || backend.liveScopes.Load() != 0 {
		t.Fatal("draining close leaked KV allocations")
	}
}

func Test_KVCacheBounds(t *testing.T) {
	var (
		backend fakeBackend
		device  = newDevice(&backend, math.MaxUint64)
		other   = newDevice(&backend, 512)
		cache   = requireKVCache(t, device, 3, 2, 2)
		key     = requireTensor(t, device, 1, 4)
		query   = requireTensor(t, device, 2, 6)
		output  = requireTensor(t, device, 2, 6)
		value   = requireTensor(t, device, 1, 2, 2)
		wrong   = requireTensor(t, device, 1, 3)
		history = requireScope(t, device)
	)
	defer func() {
		for _, tensor := range []*Tensor{key, query, output, value, wrong} {
			requireError(t, tensor.Close(), nil)
		}
		requireError(t, history.Close(), nil)
		requireError(t, cache.Close(), nil)
		requireError(t, device.Close(), nil)
		requireError(t, other.Close(), nil)
	}()
	for _, dims := range [][3]int64{{0, 2, 2}, {1, -1, 2}, {1, 2, 0}, {math.MaxInt64, 2, 2}, {1 << 30, 2, 1}} {
		var _, err = NewKVCache(device, dims[0], dims[1], dims[2])
		requireError(t, err, ErrInvalid)
	}
	requireError(t, key.Upload([]float32{1, 2, 3, 4}), nil)
	requireError(t, value.Upload([]float32{5, 6, 7, 8}), nil)
	for _, position := range []int64{-1, 1, 3, math.MaxInt64} {
		var _, err = history.AppendKV(cache, key, value, position)
		requireError(t, err, ErrInvalid)
	}
	var _, err = history.AppendKV(cache, wrong, value, 0)
	requireError(t, err, ErrInvalid)
	requireError(t, history.CausalAttention(query, cache, output, 0), ErrInvalid)
	requireError(t, history.CausalAttention(query, cache, output, math.MaxInt64), ErrInvalid)
	var foreign = requireScope(t, other)
	_, err = foreign.AppendKV(cache, key, value, 0)
	requireError(t, err, ErrInvalid)
	requireError(t, foreign.Close(), nil)

	var update = stageKV(t, history, cache, key, value, 0)
	requireError(t, history.Submit(context.Background()), nil)
	requireError(t, history.Wait(canceledContext()), context.Canceled)
	requireError(t, update.Publish(), ErrBusy)
	requireError(t, history.Wait(context.Background()), nil)
	requireError(t, update.Publish(), nil)
	requireError(t, cache.Reset(), nil)
	if cache.Len() != 0 {
		t.Fatal("reset exposes committed history")
	}
}

func Test_KVCacheWithDTypeValidationAndFloat16(t *testing.T) {
	var (
		backend fakeBackend
		device  = newDevice(&backend, 4096)
		ctx     = context.Background()
	)
	defer func() {
		requireError(t, device.Close(), nil)
		if backend.liveBuffers.Load() != 0 || backend.liveScopes.Load() != 0 {
			t.Fatal("KV cache dtype test leaked resources")
		}
	}()

	for _, bad := range []DType{0, BFloat16, DType(99)} {
		var out, err = NewKVCacheWithDType(device, 4, 1, 2, bad)
		if out != nil || !errors.Is(err, ErrInvalid) {
			t.Fatalf("NewKVCacheWithDType(%v) = %v, %v, want nil and ErrInvalid", bad, out, err)
		}
	}

	var nilCache *KVCache
	if nilCache.DType().Valid() {
		t.Fatalf("nil cache DType = %v, want invalid", nilCache.DType())
	}

	var (
		f32Cache, err32 = NewKVCacheWithDType(device, 4, 1, 2, Float32)
		f16Cache, err16 = NewKVCacheWithDType(device, 4, 1, 2, Float16)
	)
	requireError(t, err32, nil)
	requireError(t, err16, nil)
	if f32Cache.DType() != Float32 || f16Cache.DType() != Float16 {
		t.Fatalf("cache dtypes = (%v, %v), want (Float32, Float16)", f32Cache.DType(), f16Cache.DType())
	}
	if backend.liveBuffers.Load() != 2 {
		t.Fatalf("live KV buffers for two single-slot caches = %d, want 2", backend.liveBuffers.Load())
	}
	if f16Cache.state.slot.Bytes()*2 != f32Cache.state.slot.Bytes() {
		t.Fatalf("FP16 KV slot bytes = %d, want half of FP32 %d", f16Cache.state.slot.Bytes(), f32Cache.state.slot.Bytes())
	}

	var (
		key   = requireTensor(t, device, 1, 2)
		value = requireTensor(t, device, 1, 2)
		scope = requireScope(t, device)
	)
	requireError(t, key.Upload([]float32{1.5, -2.25}), nil)
	requireError(t, value.Upload([]float32{0.5, 4.0}), nil)
	var update = stageKV(t, scope, f16Cache, key, value, 0)
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)
	requireError(t, update.Publish(), nil)
	if f16Cache.Len() != 1 {
		t.Fatalf("FP16 cache len = %d, want 1", f16Cache.Len())
	}

	// Second append writes new Float16 rows in place without copying the prefix.
	requireError(t, key.Upload([]float32{3.0, -0.5}), nil)
	requireError(t, value.Upload([]float32{-1.0, 2.0}), nil)
	var second = requireScope(t, device)
	var update2 = stageKV(t, second, f16Cache, key, value, 1)
	requireError(t, second.Submit(ctx), nil)
	requireError(t, second.Wait(ctx), nil)
	requireError(t, second.Close(), nil)
	requireError(t, update2.Publish(), nil)

	var active = f16Cache.state.slot.state.storage.buffer.(*fakeAllocation)
	var wantPrefix = []uint16{
		Float32ToFloat16(1.5), Float32ToFloat16(-2.25),
		Float32ToFloat16(3.0), Float32ToFloat16(-0.5),
	}
	if !slices.Equal(active.bits16[:4], wantPrefix) {
		t.Fatalf("FP16 KV keys = %x, want %x", active.bits16[:4], wantPrefix)
	}
	var wantValues = []uint16{
		Float32ToFloat16(0.5), Float32ToFloat16(4.0),
		Float32ToFloat16(-1.0), Float32ToFloat16(2.0),
	}
	if !slices.Equal(active.bits16[8:12], wantValues) {
		t.Fatalf("FP16 KV values = %x, want %x", active.bits16[8:12], wantValues)
	}

	requireError(t, key.Close(), nil)
	requireError(t, value.Close(), nil)
	requireError(t, f32Cache.Close(), nil)
	requireError(t, f16Cache.Close(), nil)
	if f16Cache.DType() != Float16 {
		t.Fatalf("closed cache DType = %v, want Float16", f16Cache.DType())
	}
}

func Test_KVCacheResetBusyWithActiveReaderAndFailedAppendReadable(t *testing.T) {
	var (
		backend fakeBackend
		device  = newDevice(&backend, 1024)
		cache   = requireKVCache(t, device, 3, 1, 2)
		key     = requireTensor(t, device, 1, 2)
		value   = requireTensor(t, device, 1, 2)
		query   = requireTensor(t, device, 1, 2)
		output  = requireTensor(t, device, 1, 2)
		ctx     = context.Background()
	)
	defer func() {
		for _, tensor := range []*Tensor{key, value, query, output} {
			requireError(t, tensor.Close(), nil)
		}
		requireError(t, cache.Close(), nil)
		requireError(t, device.Close(), nil)
		if backend.liveBuffers.Load() != 0 || backend.liveScopes.Load() != 0 {
			t.Fatal("KV cache test leaked resources")
		}
	}()

	requireError(t, key.Upload([]float32{1, 2}), nil)
	requireError(t, value.Upload([]float32{3, 4}), nil)
	requireError(t, query.Upload([]float32{1, 0}), nil)

	var initial = requireScope(t, device)
	var first = stageKV(t, initial, cache, key, value, 0)
	requireError(t, initial.Submit(ctx), nil)
	requireError(t, initial.Wait(ctx), nil)
	requireError(t, initial.Close(), nil)
	requireError(t, first.Publish(), nil)

	// Active reader must block Reset and competing AppendKV with ErrBusy.
	var reader = requireScope(t, device)
	requireError(t, reader.CausalAttention(query, cache, output, 0), nil)
	requireError(t, cache.Reset(), ErrBusy)
	var writer = requireScope(t, device)
	var _, err = writer.AppendKV(cache, key, value, 1)
	requireError(t, err, ErrBusy)
	requireError(t, writer.Close(), nil)
	requireError(t, reader.Submit(ctx), nil)
	requireError(t, cache.Reset(), ErrBusy)
	requireError(t, reader.Wait(ctx), nil)
	requireError(t, reader.Close(), nil)

	// A failed append at position 1 must leave the committed prefix initialized
	// and readable by subsequent CausalAttention scopes.
	var (
		failed = requireScope(t, device)
		native = failed.state.commands.(*fakeCommandBuffer)
		fault  = errors.New("injected append GPU error")
	)
	native.waitErr = fault
	var failedUpdate = stageKV(t, failed, cache, key, value, 1)
	requireError(t, failed.Submit(ctx), nil)
	requireError(t, failed.Wait(ctx), fault)
	requireError(t, failed.Close(), nil)
	requireError(t, failedUpdate.Publish(), ErrInvalid)
	if cache.Len() != 1 {
		t.Fatalf("cache len after failed append = %d, want 1", cache.Len())
	}

	var afterFailReader = requireScope(t, device)
	requireError(t, afterFailReader.CausalAttention(query, cache, output, 0), nil)
	requireError(t, afterFailReader.Submit(ctx), nil)
	requireError(t, afterFailReader.Wait(ctx), nil)
	requireError(t, afterFailReader.Close(), nil)

	requireError(t, cache.Reset(), nil)
	if cache.Len() != 0 {
		t.Fatalf("cache len after Reset = %d, want 0", cache.Len())
	}
}

func canceledContext() (ctx context.Context) {
	var end context.CancelFunc
	ctx, end = context.WithCancel(context.Background())
	end()
	return ctx
}
