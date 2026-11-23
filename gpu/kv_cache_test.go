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
				active = cache.state.slots[cache.state.active].state.buffer.(*fakeAllocation)
				before = slices.Clone(active.values)
				failed = requireScope(t, device)
				native = failed.state.commands.(*fakeCommandBuffer)
				update *KVAppend
				err    error
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
			if cache.Len() != 1 || !slices.Equal(before, active.values) {
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
				result = cache.state.slots[cache.state.active].state.buffer.(*fakeAllocation)
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

func canceledContext() (ctx context.Context) {
	var end context.CancelFunc
	ctx, end = context.WithCancel(context.Background())
	end()
	return ctx
}
