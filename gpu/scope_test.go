package gpu

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"
	"time"
)

func Test_ScopeValidation(t *testing.T) {
	var (
		b         fakeBackend
		d         = newDevice(&b, math.MaxUint64)
		other     = newDevice(&b, math.MaxUint64)
		input     = requireTensor(t, d, 2)
		output    = requireTensor(t, d, 2)
		different = requireTensor(t, d, 1, 2)
		foreign   = requireTensor(t, other, 2)
		scope     = requireScope(t, d)
		ctx       = context.Background()
	)
	requireError(t, scope.Submit(ctx), ErrInvalid)
	requireError(t, scope.Wait(ctx), ErrInvalid)
	requireError(t, scope.Fill(nil, 0), ErrInvalid)
	requireError(t, scope.Copy(input, output), ErrUninitialized)
	requireError(t, scope.Copy(input, different), ErrInvalid)
	requireError(t, scope.Copy(foreign, output), ErrInvalid)
	requireError(t, scope.Copy(input, foreign), ErrInvalid)

	var alias = *input
	requireError(t, scope.Copy(input, &alias), ErrInvalid)
	requireError(t, alias.Close(), nil)
	requireError(t, scope.Copy(input, output), ErrClosed)
	requireError(t, scope.Fill(output, 5), nil)
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Submit(ctx), ErrInvalid)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)
	requireError(t, scope.Wait(ctx), ErrClosed)
	requireError(t, scope.Fill(output, 1), ErrClosed)
	if b.submissions.Load() != 1 || b.waits.Load() != 1 {
		t.Fatalf("submissions = %d, waits = %d", b.submissions.Load(), b.waits.Load())
	}
}

func Test_ScopeFillIndexLimit(t *testing.T) {
	if uint64(math.MaxInt)/4 <= math.MaxUint32 {
		t.Skip("host byte limit is below the Metal fill index limit")
	}
	var (
		b         fakeBackend
		d         = newDevice(&b, math.MaxUint64)
		oversized = requireTensor(t, d, int64(math.MaxUint32)+1)
		scope     = requireScope(t, d)
	)
	requireError(t, scope.Fill(oversized, 0), ErrInvalid)
}

func Test_ScopePartialEncodeAndCommitFailures(t *testing.T) {
	var fault = errors.New("injected native failure")
	var tests = []struct {
		name string
		fail func(*Scope, *Tensor) error
		want error
	}{
		{
			name: "validation after successful encode",
			fail: func(scope *Scope, target *Tensor) (err error) { return scope.Copy(target, target) },
			want: ErrInvalid,
		},
		{
			name: "native error after encoding",
			fail: func(scope *Scope, target *Tensor) (err error) {
				scope.state.commands.(*fakeCommandBuffer).encodeErr = fault
				return scope.Fill(target, 7)
			},
			want: fault,
		},
		{
			name: "commit failure before submission",
			fail: func(scope *Scope, target *Tensor) (err error) {
				scope.state.commands.(*fakeCommandBuffer).commitErr = fault
				return scope.Submit(context.Background())
			},
			want: fault,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				b      fakeBackend
				d      = newDevice(&b, 128)
				target = requireTensor(t, d, 1)
				scope  = requireScope(t, d)
				output [1]float32
			)
			requireError(t, target.Upload([]float32{3}), nil)
			requireError(t, scope.Fill(target, 5), nil)
			requireError(t, tt.fail(scope, target), tt.want)
			requireError(t, scope.Submit(context.Background()), ErrInvalid)
			requireError(t, scope.Close(), nil)
			requireError(t, target.DownloadInto(output[:]), nil)
			if output[0] != 3 || b.submissions.Load() != 0 || b.liveScopes.Load() != 0 {
				t.Fatalf("discard changed values or submitted commands: output=%v submissions=%d scopes=%d", output, b.submissions.Load(), b.liveScopes.Load())
			}
			requireError(t, target.Close(), nil)
			requireError(t, d.Close(), nil)
		})
	}
}

func Test_ScopeCanceledBeforeSubmission(t *testing.T) {
	var (
		b           fakeBackend
		d           = newDevice(&b, 128)
		target      = requireTensor(t, d, 1)
		scope       = requireScope(t, d)
		ctx, cancel = context.WithCancel(context.Background())
		output      [1]float32
	)
	defer cancel()
	requireError(t, target.Upload([]float32{4}), nil)
	requireError(t, scope.Fill(target, 8), nil)
	requireError(t, target.DownloadInto(output[:]), ErrBusy)
	cancel()
	requireError(t, scope.Submit(ctx), context.Canceled)
	requireError(t, target.DownloadInto(output[:]), nil)
	if output[0] != 4 || b.submissions.Load() != 0 || b.liveScopes.Load() != 0 {
		t.Fatalf("canceled scope changed data or retained resources: output=%v submissions=%d scopes=%d", output, b.submissions.Load(), b.liveScopes.Load())
	}
	requireError(t, scope.Wait(context.Background()), ErrInvalid)
	requireError(t, scope.Close(), nil)
	requireError(t, target.Close(), nil)
	requireError(t, d.Close(), nil)
}

func Test_ScopeCanceledWaitKeepsLeasesUntilRetry(t *testing.T) {
	var (
		b           fakeBackend
		d           = newDevice(&b, 128)
		source      = requireTensor(t, d, 1)
		target      = requireTensor(t, d, 1)
		scope       = requireScope(t, d)
		native      = scope.state.commands.(*fakeCommandBuffer)
		gate        = make(chan struct{})
		waitResult  = make(chan error, 1)
		ctx, cancel = context.WithCancel(context.Background())
		output      [1]float32
	)
	defer cancel()
	var unblock = unblockAtCleanup(t, gate)
	native.waitStarted = make(chan struct{})
	native.waitGate = gate

	requireError(t, source.Upload([]float32{-9}), nil)
	requireError(t, scope.Copy(source, target), nil)
	requireError(t, scope.Submit(context.Background()), nil)
	go func() { waitResult <- scope.Wait(ctx) }()
	waitSignal(t, native.waitStarted)
	cancel()
	requireError(t, <-waitResult, context.Canceled)
	requireError(t, source.Close(), nil)
	requireError(t, source.Upload([]float32{4}), ErrClosed)
	requireError(t, target.DownloadInto(output[:]), ErrBusy)
	requireError(t, d.Close(), ErrBusy)
	if b.liveBuffers.Load() != 2 || b.downloads.Load() != 0 {
		t.Fatalf("early release or hidden download: buffers=%d downloads=%d", b.liveBuffers.Load(), b.downloads.Load())
	}

	var completion = scope.state.completion
	unblock()
	waitSignal(t, completion.done)
	requireError(t, target.DownloadInto(output[:]), ErrBusy)
	requireError(t, scope.Wait(context.Background()), nil)
	requireError(t, target.DownloadInto(output[:]), nil)
	if output[0] != -9 || b.waits.Load() != 1 || b.liveBuffers.Load() != 1 || b.liveScopes.Load() != 0 {
		t.Fatalf("retry output=%v waits=%d buffers=%d scopes=%d", output, b.waits.Load(), b.liveBuffers.Load(), b.liveScopes.Load())
	}
	requireError(t, scope.Close(), nil)
	requireError(t, target.Close(), nil)
	requireError(t, d.Close(), nil)
}

func Test_ScopeCloseDrainsFailedSubmission(t *testing.T) {
	var (
		b           fakeBackend
		d           = newDevice(&b, 128)
		source      = requireTensor(t, d, 1)
		target      = requireTensor(t, d, 1)
		scope       = requireScope(t, d)
		native      = scope.state.commands.(*fakeCommandBuffer)
		fault       = errors.New("injected execution failure after writing")
		gate        = make(chan struct{})
		closeResult = make(chan error, 1)
		ctx, cancel = context.WithCancel(context.Background())
		output      [1]float32
	)
	defer cancel()
	var unblock = unblockAtCleanup(t, gate)
	native.waitErr = fault
	native.waitStarted = make(chan struct{})
	native.waitGate = gate
	requireError(t, source.Upload([]float32{8}), nil)
	requireError(t, target.Upload([]float32{2}), nil)
	requireError(t, scope.Copy(source, target), nil)
	requireError(t, scope.Submit(context.Background()), nil)
	cancel()
	requireError(t, scope.Wait(ctx), context.Canceled)
	go func() { closeResult <- scope.Close() }()
	waitSignal(t, native.waitStarted)
	requireError(t, target.Upload([]float32{0}), ErrBusy)
	unblock()
	requireError(t, <-closeResult, fault)
	requireError(t, scope.Close(), nil)
	if got := target.state.buffer.(*fakeAllocation).values[0]; got != 8 {
		t.Fatalf("fault did not simulate a completed native write: %g", got)
	}
	requireError(t, target.DownloadInto(output[:]), ErrUninitialized)
	requireError(t, source.DownloadInto(output[:]), nil)
	if output[0] != 8 {
		t.Fatalf("read-only source was invalidated or modified: %v", output)
	}

	var retry = requireScope(t, d)
	requireError(t, retry.Fill(target, 6), nil)
	requireError(t, retry.Submit(context.Background()), nil)
	requireError(t, retry.Close(), nil)
	requireError(t, target.DownloadInto(output[:]), nil)
	if output[0] != 6 {
		t.Fatalf("full rewrite did not recover destination: %v", output)
	}
}

func Test_ScopeSharedReadsAndConflictingWrites(t *testing.T) {
	var (
		b       fakeBackend
		d       = newDevice(&b, 128)
		source  = requireTensor(t, d, 1)
		first   = requireTensor(t, d, 1)
		second  = requireTensor(t, d, 1)
		a       = requireScope(t, d)
		c       = requireScope(t, d)
		writer  = requireScope(t, d)
		ctx     = context.Background()
		results = make(chan error, 2)
		value   [1]float32
	)
	requireError(t, source.Upload([]float32{3}), nil)
	requireError(t, a.Copy(source, first), nil)
	requireError(t, c.Copy(source, second), nil)
	requireError(t, source.Upload([]float32{4}), ErrBusy)
	requireError(t, source.DownloadInto(value[:]), ErrBusy)
	requireError(t, writer.Fill(first, 5), ErrBusy)
	requireError(t, writer.Fill(source, 4), ErrBusy)
	requireError(t, a.Submit(ctx), nil)
	requireError(t, c.Submit(ctx), nil)

	go func() { results <- a.Wait(ctx) }()
	go func() { results <- c.Wait(ctx) }()
	requireError(t, <-results, nil)
	requireError(t, <-results, nil)
	requireError(t, writer.Fill(source, 4), nil)
	requireError(t, writer.Submit(ctx), nil)
	requireError(t, writer.Wait(ctx), nil)
	for _, tensor := range []*Tensor{first, second} {
		requireError(t, tensor.DownloadInto(value[:]), nil)
		if value[0] != 3 {
			t.Fatalf("independent scope output = %v, want 3", value)
		}
	}
	requireError(t, source.DownloadInto(value[:]), nil)
	if value[0] != 4 {
		t.Fatalf("source after writer = %v, want 4", value)
	}
}

func waitSignal(t *testing.T, signal <-chan struct{}) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for test backend")
	}
}

func unblockAtCleanup(t *testing.T, gate chan struct{}) (unblock func()) {
	t.Helper()
	var once sync.Once
	unblock = func() { once.Do(func() { close(gate) }) }
	t.Cleanup(unblock)
	return unblock
}
