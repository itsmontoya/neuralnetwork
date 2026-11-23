package gpu

import (
	"context"
	"errors"
	"testing"
)

func Test_FFNDiscardAndRetry(t *testing.T) {
	var (
		ctx         = context.Background()
		stop, end   = context.WithCancel(ctx)
		commitFault = errors.New("injected FFN commit failure")
	)
	end()
	var tests = []struct {
		name string
		ctx  context.Context
		want error
	}{
		{name: "poisoned by down projection validation", ctx: ctx, want: ErrInvalid},
		{name: "canceled before submit", ctx: stop, want: context.Canceled},
		{name: "commit failure", ctx: ctx, want: commitFault},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				backend   fakeBackend
				gpu       = newDevice(&backend, 128)
				weights   = newFFNWeights(t, gpu, 1, 2)
				scratch   = newFFNScratch(t, gpu, 1, 1, 2)
				scope     = requireScope(t, gpu)
				badWeight = requireTensor(t, gpu, 2, 1)
				output    [1]float32
			)
			requireError(t, weights[0].Upload([]float32{2, 3}), nil)
			requireError(t, weights[1].Upload([]float32{4, 5}), nil)
			requireError(t, weights[2].Upload([]float32{6, 7}), nil)
			requireError(t, scratch.input.Upload([]float32{1}), nil)
			requireError(t, scratch.output.Upload([]float32{9}), nil)

			var encodeWeights = weights
			if tt.want == ErrInvalid {
				encodeWeights[2] = badWeight
			}
			var encodeErr = scratch.encode(scope, encodeWeights)
			if tt.want == ErrInvalid {
				requireError(t, encodeErr, ErrInvalid)
			} else {
				requireError(t, encodeErr, nil)
			}
			if tt.want == commitFault {
				scope.state.commands.(*fakeCommandBuffer).commitErr = commitFault
			}
			requireError(t, scope.Submit(tt.ctx), tt.want)
			requireError(t, scope.Close(), nil)
			requireError(t, scratch.output.DownloadInto(output[:]), nil)
			if output[0] != 9 || backend.submissions.Load() != 0 || backend.waits.Load() != 0 || backend.liveScopes.Load() != 0 {
				t.Fatalf("discard published work: output=%v, submissions=%d, waits=%d, scopes=%d",
					output, backend.submissions.Load(), backend.waits.Load(), backend.liveScopes.Load())
			}

			var retry = requireScope(t, gpu)
			requireError(t, scratch.encode(retry, weights), nil)
			requireError(t, retry.Submit(ctx), nil)
			requireError(t, retry.Wait(ctx), nil)
			requireError(t, retry.Close(), nil)
			if backend.submissions.Load() != 1 || backend.waits.Load() != 1 {
				t.Fatal("retry must use one submission and one wait")
			}
		})
	}
}

func Test_FFNFailedWriteAndDrain(t *testing.T) {
	var (
		ctx         = context.Background()
		stop, end   = context.WithCancel(ctx)
		backend     fakeBackend
		gpu         = newDevice(&backend, 128)
		weights     = newFFNWeights(t, gpu, 1, 2)
		scratch     = newFFNScratch(t, gpu, 1, 1, 2)
		scope       = requireScope(t, gpu)
		native      = scope.state.commands.(*fakeCommandBuffer)
		fault       = errors.New("injected FFN failure after a native write")
		gate        = make(chan struct{})
		waitResult  = make(chan error, 1)
		closeResult = make(chan error, 1)
		output      [1]float32
	)
	defer end()
	var unblock = unblockAtCleanup(t, gate)
	native.waitErr = fault
	native.waitStarted = make(chan struct{})
	native.waitGate = gate
	for _, weight := range weights {
		requireError(t, weight.Upload([]float32{2, 3}), nil)
	}
	requireError(t, scratch.input.Upload([]float32{1}), nil)
	requireError(t, scratch.output.Upload([]float32{9}), nil)
	requireError(t, scratch.encode(scope, weights), nil)
	if backend.matmuls.Load() != 3 || backend.elementwise.Load() != 1 ||
		backend.uploads.Load() != 5 || backend.downloads.Load() != 0 || backend.waits.Load() != 0 || backend.submissions.Load() != 0 {
		t.Fatal("recording FFN must encode four operations without transferring, submitting or waiting")
	}

	// The fake math encoders record counts. Simulate a down write before
	// returning the command failure so the test detects accidental publication.
	var storage = scratch.output.state.buffer.(*fakeAllocation)
	native.commands = append(native.commands, func() { storage.values[0] = 12 })
	requireError(t, scope.Submit(ctx), nil)
	go func() { waitResult <- scope.Wait(stop) }()
	waitSignal(t, native.waitStarted)
	end()
	requireError(t, <-waitResult, context.Canceled)
	go func() { closeResult <- scope.Close() }()
	requireError(t, scratch.activated.Close(), nil)
	requireError(t, scratch.output.DownloadInto(output[:]), ErrBusy)
	requireError(t, weights[0].Upload([]float32{0, 0}), ErrBusy)
	requireError(t, gpu.Close(), ErrBusy)
	if backend.liveBuffers.Load() != 8 || backend.downloads.Load() != 0 {
		t.Fatal("outstanding FFN must retain all allocations and prohibit readback")
	}
	unblock()
	requireError(t, <-closeResult, fault)
	requireError(t, scope.Close(), nil)
	if storage.values[0] != 12 || backend.liveBuffers.Load() != 7 {
		t.Fatal("failed write was not simulated or draining close leaked scratch")
	}
	requireError(t, scratch.output.DownloadInto(output[:]), ErrUninitialized)
	var intermediate [2]float32
	requireError(t, scratch.gate.DownloadInto(intermediate[:]), ErrUninitialized)
	requireError(t, scratch.up.DownloadInto(intermediate[:]), ErrUninitialized)

	// Replace the explicitly closed activation buffer and rewrite the other
	// destinations with a fresh scope. The original weight leases have drained.
	scratch.activated = requireTensor(t, gpu, 1, 2)
	var retry = requireScope(t, gpu)
	requireError(t, scratch.encode(retry, weights), nil)
	requireError(t, retry.Submit(ctx), nil)
	requireError(t, retry.Wait(ctx), nil)
	requireError(t, retry.Close(), nil)
	requireError(t, scratch.output.DownloadInto(output[:]), nil)
	if backend.uploads.Load() != 5 || backend.downloads.Load() != 1 || backend.submissions.Load() != 2 || backend.waits.Load() != 2 {
		t.Fatal("failure and retry should reuse weights and allow just the final readback")
	}
	for _, tensor := range []*Tensor{weights[0], weights[1], weights[2], scratch.input, scratch.gate, scratch.up, scratch.activated, scratch.output} {
		requireError(t, tensor.Close(), nil)
	}
	requireError(t, gpu.Close(), nil)
	if backend.liveBuffers.Load() != 0 || backend.liveScopes.Load() != 0 {
		t.Fatal("drained FFN leaked fake resources")
	}
}
