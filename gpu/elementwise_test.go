package gpu

import (
	"context"
	"errors"
	"math"
	"testing"
)

func Test_ScopeElementwiseValidation(t *testing.T) {
	var (
		backend       fakeBackend
		device        = newDevice(&backend, 128)
		other         = newDevice(&backend, 128)
		input         = requireTensor(t, device, 1, 3)
		uninitialized = requireTensor(t, device, 1, 3)
		destination   = requireTensor(t, device, 1, 3)
		vector        = requireTensor(t, device, 3)
		foreign       = requireTensor(t, other, 1, 3)
		closed        = requireTensor(t, device, 1, 3)
		alias         = *input
		scope         = requireScope(t, device)
	)
	requireError(t, input.Upload([]float32{1, 2, 3}), nil)
	requireError(t, vector.Upload([]float32{1, 2, 3}), nil)
	requireError(t, foreign.Upload([]float32{1, 2, 3}), nil)
	requireError(t, closed.Close(), nil)

	var tests = []struct {
		name string
		run  func() error
		want error
	}{
		{name: "missing left", run: func() error { return scope.Add(nil, input, destination) }, want: ErrInvalid},
		{name: "missing right", run: func() error { return scope.Mul(input, nil, destination) }, want: ErrInvalid},
		{name: "zero input handle", run: func() error { return scope.SiLU(&Tensor{}, destination) }, want: ErrInvalid},
		{name: "missing destination", run: func() error { return scope.SwiGLU(input, input, nil) }, want: ErrInvalid},
		{name: "matching length different rank", run: func() error { return scope.Add(input, vector, destination) }, want: ErrInvalid},
		{name: "different destination shape", run: func() error { return scope.SiLU(input, vector) }, want: ErrInvalid},
		{name: "foreign first input", run: func() error { return scope.Mul(foreign, input, destination) }, want: ErrInvalid},
		{name: "foreign second input", run: func() error { return scope.SwiGLU(input, foreign, destination) }, want: ErrInvalid},
		{name: "foreign destination", run: func() error { return scope.SiLU(input, foreign) }, want: ErrInvalid},
		{name: "aliased first input", run: func() error { return scope.Add(input, input, &alias) }, want: ErrInvalid},
		{name: "aliased second input", run: func() error { return scope.Mul(input, destination, destination) }, want: ErrInvalid},
		{name: "uninitialized input", run: func() error { return scope.SiLU(uninitialized, destination) }, want: ErrUninitialized},
		{name: "closed input", run: func() error { return scope.SwiGLU(input, closed, destination) }, want: ErrClosed},
		{name: "closed destination", run: func() error { return scope.SiLU(input, closed) }, want: ErrClosed},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requireError(t, tt.run(), tt.want)
		})
	}
	if backend.elementwise.Load() != 0 {
		t.Fatal("invalid elementwise operation reached the backend")
	}
	requireError(t, scope.Add(input, &alias, destination), nil)
	requireError(t, scope.Close(), nil)
	requireError(t, scope.SiLU(input, destination), ErrClosed)
}

func Test_ScopeElementwiseIndexLimit(t *testing.T) {
	if uint64(math.MaxInt)/4 <= math.MaxUint32 {
		t.Skip("host byte limit is below the Metal elementwise index limit")
	}
	var (
		backend = fakeBackend{}
		device  = newDevice(&backend, math.MaxUint64)
		wide    = requireTensor(t, device, 1, int64(math.MaxUint32)+1)
		atLimit = requireTensor(t, device, int64(math.MaxUint32))
		output  = requireTensor(t, device, int64(math.MaxUint32))
		scope   = requireScope(t, device)
	)
	requireError(t, scope.SiLU(wide, wide), ErrInvalid)
	if backend.elementwise.Load() != 0 {
		t.Fatal("oversized operation reached the backend")
	}
	requireError(t, scope.Fill(atLimit, 1), nil)
	requireError(t, scope.SiLU(atLimit, output), nil)
}

func Test_ScopeElementwiseFailures(t *testing.T) {
	var (
		backend fakeBackend
		device  = newDevice(&backend, 128)
		input   = requireTensor(t, device, 1)
		output  = requireTensor(t, device, 1)
		ctx     = context.Background()
		fault   = errors.New("injected elementwise failure")
		value   [1]float32
	)
	requireError(t, input.Upload([]float32{3}), nil)
	requireError(t, output.Upload([]float32{2}), nil)

	var invalid = requireScope(t, device)
	requireError(t, invalid.Add(input, input, output), nil)
	requireError(t, invalid.SiLU(nil, output), ErrInvalid)
	requireError(t, invalid.Submit(ctx), ErrInvalid)
	requireError(t, invalid.Close(), nil)

	var encoding = requireScope(t, device)
	requireError(t, encoding.Fill(output, 5), nil)
	encoding.state.commands.(*fakeCommandBuffer).encodeErr = fault
	requireError(t, encoding.SiLU(input, output), fault)
	requireError(t, encoding.Submit(ctx), ErrInvalid)
	requireError(t, encoding.Close(), nil)

	var submission = requireScope(t, device)
	requireError(t, submission.Mul(input, input, output), nil)
	submission.state.commands.(*fakeCommandBuffer).commitErr = fault
	requireError(t, submission.Submit(ctx), fault)
	requireError(t, submission.Close(), nil)

	var (
		canceled  = requireScope(t, device)
		stop, end = context.WithCancel(ctx)
	)
	end()
	requireError(t, canceled.SwiGLU(input, input, output), nil)
	requireError(t, canceled.Submit(stop), context.Canceled)
	requireError(t, canceled.Close(), nil)
	requireError(t, output.DownloadInto(value[:]), nil)
	if value[0] != 2 || backend.submissions.Load() != 0 {
		t.Fatalf("discard changed output or submitted work: output=%g submissions=%d", value[0], backend.submissions.Load())
	}

	var execution = requireScope(t, device)
	execution.state.commands.(*fakeCommandBuffer).waitErr = fault
	requireError(t, execution.SwiGLU(input, input, output), nil)
	requireError(t, execution.Submit(ctx), nil)
	requireError(t, execution.Wait(stop), context.Canceled)
	requireError(t, output.DownloadInto(value[:]), ErrBusy)
	requireError(t, execution.Wait(ctx), fault)
	requireError(t, output.DownloadInto(value[:]), ErrUninitialized)
	requireError(t, input.DownloadInto(value[:]), nil)
	if value[0] != 3 {
		t.Fatalf("input changed after failed submission: %g", value[0])
	}
	requireError(t, execution.Close(), nil)
	var retry = requireScope(t, device)
	requireError(t, retry.SwiGLU(input, input, output), nil)
	requireError(t, retry.Submit(ctx), nil)
	requireError(t, retry.Close(), nil)
	requireError(t, output.DownloadInto(value[:]), nil)
}
