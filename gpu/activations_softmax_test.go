package gpu

import (
	"context"
	"errors"
	"math"
	"testing"
)

func Test_ScopeActivationsAndSoftmaxValidation(t *testing.T) {
	var (
		backend       fakeBackend
		device        = newDevice(&backend, 256)
		other         = newDevice(&backend, 256)
		input         = requireTensor(t, device, 2, 3, 4)
		uninitialized = requireTensor(t, device, 2, 3, 4)
		destination   = requireTensor(t, device, 2, 3, 4)
		flat          = requireTensor(t, device, 24)
		foreign       = requireTensor(t, other, 2, 3, 4)
		closed        = requireTensor(t, device, 2, 3, 4)
		f16Input      = requireTensorWithDType(t, device, Float16, 2, 3, 4)
		f16Dest       = requireTensorWithDType(t, device, Float16, 2, 3, 4)
		alias         = *input
		view, err     = input.View([]int64{2, 3, 4})
		scope         = requireScope(t, device)
		values        = make([]float32, 24)
	)
	requireError(t, err, nil)
	for i := range values {
		values[i] = float32(i) * 0.25
	}
	requireError(t, input.Upload(values), nil)
	requireError(t, flat.Upload(values), nil)
	requireError(t, foreign.Upload(values), nil)
	requireError(t, f16Input.UploadFloat32As(values), nil)
	requireError(t, closed.Close(), nil)

	var tests = []struct {
		name string
		run  func() error
		want error
	}{
		{name: "relu missing input", run: func() error { return scope.ReLU(nil, destination) }, want: ErrInvalid},
		{name: "relu zero handle", run: func() error { return scope.ReLU(&Tensor{}, destination) }, want: ErrInvalid},
		{name: "relu missing destination", run: func() error { return scope.ReLU(input, nil) }, want: ErrInvalid},
		{name: "relu different rank", run: func() error { return scope.ReLU(input, flat) }, want: ErrInvalid},
		{name: "relu non-float32 input", run: func() error { return scope.ReLU(f16Input, destination) }, want: ErrInvalid},
		{name: "relu non-float32 destination", run: func() error { return scope.ReLU(input, f16Dest) }, want: ErrInvalid},
		{name: "relu foreign input", run: func() error { return scope.ReLU(foreign, destination) }, want: ErrInvalid},
		{name: "relu foreign destination", run: func() error { return scope.ReLU(input, foreign) }, want: ErrInvalid},
		{name: "relu aliased handle", run: func() error { return scope.ReLU(input, &alias) }, want: ErrInvalid},
		{name: "relu aliased view", run: func() error { return scope.ReLU(input, view) }, want: ErrInvalid},
		{name: "relu uninitialized", run: func() error { return scope.ReLU(uninitialized, destination) }, want: ErrUninitialized},
		{name: "relu closed input", run: func() error { return scope.ReLU(closed, destination) }, want: ErrClosed},
		{name: "relu closed destination", run: func() error { return scope.ReLU(input, closed) }, want: ErrClosed},

		{name: "sigmoid missing input", run: func() error { return scope.Sigmoid(nil, destination) }, want: ErrInvalid},
		{name: "sigmoid different rank", run: func() error { return scope.Sigmoid(input, flat) }, want: ErrInvalid},
		{name: "sigmoid aliased view", run: func() error { return scope.Sigmoid(input, view) }, want: ErrInvalid},
		{name: "sigmoid uninitialized", run: func() error { return scope.Sigmoid(uninitialized, destination) }, want: ErrUninitialized},
		{name: "sigmoid closed input", run: func() error { return scope.Sigmoid(closed, destination) }, want: ErrClosed},

		{name: "tanh missing input", run: func() error { return scope.Tanh(nil, destination) }, want: ErrInvalid},
		{name: "tanh different rank", run: func() error { return scope.Tanh(input, flat) }, want: ErrInvalid},
		{name: "tanh aliased view", run: func() error { return scope.Tanh(input, view) }, want: ErrInvalid},
		{name: "tanh uninitialized", run: func() error { return scope.Tanh(uninitialized, destination) }, want: ErrUninitialized},
		{name: "tanh closed destination", run: func() error { return scope.Tanh(input, closed) }, want: ErrClosed},

		{name: "softmax missing input", run: func() error { return scope.Softmax(nil, destination) }, want: ErrInvalid},
		{name: "softmax zero handle", run: func() error { return scope.Softmax(&Tensor{}, destination) }, want: ErrInvalid},
		{name: "softmax missing destination", run: func() error { return scope.Softmax(input, nil) }, want: ErrInvalid},
		{name: "softmax different rank", run: func() error { return scope.Softmax(input, flat) }, want: ErrInvalid},
		{name: "softmax non-float32 input", run: func() error { return scope.Softmax(f16Input, destination) }, want: ErrInvalid},
		{name: "softmax non-float32 destination", run: func() error { return scope.Softmax(input, f16Dest) }, want: ErrInvalid},
		{name: "softmax foreign input", run: func() error { return scope.Softmax(foreign, destination) }, want: ErrInvalid},
		{name: "softmax foreign destination", run: func() error { return scope.Softmax(input, foreign) }, want: ErrInvalid},
		{name: "softmax aliased handle", run: func() error { return scope.Softmax(input, &alias) }, want: ErrInvalid},
		{name: "softmax aliased view", run: func() error { return scope.Softmax(input, view) }, want: ErrInvalid},
		{name: "softmax uninitialized", run: func() error { return scope.Softmax(uninitialized, destination) }, want: ErrUninitialized},
		{name: "softmax closed input", run: func() error { return scope.Softmax(closed, destination) }, want: ErrClosed},
		{name: "softmax closed destination", run: func() error { return scope.Softmax(input, closed) }, want: ErrClosed},

		{name: "softmax axis negative", run: func() error { return scope.SoftmaxAxis(input, destination, -1) }, want: ErrInvalid},
		{name: "softmax axis equal rank", run: func() error { return scope.SoftmaxAxis(input, destination, 3) }, want: ErrInvalid},
		{name: "softmax axis above rank", run: func() error { return scope.SoftmaxAxis(input, destination, 99) }, want: ErrInvalid},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requireError(t, tt.run(), tt.want)
		})
	}
	if backend.elementwise.Load() != 0 || backend.softmaxes.Load() != 0 {
		t.Fatalf("invalid operations reached backend: elementwise=%d softmax=%d",
			backend.elementwise.Load(), backend.softmaxes.Load())
	}

	// Valid operations succeed on a fresh scope.
	var valid = requireScope(t, device)
	requireError(t, valid.ReLU(input, destination), nil)
	requireError(t, valid.Sigmoid(input, destination), nil)
	requireError(t, valid.Tanh(input, destination), nil)
	requireError(t, valid.Softmax(input, destination), nil)
	requireError(t, valid.SoftmaxAxis(input, destination, 0), nil)
	requireError(t, valid.SoftmaxAxis(input, destination, 1), nil)
	requireError(t, valid.SoftmaxAxis(input, destination, 2), nil)
	if backend.elementwise.Load() != 3 || backend.softmaxes.Load() != 4 {
		t.Fatalf("unexpected backend call counts: elementwise=%d softmax=%d",
			backend.elementwise.Load(), backend.softmaxes.Load())
	}
	requireError(t, valid.Close(), nil)
	requireError(t, valid.ReLU(input, destination), ErrClosed)
	requireError(t, valid.Softmax(input, destination), ErrClosed)
	requireError(t, scope.Close(), nil)
	requireError(t, view.Close(), nil)
}

func Test_ScopeSoftmaxIndexLimit(t *testing.T) {
	if uint64(math.MaxInt)/4 <= math.MaxUint32 {
		t.Skip("host byte limit is below the Metal softmax index limit")
	}
	var (
		backend = fakeBackend{}
		device  = newDevice(&backend, math.MaxUint64)
		wide    = requireTensor(t, device, 1, int64(math.MaxUint32)+1)
		wideOut = requireTensor(t, device, 1, int64(math.MaxUint32)+1)
		atLimit = requireTensor(t, device, int64(math.MaxUint32))
		output  = requireTensor(t, device, int64(math.MaxUint32))
		scope   = requireScope(t, device)
	)
	requireError(t, scope.Softmax(wide, wideOut), ErrInvalid)
	requireError(t, scope.ReLU(wide, wideOut), ErrInvalid)
	if backend.softmaxes.Load() != 0 || backend.elementwise.Load() != 0 {
		t.Fatal("oversized operation reached the backend")
	}
	requireError(t, scope.Fill(atLimit, 1), nil)
	requireError(t, scope.Softmax(atLimit, output), nil)
	requireError(t, scope.ReLU(atLimit, output), nil)
	requireError(t, scope.Sigmoid(atLimit, output), nil)
	requireError(t, scope.Tanh(atLimit, output), nil)
	requireError(t, scope.Close(), nil)
}

func Test_ScopeSoftmaxFailures(t *testing.T) {
	var (
		backend fakeBackend
		device  = newDevice(&backend, 128)
		input   = requireTensor(t, device, 2, 3)
		output  = requireTensor(t, device, 2, 3)
		ctx     = context.Background()
		fault   = errors.New("injected softmax failure")
	)
	requireError(t, input.Upload([]float32{1, 2, 3, 4, 5, 6}), nil)
	requireError(t, output.Upload([]float32{0, 0, 0, 0, 0, 0}), nil)

	var encoding = requireScope(t, device)
	encoding.state.commands.(*fakeCommandBuffer).encodeErr = fault
	requireError(t, encoding.Softmax(input, output), fault)
	requireError(t, encoding.Submit(ctx), ErrInvalid)
	requireError(t, encoding.Close(), nil)

	var invalidAxis = requireScope(t, device)
	requireError(t, invalidAxis.SoftmaxAxis(input, output, 2), ErrInvalid)
	requireError(t, invalidAxis.Submit(ctx), ErrInvalid)
	requireError(t, invalidAxis.Close(), nil)
}
