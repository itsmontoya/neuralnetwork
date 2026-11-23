package gpu

import (
	"context"
	"errors"
	"math"
	"testing"
)

func Test_ScopeMatMulValidation(t *testing.T) {
	var (
		backend      fakeBackend
		device       = newDevice(&backend, 1024)
		other        = newDevice(&backend, 1024)
		left         = requireTensor(t, device, 2, 3)
		right        = requireTensor(t, device, 3, 4)
		transposed   = requireTensor(t, device, 4, 3)
		output       = requireTensor(t, device, 2, 4)
		wrongOutput  = requireTensor(t, device, 4, 2)
		vector       = requireTensor(t, device, 3)
		foreignLeft  = requireTensor(t, other, 2, 3)
		foreignRight = requireTensor(t, other, 3, 4)
		foreignOut   = requireTensor(t, other, 2, 4)
		scope        = requireScope(t, device)
	)
	requireError(t, left.Upload(make([]float32, 6)), nil)
	requireError(t, right.Upload(make([]float32, 12)), nil)
	var tests = []struct {
		name                string
		left, right, output *Tensor
		rightTransposed     bool
		want                error
	}{
		{name: "nil", right: right, output: output, want: ErrInvalid},
		{name: "zero handle", left: left, right: right, output: &Tensor{}, want: ErrInvalid},
		{name: "vector", left: vector, right: right, output: output, want: ErrInvalid},
		{name: "inner mismatch", left: left, right: transposed, output: output, want: ErrInvalid},
		{name: "transposed inner mismatch", left: left, right: right, output: output, rightTransposed: true, want: ErrInvalid},
		{name: "wrong output shape", left: left, right: right, output: wrongOutput, want: ErrInvalid},
		{name: "foreign left", left: foreignLeft, right: right, output: output, want: ErrInvalid},
		{name: "foreign right", left: left, right: foreignRight, output: output, want: ErrInvalid},
		{name: "foreign output", left: left, right: right, output: foreignOut, want: ErrInvalid},
		{name: "uninitialized right", left: left, right: transposed, output: output, rightTransposed: true, want: ErrUninitialized},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error
			if tt.rightTransposed {
				err = scope.MatMulRightTranspose(tt.left, tt.right, tt.output)
			} else {
				err = scope.MatMul(tt.left, tt.right, tt.output)
			}
			requireError(t, err, tt.want)
		})
	}
	var _, zeroErr = device.NewFloat32([]int64{2, 0})
	requireError(t, zeroErr, ErrInvalid)

	var (
		square = requireTensor(t, device, 2, 2)
		result = requireTensor(t, device, 2, 2)
		alias  = *square
	)
	requireError(t, square.Upload([]float32{1, 2, 3, 4}), nil)
	requireError(t, scope.MatMul(square, square, &alias), ErrInvalid)
	if backend.matmuls.Load() != 0 {
		t.Fatal("invalid matmul reached the backend")
	}
	requireError(t, scope.MatMul(square, &alias, result), nil)
	requireError(t, square.Close(), nil)
	requireError(t, scope.Submit(context.Background()), nil)
	requireError(t, scope.Wait(context.Background()), nil)
	requireError(t, scope.Close(), nil)

	var next = requireScope(t, device)
	requireError(t, next.MatMul(square, &alias, result), ErrClosed)
}

func Test_ScopeMatMulIndexLimit(t *testing.T) {
	if uint64(math.MaxInt)/4 <= math.MaxUint32 {
		t.Skip("host byte limit is below the Metal matmul index limit")
	}
	var (
		backend fakeBackend
		device  = newDevice(&backend, math.MaxUint64)
		wide    = requireTensor(t, device, 1, int64(math.MaxUint32)+1)
		product = requireTensor(t, device, 65536, 65536)
		one     = requireTensor(t, device, 1, 1)
		scope   = requireScope(t, device)
	)
	requireError(t, scope.MatMul(wide, one, one), ErrInvalid)
	requireError(t, scope.MatMulRightTranspose(one, product, one), ErrInvalid)
	requireError(t, scope.MatMul(one, one, product), ErrInvalid)
	if backend.matmuls.Load() != 0 {
		t.Fatal("oversized matmul reached the backend")
	}
}

func Test_ScopeMatMulFailures(t *testing.T) {
	var (
		backend fakeBackend
		device  = newDevice(&backend, 128)
		left    = requireTensor(t, device, 1, 1)
		weight  = requireTensor(t, device, 1, 1)
		output  = requireTensor(t, device, 1, 1)
		ctx     = context.Background()
		fault   = errors.New("injected matmul failure")
		value   [1]float32
	)
	requireError(t, left.Upload([]float32{3}), nil)
	requireError(t, weight.Upload([]float32{4}), nil)
	requireError(t, output.Upload([]float32{2}), nil)

	var badShape = requireScope(t, device)
	requireError(t, badShape.Fill(output, 5), nil)
	requireError(t, badShape.MatMul(left, nil, output), ErrInvalid)
	requireError(t, badShape.Submit(ctx), ErrInvalid)
	requireError(t, badShape.Close(), nil)

	var encoding = requireScope(t, device)
	requireError(t, encoding.Fill(output, 5), nil)
	encoding.state.commands.(*fakeCommandBuffer).encodeErr = fault
	requireError(t, encoding.MatMul(left, weight, output), fault)
	requireError(t, encoding.Submit(ctx), ErrInvalid)
	requireError(t, encoding.Close(), nil)
	requireError(t, output.DownloadInto(value[:]), nil)
	if value[0] != 2 || backend.submissions.Load() != 0 {
		t.Fatalf("discard changed output or submitted work: output=%g submissions=%d", value[0], backend.submissions.Load())
	}

	var execution = requireScope(t, device)
	execution.state.commands.(*fakeCommandBuffer).waitErr = fault
	requireError(t, execution.MatMulRightTranspose(left, weight, output), nil)
	requireError(t, execution.Submit(ctx), nil)
	requireError(t, execution.Wait(ctx), fault)
	requireError(t, output.DownloadInto(value[:]), ErrUninitialized)
	requireError(t, weight.DownloadInto(value[:]), nil)
	if value[0] != 4 {
		t.Fatalf("weight changed after failed submission: %g", value[0])
	}
	requireError(t, execution.Close(), nil)
	var retry = requireScope(t, device)
	requireError(t, retry.MatMulRightTranspose(left, weight, output), nil)
	requireError(t, retry.Submit(ctx), nil)
	requireError(t, retry.Close(), nil)
	requireError(t, output.DownloadInto(value[:]), nil)
}
