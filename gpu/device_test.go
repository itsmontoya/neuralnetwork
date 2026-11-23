package gpu

import (
	"errors"
	"math"
	"testing"
)

func Test_DeviceNewFloat32Shape(t *testing.T) {
	var (
		b     fakeBackend
		d     = newDevice(&b, 128)
		tests = []struct {
			name  string
			shape []int64
		}{
			{name: "nil shape"},
			{name: "empty shape", shape: []int64{}},
			{name: "zero dimension", shape: []int64{1, 0}},
			{name: "negative dimension", shape: []int64{2, -1}},
			{name: "over device length", shape: []int64{33}},
			{name: "element overflow", shape: []int64{math.MaxInt64, 4}},
			{name: "byte overflow", shape: []int64{math.MaxInt64 / 2}},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out, err = d.NewFloat32(tt.shape)
			if out != nil || !errors.Is(err, ErrInvalid) {
				t.Fatalf("NewFloat32(%v) = %v, %v, want nil and ErrInvalid", tt.shape, out, err)
			}
		})
	}

	var tensor = requireTensor(t, d, 4, 8)
	requireError(t, d.Close(), ErrBusy)
	requireError(t, tensor.Close(), nil)
	requireError(t, d.Close(), nil)
	requireError(t, d.Close(), nil)
	var _, tensorErr = d.NewFloat32([]int64{1})
	requireError(t, tensorErr, ErrClosed)
	var _, scopeErr = d.NewScope()
	requireError(t, scopeErr, ErrClosed)
}

func Test_DeviceNativeCreationFailures(t *testing.T) {
	var (
		fault = errors.New("injected creation failure")
		b     = fakeBackend{bufferErr: fault, scopeErr: fault}
		d     = newDevice(&b, 128)
	)
	var buffer, tensorErr = d.NewFloat32([]int64{1})
	requireError(t, tensorErr, fault)
	if buffer != nil {
		t.Fatal("failed allocation returned a tensor")
	}
	var scope, scopeErr = d.NewScope()
	requireError(t, scopeErr, fault)
	if scope != nil {
		t.Fatal("failed allocation returned a scope")
	}
	requireError(t, d.Close(), nil)
}

func Test_DeviceHostByteOverflow(t *testing.T) {
	var b fakeBackend
	var d = newDevice(&b, math.MaxUint64)
	var _, err = d.NewFloat32([]int64{int64(math.MaxInt/4) + 1})
	requireError(t, err, ErrInvalid)
	requireError(t, d.Close(), nil)
}

func requireError(t *testing.T, err, want error) {
	t.Helper()
	if !errors.Is(err, want) {
		t.Fatalf("error = %v, want %v", err, want)
	}
}

func requireTensor(t *testing.T, d *Device, shape ...int64) (out *Tensor) {
	t.Helper()
	var tensor, err = d.NewFloat32(shape)
	requireError(t, err, nil)
	t.Cleanup(func() { requireError(t, tensor.Close(), nil) })
	return tensor
}

func requireMetalDevice(t *testing.T) (out *Device) {
	t.Helper()
	var d, err = OpenDefault()
	if errors.Is(err, ErrUnavailable) {
		t.Skip("default Metal device unavailable")
	}
	requireError(t, err, nil)
	t.Cleanup(func() { requireError(t, d.Close(), nil) })
	return d
}

func requireScope(t *testing.T, d *Device) (out *Scope) {
	t.Helper()
	var scope, err = d.NewScope()
	requireError(t, err, nil)
	t.Cleanup(func() { requireError(t, scope.Close(), nil) })
	return scope
}
