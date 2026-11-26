package gpu

import (
	"context"
	"math"
	"testing"
)

func Test_Scope_GELU(t *testing.T) {
	var ops = []struct {
		name   string
		encode func(*Scope, *Tensor, *Tensor) error
	}{
		{name: "GELU", encode: (*Scope).GELU},
		{name: "GELUTanh", encode: (*Scope).GELUTanh},
	}

	t.Run("valid_shapes", func(t *testing.T) {
		var (
			backend = &fakeBackend{}
			dev     = newDevice(backend, ^uint64(0))
			ctx     = context.Background()
			shapes  = [][]int64{{8}, {2, 8}, {2, 3, 8}, {2, 2, 3, 8}}
		)
		for _, shape := range shapes {
			var (
				input = mustNewTensor(t, dev, shape...)
				dst   = mustNewTensor(t, dev, shape...)
			)
			requireError(t, input.Upload(make([]float32, input.state.storage.count)), nil)
			for _, op := range ops {
				var scope, err = dev.NewScope()
				requireError(t, err, nil)
				requireError(t, op.encode(scope, input, dst), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)
			}
			requireError(t, input.Close(), nil)
			requireError(t, dst.Close(), nil)
		}
		if got := backend.elementwise.Load(); got != int32(len(shapes)*len(ops)) {
			t.Fatalf("expected %d elementwise dispatches, got %d", len(shapes)*len(ops), got)
		}
		requireError(t, dev.Close(), nil)
	})

	t.Run("validation_errors", func(t *testing.T) {
		var (
			dev       = newDevice(&fakeBackend{}, ^uint64(0))
			otherDev  = newDevice(&fakeBackend{}, ^uint64(0))
			input     = mustNewTensor(t, dev, 2, 16)
			dst       = mustNewTensor(t, dev, 2, 16)
			wrongDst  = mustNewTensor(t, dev, 2, 8)
			f16In     = mustNewTensorWithDType(t, dev, Float16, 2, 16)
			f16Dst    = mustNewTensorWithDType(t, dev, Float16, 2, 16)
			uninitIn  = mustNewTensor(t, dev, 2, 16)
			otherIn   = mustNewTensor(t, otherDev, 2, 16)
			closedIn  = mustNewTensor(t, dev, 2, 16)
			largeIn   = mustNewTensor(t, dev, 1<<17, 1<<16)
			largeDst  = mustNewTensor(t, dev, 1<<17, 1<<16)
			inputView *Tensor
			err       error
		)
		defer func() {
			for _, tensor := range []*Tensor{
				input, dst, wrongDst, f16In, f16Dst, uninitIn, otherIn, largeIn, largeDst, inputView,
			} {
				if tensor != nil {
					_ = tensor.Close()
				}
			}
			_ = dev.Close()
			_ = otherDev.Close()
		}()
		requireError(t, input.Upload(make([]float32, 32)), nil)
		requireError(t, f16In.UploadFloat32As(make([]float32, 32)), nil)
		requireError(t, otherIn.Upload(make([]float32, 32)), nil)
		requireError(t, closedIn.Upload(make([]float32, 32)), nil)
		requireError(t, closedIn.Close(), nil)
		inputView, err = input.View([]int64{2, 16})
		requireError(t, err, nil)

		type testcase struct {
			name string
			in   *Tensor
			dst  *Tensor
			want error
		}
		var tests = []testcase{
			{name: "nil_input", in: nil, dst: dst, want: ErrInvalid},
			{name: "nil_destination", in: input, dst: nil, want: ErrInvalid},
			{name: "shape_mismatch", in: input, dst: wrongDst, want: ErrInvalid},
			{name: "non_float32_input", in: f16In, dst: dst, want: ErrInvalid},
			{name: "non_float32_dst", in: input, dst: f16Dst, want: ErrInvalid},
			{name: "direct_alias", in: input, dst: input, want: ErrInvalid},
			{name: "view_alias", in: input, dst: inputView, want: ErrInvalid},
			{name: "uninitialized_input", in: uninitIn, dst: dst, want: ErrUninitialized},
			{name: "other_device", in: otherIn, dst: dst, want: ErrInvalid},
			{name: "closed_input", in: closedIn, dst: dst, want: ErrClosed},
			{name: "exceeds_uint32", in: largeIn, dst: largeDst, want: ErrInvalid},
		}

		for _, op := range ops {
			for _, tc := range tests {
				t.Run(op.name+"/"+tc.name, func(t *testing.T) {
					var scope, scopeErr = dev.NewScope()
					requireError(t, scopeErr, nil)
					defer scope.Close()
					requireError(t, op.encode(scope, tc.in, tc.dst), tc.want)
				})
			}
		}

		// Verify uint32 upper bound constant matches expectation.
		if largeIn.state.storage.count <= math.MaxUint32 {
			t.Fatalf("expected largeIn count > MaxUint32")
		}
	})
}
