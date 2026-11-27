package gpu

import (
	"context"
	"errors"
	"slices"
	"testing"
)

func Test_Scope_Concat(t *testing.T) {
	t.Run("valid_all_dense_dtypes_and_duplicate_inputs", func(t *testing.T) {
		var (
			backend = &fakeBackend{}
			dev     = newDevice(backend, ^uint64(0))
			ctx     = context.Background()
			dtypes  = []DType{Float32, Float16, BFloat16}
		)
		for _, dt := range dtypes {
			var (
				a   = mustNewTensorWithDType(t, dev, dt, 2, 2, 3)
				b   = mustNewTensorWithDType(t, dev, dt, 2, 1, 3)
				dst = mustNewTensorWithDType(t, dev, dt, 2, 5, 3)
			)
			if dt == Float32 {
				requireError(t, a.Upload([]float32{
					1, 2, 3, 4, 5, 6,
					7, 8, 9, 10, 11, 12,
				}), nil)
				requireError(t, b.Upload([]float32{
					100, 101, 102,
					200, 201, 202,
				}), nil)
			} else {
				requireError(t, a.UploadBits([]uint16{
					1, 2, 3, 4, 5, 6,
					7, 8, 9, 10, 11, 12,
				}), nil)
				requireError(t, b.UploadBits([]uint16{
					100, 101, 102,
					200, 201, 202,
				}), nil)
			}

			var (
				before     = backend.concats.Load()
				scope, err = dev.NewScope()
			)
			requireError(t, err, nil)
			// Duplicate read input [a, b, a] must share a single read lease cleanly.
			requireError(t, scope.Concat([]*Tensor{a, b, a}, dst, 1), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)
			if backend.concats.Load() != before+1 {
				t.Fatalf("%s expected 1 concat dispatch, got %d", dt, backend.concats.Load()-before)
			}

			if dt == Float32 {
				var got = make([]float32, 30)
				requireError(t, dst.DownloadInto(got), nil)
				var want = []float32{
					1, 2, 3, 4, 5, 6, 100, 101, 102, 1, 2, 3, 4, 5, 6,
					7, 8, 9, 10, 11, 12, 200, 201, 202, 7, 8, 9, 10, 11, 12,
				}
				if !slices.Equal(got, want) {
					t.Fatalf("Float32 concat mismatch:\ngot:  %v\nwant: %v", got, want)
				}
			} else {
				var got = make([]uint16, 30)
				requireError(t, dst.DownloadBitsInto(got), nil)
				var want = []uint16{
					1, 2, 3, 4, 5, 6, 100, 101, 102, 1, 2, 3, 4, 5, 6,
					7, 8, 9, 10, 11, 12, 200, 201, 202, 7, 8, 9, 10, 11, 12,
				}
				if !slices.Equal(got, want) {
					t.Fatalf("%s concat mismatch:\ngot:  %v\nwant: %v", dt, got, want)
				}
			}

			for _, tensor := range []*Tensor{a, b, dst} {
				requireError(t, tensor.Close(), nil)
			}
		}
		requireError(t, dev.Close(), nil)
	})

	t.Run("validation_errors", func(t *testing.T) {
		var (
			backend  = &fakeBackend{}
			dev      = newDevice(backend, ^uint64(0))
			otherDev = newDevice(&fakeBackend{}, ^uint64(0))

			a        = mustNewTensorWithDType(t, dev, Float32, 2, 3)
			b        = mustNewTensorWithDType(t, dev, Float32, 2, 4)
			dst      = mustNewTensorWithDType(t, dev, Float32, 2, 7)
			dstShort = mustNewTensorWithDType(t, dev, Float32, 2, 6)
			dstLong  = mustNewTensorWithDType(t, dev, Float32, 2, 8)
			badDim   = mustNewTensorWithDType(t, dev, Float32, 3, 4)
			rank3    = mustNewTensorWithDType(t, dev, Float32, 1, 2, 4)
			rank7In  = mustNewTensorWithDType(t, dev, Float32, 1, 1, 1, 1, 1, 1, 2)
			rank7Dst = mustNewTensorWithDType(t, dev, Float32, 1, 1, 1, 1, 1, 1, 4)
			f16      = mustNewTensorWithDType(t, dev, Float16, 2, 4)
			q8In     = mustNewTensorWithDType(t, dev, Q8_0, 1, 32)
			q8Dst    = mustNewTensorWithDType(t, dev, Q8_0, 2, 32)
			q4In     = mustNewTensorWithDType(t, dev, Q4_0, 1, 32)
			q4Dst    = mustNewTensorWithDType(t, dev, Q4_0, 2, 32)
			q41In    = mustNewTensorWithDType(t, dev, Q4_1, 1, 32)
			q41Dst   = mustNewTensorWithDType(t, dev, Q4_1, 2, 32)
			uninit   = mustNewTensorWithDType(t, dev, Float32, 2, 4)
			closedIn = mustNewTensorWithDType(t, dev, Float32, 2, 4)
			otherIn  = mustNewTensorWithDType(t, otherDev, Float32, 2, 4)
			hugeIn   = mustNewTensorWithDType(t, dev, Float32, 1<<20, 4096)
			hugeDst  = mustNewTensorWithDType(t, dev, Float32, 1<<20, 8192)
		)
		requireError(t, a.Upload(make([]float32, 6)), nil)
		requireError(t, b.Upload(make([]float32, 8)), nil)
		requireError(t, badDim.Upload(make([]float32, 12)), nil)
		requireError(t, rank3.Upload(make([]float32, 8)), nil)
		requireError(t, rank7In.Upload(make([]float32, 2)), nil)
		requireError(t, f16.UploadBits(make([]uint16, 8)), nil)
		requireError(t, q8In.UploadBytes(make([]byte, QuantBlockBytesQ8_0)), nil)
		requireError(t, q4In.UploadBytes(make([]byte, QuantBlockBytesQ4_0)), nil)
		requireError(t, q41In.UploadBytes(make([]byte, QuantBlockBytesQ4_1)), nil)
		requireError(t, closedIn.Upload(make([]float32, 8)), nil)
		requireError(t, closedIn.Close(), nil)
		requireError(t, otherIn.Upload(make([]float32, 8)), nil)

		var dstView, viewErr = dst.View([]int64{2, 7})
		requireError(t, viewErr, nil)
		var aliasIn = mustNewTensorWithDType(t, dev, Float32, 2, 7)
		requireError(t, aliasIn.Upload(make([]float32, 14)), nil)
		var aliasView, aliasViewErr = aliasIn.View([]int64{2, 7})
		requireError(t, aliasViewErr, nil)

		type testcase struct {
			name   string
			inputs []*Tensor
			dst    *Tensor
			axis   int
			want   error
		}
		var tests = []testcase{
			{name: "empty_inputs", inputs: nil, dst: dst, axis: 1, want: ErrInvalid},
			{name: "empty_slice", inputs: []*Tensor{}, dst: dst, axis: 1, want: ErrInvalid},
			{name: "nil_destination", inputs: []*Tensor{a, b}, dst: nil, axis: 1, want: ErrInvalid},
			{name: "nil_input", inputs: []*Tensor{a, nil}, dst: dst, axis: 1, want: ErrInvalid},
			{name: "negative_axis", inputs: []*Tensor{a, b}, dst: dst, axis: -1, want: ErrInvalid},
			{name: "axis_equals_rank", inputs: []*Tensor{a, b}, dst: dst, axis: 2, want: ErrInvalid},
			{name: "dtype_mismatch", inputs: []*Tensor{a, f16}, dst: dst, axis: 1, want: ErrInvalid},
			{name: "rank_mismatch", inputs: []*Tensor{a, rank3}, dst: dst, axis: 1, want: ErrInvalid},
			{name: "rank_exceeds_6", inputs: []*Tensor{rank7In, rank7In}, dst: rank7Dst, axis: 6, want: ErrInvalid},
			{name: "non_axis_dim_mismatch", inputs: []*Tensor{a, badDim}, dst: dst, axis: 1, want: ErrInvalid},
			{name: "axis_sum_exceeds_dst", inputs: []*Tensor{a, b}, dst: dstShort, axis: 1, want: ErrInvalid},
			{name: "axis_sum_short_of_dst", inputs: []*Tensor{a, b}, dst: dstLong, axis: 1, want: ErrInvalid},
			{name: "quantized_q8_0", inputs: []*Tensor{q8In, q8In}, dst: q8Dst, axis: 0, want: ErrInvalid},
			{name: "quantized_q4_0", inputs: []*Tensor{q4In, q4In}, dst: q4Dst, axis: 0, want: ErrInvalid},
			{name: "quantized_q4_1", inputs: []*Tensor{q41In, q41In}, dst: q41Dst, axis: 0, want: ErrInvalid},
			{name: "direct_alias", inputs: []*Tensor{aliasIn}, dst: aliasIn, axis: 0, want: ErrInvalid},
			{name: "view_alias", inputs: []*Tensor{aliasView}, dst: aliasIn, axis: 0, want: ErrInvalid},
			{name: "dst_view_alias", inputs: []*Tensor{aliasIn}, dst: aliasView, axis: 0, want: ErrInvalid},
			{name: "uninitialized_input", inputs: []*Tensor{a, uninit}, dst: dst, axis: 1, want: ErrUninitialized},
			{name: "closed_input", inputs: []*Tensor{a, closedIn}, dst: dst, axis: 1, want: ErrClosed},
			{name: "other_device_input", inputs: []*Tensor{a, otherIn}, dst: dst, axis: 1, want: ErrInvalid},
			{name: "uint32_overflow", inputs: []*Tensor{hugeIn, hugeIn}, dst: hugeDst, axis: 1, want: ErrInvalid},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var scope, err = dev.NewScope()
				requireError(t, err, nil)
				err = scope.Concat(tt.inputs, tt.dst, tt.axis)
				if !errors.Is(err, tt.want) {
					t.Fatalf("Concat() error = %v, want %v", err, tt.want)
				}
				requireError(t, scope.Close(), nil)
			})
		}

		// Closed scope returns ErrClosed.
		var closedScope, err = dev.NewScope()
		requireError(t, err, nil)
		requireError(t, closedScope.Close(), nil)
		requireError(t, closedScope.Concat([]*Tensor{a, b}, dst, 1), ErrClosed)

		for _, tensor := range []*Tensor{
			a, b, dst, dstView, dstShort, dstLong, badDim, rank3, rank7In, rank7Dst,
			f16, q8In, q8Dst, q4In, q4Dst, q41In, q41Dst, uninit, aliasIn, aliasView,
			hugeIn, hugeDst,
		} {
			requireError(t, tensor.Close(), nil)
		}
		requireError(t, otherIn.Close(), nil)
		requireError(t, dev.Close(), nil)
		requireError(t, otherDev.Close(), nil)
	})
}
