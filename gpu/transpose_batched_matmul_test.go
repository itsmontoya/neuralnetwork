package gpu

import (
	"context"
	"errors"
	"testing"
)

func Test_Scope_Transpose(t *testing.T) {
	t.Run("valid_identity_and_non_identity_all_dense_dtypes", func(t *testing.T) {
		var (
			backend = &fakeBackend{}
			dev     = newDevice(backend, ^uint64(0))
			ctx     = context.Background()
			dtypes  = []DType{Float32, Float16, BFloat16}
		)
		for _, dt := range dtypes {
			// Identity permutation fast-paths to copy.
			var (
				in1  = mustNewTensorWithDType(t, dev, dt, 2, 3, 4)
				dst1 = mustNewTensorWithDType(t, dev, dt, 2, 3, 4)
			)
			requireError(t, in1.UploadFloat32As(make([]float32, 24)), nil)
			var (
				beforeCopies     = backend.copies.Load()
				beforeTransposes = backend.transposes.Load()
				scope, err       = dev.NewScope()
			)
			requireError(t, err, nil)
			requireError(t, scope.Transpose(in1, dst1, []int{0, 1, 2}), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)
			if backend.copies.Load() != beforeCopies+1 || backend.transposes.Load() != beforeTransposes {
				t.Fatalf("%s identity transpose expected copy fast path", dt)
			}

			// Size-1 axis swap also fast-paths to copy.
			var dst1b = mustNewTensorWithDType(t, dev, dt, 2, 4, 1)
			var in1b = mustNewTensorWithDType(t, dev, dt, 2, 1, 4)
			requireError(t, in1b.UploadFloat32As(make([]float32, 8)), nil)
			beforeCopies = backend.copies.Load()
			beforeTransposes = backend.transposes.Load()
			scope, err = dev.NewScope()
			requireError(t, err, nil)
			requireError(t, scope.Transpose(in1b, dst1b, []int{0, 2, 1}), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)
			if backend.copies.Load() != beforeCopies+1 || backend.transposes.Load() != beforeTransposes {
				t.Fatalf("%s size-1 transpose expected copy fast path", dt)
			}

			// Non-identity 4D transpose uses transpose kernel.
			var (
				in4  = mustNewTensorWithDType(t, dev, dt, 2, 3, 4, 5)
				dst4 = mustNewTensorWithDType(t, dev, dt, 2, 4, 3, 5)
			)
			requireError(t, in4.UploadFloat32As(make([]float32, 120)), nil)
			beforeTransposes = backend.transposes.Load()
			scope, err = dev.NewScope()
			requireError(t, err, nil)
			requireError(t, scope.Transpose(in4, dst4, []int{0, 2, 1, 3}), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)
			if backend.transposes.Load() != beforeTransposes+1 {
				t.Fatalf("%s non-identity transpose expected transpose dispatch", dt)
			}

			for _, tensor := range []*Tensor{in1, dst1, in1b, dst1b, in4, dst4} {
				requireError(t, tensor.Close(), nil)
			}
		}
		requireError(t, dev.Close(), nil)
	})

	t.Run("validation_errors", func(t *testing.T) {
		var (
			backend = &fakeBackend{}
			dev     = newDevice(backend, ^uint64(0))
			other   = newDevice(&fakeBackend{}, ^uint64(0))
			in2     = mustNewTensor(t, dev, 2, 3)
			dst2    = mustNewTensor(t, dev, 3, 2)
			in7     = mustNewTensor(t, dev, 1, 1, 1, 1, 1, 1, 2)
			dst7    = mustNewTensor(t, dev, 1, 1, 1, 1, 1, 1, 2)
			f16     = mustNewTensorWithDType(t, dev, Float16, 3, 2)
			q8In    = mustNewTensorWithDType(t, dev, Q8_0, 32, 32)
			q8Dst   = mustNewTensorWithDType(t, dev, Q8_0, 32, 32)
			uninit  = mustNewTensor(t, dev, 2, 3)
			foreign = mustNewTensor(t, other, 3, 2)
		)
		requireError(t, in2.Upload(make([]float32, 6)), nil)
		requireError(t, in7.Upload(make([]float32, 2)), nil)
		requireError(t, f16.UploadFloat32As(make([]float32, 6)), nil)
		var view, err = in2.View([]int64{3, 2})
		requireError(t, err, nil)

		type testcase struct {
			name string
			in   *Tensor
			dst  *Tensor
			perm []int
			want error
		}
		var tests = []testcase{
			{name: "nil_input", in: nil, dst: dst2, perm: []int{1, 0}, want: ErrInvalid},
			{name: "nil_dest", in: in2, dst: nil, perm: []int{1, 0}, want: ErrInvalid},
			{name: "dtype_mismatch", in: in2, dst: f16, perm: []int{1, 0}, want: ErrInvalid},
			{name: "quantized", in: q8In, dst: q8Dst, perm: []int{1, 0}, want: ErrInvalid},
			{name: "rank_over_6", in: in7, dst: dst7, perm: []int{0, 1, 2, 3, 4, 5, 6}, want: ErrInvalid},
			{name: "perm_short", in: in2, dst: dst2, perm: []int{0}, want: ErrInvalid},
			{name: "perm_negative", in: in2, dst: dst2, perm: []int{-1, 0}, want: ErrInvalid},
			{name: "perm_oob", in: in2, dst: dst2, perm: []int{0, 2}, want: ErrInvalid},
			{name: "perm_duplicate", in: in2, dst: dst2, perm: []int{0, 0}, want: ErrInvalid},
			{name: "shape_mismatch", in: in2, dst: dst2, perm: []int{0, 1}, want: ErrInvalid},
			{name: "alias_same", in: in2, dst: in2, perm: []int{0, 1}, want: ErrInvalid},
			{name: "alias_view", in: in2, dst: view, perm: []int{1, 0}, want: ErrInvalid},
			{name: "uninitialized", in: uninit, dst: dst2, perm: []int{1, 0}, want: ErrUninitialized},
			{name: "different_device", in: in2, dst: foreign, perm: []int{1, 0}, want: ErrInvalid},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var scope, scopeErr = dev.NewScope()
				requireError(t, scopeErr, nil)
				var callErr = scope.Transpose(tt.in, tt.dst, tt.perm)
				if !errors.Is(callErr, tt.want) {
					t.Fatalf("got %v, want %v", callErr, tt.want)
				}
				requireError(t, scope.Close(), nil)
			})
		}

		for _, tensor := range []*Tensor{view, in2, dst2, in7, dst7, f16, q8In, q8Dst, uninit} {
			requireError(t, tensor.Close(), nil)
		}
		requireError(t, foreign.Close(), nil)
		requireError(t, dev.Close(), nil)
		requireError(t, other.Close(), nil)
	})
}

func Test_Scope_MatMulLeftTranspose(t *testing.T) {
	t.Run("valid_dtypes_and_shared_inputs", func(t *testing.T) {
		var (
			backend = &fakeBackend{}
			dev     = newDevice(backend, ^uint64(0))
			ctx     = context.Background()
			left    = mustNewTensor(t, dev, 4, 3)
			dst     = mustNewTensor(t, dev, 3, 5)
		)
		requireError(t, left.Upload(make([]float32, 12)), nil)
		for _, dt := range []DType{Float32, Float16, BFloat16} {
			var right = mustNewTensorWithDType(t, dev, dt, 4, 5)
			requireError(t, right.UploadFloat32As(make([]float32, 20)), nil)
			var scope, err = dev.NewScope()
			requireError(t, err, nil)
			requireError(t, scope.MatMulLeftTranspose(left, right, dst), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)
			requireError(t, right.Close(), nil)
		}
		// left == right (A^T * A) is allowed.
		var (
			sqDst      = mustNewTensor(t, dev, 3, 3)
			scope, err = dev.NewScope()
		)
		requireError(t, err, nil)
		requireError(t, scope.MatMulLeftTranspose(left, left, sqDst), nil)
		requireError(t, scope.Submit(ctx), nil)
		requireError(t, scope.Wait(ctx), nil)
		requireError(t, scope.Close(), nil)
		if backend.matMulLeftTranspose.Load() != 4 {
			t.Fatalf("got %d dispatches, want 4", backend.matMulLeftTranspose.Load())
		}
		for _, tensor := range []*Tensor{left, dst, sqDst} {
			requireError(t, tensor.Close(), nil)
		}
		requireError(t, dev.Close(), nil)
	})

	t.Run("validation_errors", func(t *testing.T) {
		var (
			backend = &fakeBackend{}
			dev     = newDevice(backend, ^uint64(0))
			left    = mustNewTensor(t, dev, 4, 3)
			right   = mustNewTensor(t, dev, 4, 5)
			dst     = mustNewTensor(t, dev, 3, 5)
			badK    = mustNewTensor(t, dev, 5, 5)
			badDst  = mustNewTensor(t, dev, 4, 5)
			rank3   = mustNewTensor(t, dev, 1, 4, 3)
			f16Left = mustNewTensorWithDType(t, dev, Float16, 4, 3)
			f16Dst  = mustNewTensorWithDType(t, dev, Float16, 3, 5)
			q8Right = mustNewTensorWithDType(t, dev, Q8_0, 4, 32)
			q8Dst   = mustNewTensor(t, dev, 3, 32)
		)
		requireError(t, left.Upload(make([]float32, 12)), nil)
		requireError(t, right.Upload(make([]float32, 20)), nil)
		requireError(t, badK.Upload(make([]float32, 25)), nil)
		requireError(t, rank3.Upload(make([]float32, 12)), nil)
		requireError(t, f16Left.UploadFloat32As(make([]float32, 12)), nil)
		var viewDst, err = left.View([]int64{3, 4})
		requireError(t, err, nil)
		var right4 = mustNewTensor(t, dev, 4, 4)
		requireError(t, right4.Upload(make([]float32, 16)), nil)

		type testcase struct {
			name string
			l    *Tensor
			r    *Tensor
			d    *Tensor
			want error
		}
		var tests = []testcase{
			{name: "nil_left", l: nil, r: right, d: dst, want: ErrInvalid},
			{name: "rank3", l: rank3, r: right, d: dst, want: ErrInvalid},
			{name: "f16_left", l: f16Left, r: right, d: dst, want: ErrInvalid},
			{name: "f16_dst", l: left, r: right, d: f16Dst, want: ErrInvalid},
			{name: "quantized_right", l: left, r: q8Right, d: q8Dst, want: ErrInvalid},
			{name: "k_mismatch", l: left, r: badK, d: dst, want: ErrInvalid},
			{name: "dst_mismatch", l: left, r: right, d: badDst, want: ErrInvalid},
			{name: "view_alias", l: left, r: right4, d: viewDst, want: ErrInvalid},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var scope, scopeErr = dev.NewScope()
				requireError(t, scopeErr, nil)
				var callErr = scope.MatMulLeftTranspose(tt.l, tt.r, tt.d)
				if !errors.Is(callErr, tt.want) {
					t.Fatalf("got %v, want %v", callErr, tt.want)
				}
				requireError(t, scope.Close(), nil)
			})
		}
		for _, tensor := range []*Tensor{viewDst, right4, left, right, dst, badK, badDst, rank3, f16Left, f16Dst, q8Right, q8Dst} {
			requireError(t, tensor.Close(), nil)
		}
		requireError(t, dev.Close(), nil)
	})
}

func Test_Scope_BatchedMatMul(t *testing.T) {
	t.Run("valid_ranks_broadcasts_and_dtypes", func(t *testing.T) {
		var (
			backend = &fakeBackend{}
			dev     = newDevice(backend, ^uint64(0))
			ctx     = context.Background()
		)
		type testcase struct {
			name       string
			leftShape  []int64
			rightShape []int64
			dstShape   []int64
			rightT     bool
		}
		var tests = []testcase{
			{name: "2d_x_2d", leftShape: []int64{3, 4}, rightShape: []int64{4, 5}, dstShape: []int64{3, 5}},
			{name: "3d_x_2d_shared_weight", leftShape: []int64{2, 3, 4}, rightShape: []int64{4, 5}, dstShape: []int64{2, 3, 5}},
			{name: "2d_x_3d", leftShape: []int64{3, 4}, rightShape: []int64{2, 4, 5}, dstShape: []int64{2, 3, 5}},
			{name: "4d_attention", leftShape: []int64{2, 4, 6, 8}, rightShape: []int64{2, 4, 8, 6}, dstShape: []int64{2, 4, 6, 6}},
			{name: "4d_attention_rt", leftShape: []int64{2, 4, 6, 8}, rightShape: []int64{2, 4, 6, 8}, dstShape: []int64{2, 4, 6, 6}, rightT: true},
			{name: "4d_multi_broadcast", leftShape: []int64{2, 1, 3, 4}, rightShape: []int64{1, 5, 4, 6}, dstShape: []int64{2, 5, 3, 6}},
			{name: "6d_broadcast", leftShape: []int64{2, 1, 2, 1, 3, 4}, rightShape: []int64{1, 2, 1, 3, 4, 5}, dstShape: []int64{2, 2, 2, 3, 3, 5}},
			{name: "6d_broadcast_rt", leftShape: []int64{2, 1, 2, 1, 3, 4}, rightShape: []int64{1, 2, 1, 3, 5, 4}, dstShape: []int64{2, 2, 2, 3, 3, 5}, rightT: true},
		}
		for _, tt := range tests {
			for _, dt := range []DType{Float32, Float16, BFloat16} {
				var (
					left  = mustNewTensor(t, dev, tt.leftShape...)
					right = mustNewTensorWithDType(t, dev, dt, tt.rightShape...)
					dst   = mustNewTensor(t, dev, tt.dstShape...)
				)
				requireError(t, left.Upload(make([]float32, left.state.storage.count)), nil)
				requireError(t, right.UploadFloat32As(make([]float32, right.state.storage.count)), nil)
				var scope, err = dev.NewScope()
				requireError(t, err, nil)
				if tt.rightT {
					requireError(t, scope.BatchedMatMulRightTranspose(left, right, dst), nil)
				} else {
					requireError(t, scope.BatchedMatMul(left, right, dst), nil)
				}
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)
				for _, tensor := range []*Tensor{left, right, dst} {
					requireError(t, tensor.Close(), nil)
				}
			}
		}
		if backend.batchedMatMuls.Load() != int32(len(tests)*3) {
			t.Fatalf("got %d batched matmul dispatches, want %d", backend.batchedMatMuls.Load(), len(tests)*3)
		}
		requireError(t, dev.Close(), nil)
	})

	t.Run("validation_errors", func(t *testing.T) {
		var (
			backend  = &fakeBackend{}
			dev      = newDevice(backend, ^uint64(0))
			left     = mustNewTensor(t, dev, 2, 3, 4)
			right    = mustNewTensor(t, dev, 2, 4, 5)
			dst      = mustNewTensor(t, dev, 2, 3, 5)
			rank1    = mustNewTensor(t, dev, 4)
			rank7    = mustNewTensor(t, dev, 1, 1, 1, 1, 1, 3, 4)
			badBatch = mustNewTensor(t, dev, 3, 4, 5)
			badK     = mustNewTensor(t, dev, 2, 6, 5)
			badDst   = mustNewTensor(t, dev, 2, 3, 6)
			q4Right  = mustNewTensorWithDType(t, dev, Q4_0, 5, 32)
			left32   = mustNewTensor(t, dev, 2, 3, 32)
		)
		requireError(t, left.Upload(make([]float32, left.state.storage.count)), nil)
		requireError(t, right.Upload(make([]float32, right.state.storage.count)), nil)
		requireError(t, rank1.Upload(make([]float32, rank1.state.storage.count)), nil)
		requireError(t, rank7.Upload(make([]float32, rank7.state.storage.count)), nil)
		requireError(t, badBatch.Upload(make([]float32, badBatch.state.storage.count)), nil)
		requireError(t, badK.Upload(make([]float32, badK.state.storage.count)), nil)
		requireError(t, left32.Upload(make([]float32, left32.state.storage.count)), nil)
		var viewDst, err = left.View([]int64{2, 3, 4})
		requireError(t, err, nil)
		var sqRight = mustNewTensor(t, dev, 2, 4, 4)
		requireError(t, sqRight.Upload(make([]float32, sqRight.state.storage.count)), nil)

		type testcase struct {
			name   string
			l      *Tensor
			r      *Tensor
			d      *Tensor
			rightT bool
			want   error
		}
		var tests = []testcase{
			{name: "rank1", l: rank1, r: right, d: dst, want: ErrInvalid},
			{name: "rank7", l: rank7, r: right, d: dst, want: ErrInvalid},
			{name: "batch_mismatch", l: left, r: badBatch, d: dst, want: ErrInvalid},
			{name: "k_mismatch", l: left, r: badK, d: dst, want: ErrInvalid},
			{name: "dst_mismatch", l: left, r: right, d: badDst, want: ErrInvalid},
			{name: "quantized_right_rt", l: left32, r: q4Right, d: dst, rightT: true, want: ErrInvalid},
			{name: "view_alias", l: left, r: sqRight, d: viewDst, want: ErrInvalid},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var scope, scopeErr = dev.NewScope()
				requireError(t, scopeErr, nil)
				var callErr error
				if tt.rightT {
					callErr = scope.BatchedMatMulRightTranspose(tt.l, tt.r, tt.d)
				} else {
					callErr = scope.BatchedMatMul(tt.l, tt.r, tt.d)
				}
				if !errors.Is(callErr, tt.want) {
					t.Fatalf("got %v, want %v", callErr, tt.want)
				}
				requireError(t, scope.Close(), nil)
			})
		}
		for _, tensor := range []*Tensor{viewDst, sqRight, left, right, dst, rank1, rank7, badBatch, badK, badDst, q4Right, left32} {
			requireError(t, tensor.Close(), nil)
		}
		requireError(t, dev.Close(), nil)
	})
}
