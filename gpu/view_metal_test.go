//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"math"
	"slices"
	"testing"
)

func Test_MetalTensorViewZeroBytesAndRoundTrip(t *testing.T) {
	var (
		d        = requireMetalDevice(t)
		native   = d.state.backend.(metalBackend).runtime
		ctx      = context.Background()
		baseline = native.ResourceSnapshot()
	)
	defer func() {
		var after = native.ResourceSnapshot()
		if after.LiveBuffers != baseline.LiveBuffers || after.LiveBufferBytes != baseline.LiveBufferBytes ||
			after.LiveScopes != baseline.LiveScopes {
			t.Fatalf("native resource leak: before %+v, after %+v", baseline, after)
		}
	}()

	// Verify zero added device bytes across Float32, Float16, and BFloat16 views.
	for _, dtype := range []DType{Float32, Float16, BFloat16} {
		t.Run("zero_bytes_"+dtype.String(), func(t *testing.T) {
			var (
				base      = requireTensorWithDType(t, d, dtype, 24)
				snapAfter = native.ResourceSnapshot()
			)
			var v2d, err = base.View([]int64{4, 6})
			requireError(t, err, nil)
			var v3d *Tensor
			v3d, err = v2d.View([]int64{2, 3, 4})
			requireError(t, err, nil)
			var v4d *Tensor
			v4d, err = v3d.View([]int64{2, 1, 3, 4})
			requireError(t, err, nil)

			var snapViews = native.ResourceSnapshot()
			if snapViews.LiveBuffers != snapAfter.LiveBuffers || snapViews.LiveBufferBytes != snapAfter.LiveBufferBytes {
				t.Fatalf("creating views allocated device memory: before %+v, after %+v", snapAfter, snapViews)
			}

			// Close parent while views remain open, then verify copy through view.
			var input = make([]float32, 24)
			for i := range input {
				input[i] = float32(i-12) * 0.5
			}
			requireError(t, base.UploadFloat32As(input), nil)
			requireError(t, base.Close(), nil)

			var (
				dst   = requireTensorWithDType(t, d, dtype, 2, 1, 3, 4)
				scope = requireScope(t, d)
			)
			requireError(t, scope.Copy(v4d, dst), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)

			var got = make([]float32, 24)
			requireError(t, dst.DownloadFloat32Into(got), nil)
			if !slices.Equal(got, input) {
				t.Fatalf("%s view copy mismatch: got %v, want %v", dtype, got, input)
			}

			for _, handle := range []*Tensor{dst, v4d, v3d, v2d} {
				requireError(t, handle.Close(), nil)
			}
		})
	}

	// Resident Upload -> View -> MatMul -> View -> Copy -> Download in one Scope.
	t.Run("resident_matmul_copy_roundtrip", func(t *testing.T) {
		var (
			input1D  = requireTensor(t, d, 6)    // viewed as [2, 3]
			weight1D = requireTensor(t, d, 12)   // viewed as [3, 4]
			mid2D    = requireTensor(t, d, 2, 4) // viewed as [8]
			out1D    = requireTensor(t, d, 8)
		)
		var (
			aVals = []float32{1, 2, 3, 4, 5, 6}
			bVals = []float32{
				1, 0, -1, 2,
				0, 1, 2, -1,
				2, -1, 0, 1,
			}
			want = []float32{
				7, -1, 3, 3,
				16, -1, 6, 9,
			}
		)
		requireError(t, input1D.Upload(aVals), nil)
		requireError(t, weight1D.Upload(bVals), nil)

		var a2d, err = input1D.View([]int64{2, 3})
		requireError(t, err, nil)
		var b2d *Tensor
		b2d, err = weight1D.View([]int64{3, 4})
		requireError(t, err, nil)
		var mid1D *Tensor
		mid1D, err = mid2D.View([]int64{8})
		requireError(t, err, nil)

		// Aliasing check on real Metal scope: MatMul destination aliased via view must fail.
		var badScope = requireScope(t, d)
		var aliasOut *Tensor
		aliasOut, err = input1D.View([]int64{2, 3})
		requireError(t, err, nil)
		requireError(t, badScope.Copy(a2d, aliasOut), ErrInvalid)
		requireError(t, badScope.Close(), nil)
		requireError(t, aliasOut.Close(), nil)

		var scope = requireScope(t, d)
		requireError(t, scope.MatMul(a2d, b2d, mid2D), nil)
		requireError(t, scope.Copy(mid1D, out1D), nil)
		requireError(t, scope.Submit(ctx), nil)
		requireError(t, scope.Wait(ctx), nil)
		requireError(t, scope.Close(), nil)

		var got = make([]float32, 8)
		requireError(t, out1D.DownloadInto(got), nil)
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-5 {
				t.Fatalf("matmul+view output[%d] = %g, want %g", i, got[i], want[i])
			}
		}

		for _, handle := range []*Tensor{a2d, b2d, mid1D, input1D, weight1D, mid2D, out1D} {
			requireError(t, handle.Close(), nil)
		}
	})

	// Concurrent Metal scopes sharing read-only sibling views while parent closes.
	t.Run("concurrent_sibling_reads", func(t *testing.T) {
		var (
			parent  = requireTensor(t, d, 16)
			data    = make([]float32, 16)
			dstA    = requireTensor(t, d, 4, 4)
			dstB    = requireTensor(t, d, 2, 8)
			scopeA  = requireScope(t, d)
			scopeB  = requireScope(t, d)
			results = make(chan error, 2)
		)
		for i := range data {
			data[i] = float32(i+1) * 0.25
		}
		requireError(t, parent.Upload(data), nil)
		var vA, errA = parent.View([]int64{4, 4})
		requireError(t, errA, nil)
		var vB, errB = parent.View([]int64{2, 8})
		requireError(t, errB, nil)

		requireError(t, scopeA.Copy(vA, dstA), nil)
		requireError(t, scopeB.Copy(vB, dstB), nil)
		requireError(t, parent.Close(), nil)
		requireError(t, scopeA.Submit(ctx), nil)
		requireError(t, scopeB.Submit(ctx), nil)
		go func() { results <- scopeA.Wait(ctx) }()
		go func() { results <- scopeB.Wait(ctx) }()
		requireError(t, <-results, nil)
		requireError(t, <-results, nil)
		requireError(t, scopeA.Close(), nil)
		requireError(t, scopeB.Close(), nil)

		var (
			gotA = make([]float32, 16)
			gotB = make([]float32, 16)
		)
		requireError(t, dstA.DownloadInto(gotA), nil)
		requireError(t, dstB.DownloadInto(gotB), nil)
		if !slices.Equal(gotA, data) || !slices.Equal(gotB, data) {
			t.Fatalf("concurrent view copy mismatch: A=%v B=%v want=%v", gotA, gotB, data)
		}

		for _, handle := range []*Tensor{vA, vB, dstA, dstB} {
			requireError(t, handle.Close(), nil)
		}
	})
}
