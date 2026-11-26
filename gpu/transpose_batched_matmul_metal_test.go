//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
	"testing"
)

func referenceTransposeBits[T comparable](input []T, inShape []int64, perm []int) (out []T) {
	var (
		rank           = len(inShape)
		outShape       = make([]int64, rank)
		inStride       = make([]int64, rank)
		total    int64 = 1
	)
	for i := rank - 1; i >= 0; i-- {
		inStride[i] = total
		total *= inShape[i]
	}
	for i := range rank {
		outShape[i] = inShape[perm[i]]
	}
	out = make([]T, total)
	var outStride = make([]int64, rank)
	var s int64 = 1
	for i := rank - 1; i >= 0; i-- {
		outStride[i] = s
		s *= outShape[i]
	}
	for idx := int64(0); idx < total; idx++ {
		var (
			rem   = idx
			inIdx int64
		)
		for d := range rank {
			var coord = rem / outStride[d]
			rem -= coord * outStride[d]
			inIdx += coord * inStride[perm[d]]
		}
		out[idx] = input[inIdx]
	}
	return out
}

func referenceMatMulLeftTransposeFP64(left, right []float32, k, m, n int) (out []float64) {
	out = make([]float64, m*n)
	for row := range m {
		for col := range n {
			var sum float64
			for inner := range k {
				sum += float64(left[inner*m+row]) * float64(right[inner*n+col])
			}
			out[row*n+col] = sum
		}
	}
	return out
}

func referenceBatchedMatMulFP64(
	left, right []float32,
	leftShape, rightShape []int64,
	rightTransposed bool,
) (out []float64, outShape []int64) {
	var (
		lRank     = len(leftShape)
		rRank     = len(rightShape)
		m         = int(leftShape[lRank-2])
		k         = int(leftShape[lRank-1])
		n         = int(rightShape[rRank-1])
		lBatch    = leftShape[:lRank-2]
		rBatch    = rightShape[:rRank-2]
		batchRank = max(len(lBatch), len(rBatch))
	)
	if rightTransposed {
		n = int(rightShape[rRank-2])
	}
	outShape = make([]int64, batchRank+2)
	var (
		lStrides     = make([]int, batchRank)
		rStrides     = make([]int, batchRank)
		outStrides   = make([]int, batchRank)
		totalBatches = 1
	)
	var ls = 1
	for i := len(lBatch) - 1; i >= 0; i-- {
		var idx = batchRank - len(lBatch) + i
		if lBatch[i] == 1 {
			lStrides[idx] = 0
		} else {
			lStrides[idx] = ls
			ls *= int(lBatch[i])
		}
	}
	var rs = 1
	for i := len(rBatch) - 1; i >= 0; i-- {
		var idx = batchRank - len(rBatch) + i
		if rBatch[i] == 1 {
			rStrides[idx] = 0
		} else {
			rStrides[idx] = rs
			rs *= int(rBatch[i])
		}
	}
	for i := range batchRank {
		var (
			ld int64 = 1
			rd int64 = 1
		)
		if i >= batchRank-len(lBatch) {
			ld = lBatch[i-(batchRank-len(lBatch))]
		}
		if i >= batchRank-len(rBatch) {
			rd = rBatch[i-(batchRank-len(rBatch))]
		}
		outShape[i] = max(ld, rd)
		totalBatches *= int(outShape[i])
	}
	var os = 1
	for i := batchRank - 1; i >= 0; i-- {
		outStrides[i] = os
		os *= int(outShape[i])
	}
	outShape[batchRank] = int64(m)
	outShape[batchRank+1] = int64(n)
	out = make([]float64, totalBatches*m*n)

	for b := range totalBatches {
		var (
			rem    = b
			lBatch int
			rBatch int
		)
		for d := range batchRank {
			var coord = rem / outStrides[d]
			rem -= coord * outStrides[d]
			lBatch += coord * lStrides[d]
			rBatch += coord * rStrides[d]
		}
		var (
			lBase = lBatch * m * k
			rBase = rBatch * k * n
			oBase = b * m * n
		)
		for row := range m {
			for col := range n {
				var sum float64
				for inner := range k {
					var rv float64
					if rightTransposed {
						rv = float64(right[rBase+col*k+inner])
					} else {
						rv = float64(right[rBase+inner*n+col])
					}
					sum += float64(left[lBase+row*k+inner]) * rv
				}
				out[oBase+row*n+col] = sum
			}
		}
	}
	return out, outShape
}

func quantizeRoundTrip(values []float32, dt DType) (out []float32) {
	out = make([]float32, len(values))
	switch dt {
	case Float16:
		for i, v := range values {
			out[i] = Float16ToFloat32(Float32ToFloat16(v))
		}
	case BFloat16:
		for i, v := range values {
			out[i] = BFloat16ToFloat32(Float32ToBFloat16(v))
		}
	default:
		copy(out, values)
	}
	return out
}

func nn17ErrorStats(got []float32, want []float64) (p50, p95, p99, maxErr float64) {
	var errs = make([]float64, len(got))
	for i := range got {
		var diff = math.Abs(float64(got[i]) - want[i])
		errs[i] = diff
		if diff > maxErr {
			maxErr = diff
		}
	}
	slices.Sort(errs)
	if len(errs) == 0 {
		return 0, 0, 0, 0
	}
	p50 = errs[len(errs)*50/100]
	p95 = errs[len(errs)*95/100]
	p99 = errs[len(errs)*99/100]
	return p50, p95, p99, maxErr
}

func Test_MetalTransposeAgainstReference(t *testing.T) {
	var (
		dev     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = dev.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
	)

	type testcase struct {
		name  string
		shape []int64
		perm  []int
	}
	var tests = []testcase{
		{name: "1d_identity", shape: []int64{17}, perm: []int{0}},
		{name: "2d_1x1", shape: []int64{1, 1}, perm: []int{1, 0}},
		{name: "2d_odd_7x13", shape: []int64{7, 13}, perm: []int{1, 0}},
		{name: "2d_tile_16x16", shape: []int64{16, 16}, perm: []int{1, 0}},
		{name: "2d_odd_tile_17x33", shape: []int64{17, 33}, perm: []int{1, 0}},
		{name: "2d_64x128", shape: []int64{64, 128}, perm: []int{1, 0}},
		{name: "3d_perm_2_0_1", shape: []int64{3, 5, 7}, perm: []int{2, 0, 1}},
		{name: "3d_size1_contiguous", shape: []int64{4, 1, 9}, perm: []int{0, 2, 1}},
		{name: "4d_attention_bshd_to_bhsd", shape: []int64{2, 17, 8, 32}, perm: []int{0, 2, 1, 3}},
		{name: "4d_attention_bhsd_to_bshd", shape: []int64{2, 8, 17, 32}, perm: []int{0, 2, 1, 3}},
		{name: "6d_arbitrary", shape: []int64{2, 3, 2, 2, 3, 2}, perm: []int{5, 2, 0, 4, 1, 3}},
	}

	for _, tt := range tests {
		var dstShape = make([]int64, len(tt.shape))
		var total int64 = 1
		for i, p := range tt.perm {
			dstShape[i] = tt.shape[p]
			total *= tt.shape[i]
		}
		for _, dt := range []DType{Float32, Float16, BFloat16} {
			t.Run(fmt.Sprintf("%s/%s", tt.name, dt), func(t *testing.T) {
				var (
					src = requireTensorWithDType(t, dev, dt, tt.shape...)
					dst = requireTensorWithDType(t, dev, dt, dstShape...)
				)
				if dt == Float32 {
					var inBits = make([]uint32, total)
					var inVals = make([]float32, total)
					for i := range total {
						var v = float32(i)*0.25 - 10.0
						inVals[i] = v
						inBits[i] = math.Float32bits(v)
					}
					if total >= 4 {
						inVals[0] = float32(math.NaN())
						inVals[1] = float32(math.Inf(1))
						inVals[2] = float32(math.Inf(-1))
						inVals[3] = float32(math.Copysign(0, -1))
						for i := range 4 {
							inBits[i] = math.Float32bits(inVals[i])
						}
					}
					requireError(t, src.Upload(inVals), nil)
					var scope, err = dev.NewScope()
					requireError(t, err, nil)
					requireError(t, scope.Transpose(src, dst, tt.perm), nil)
					requireError(t, scope.Submit(ctx), nil)
					requireError(t, scope.Wait(ctx), nil)
					requireError(t, scope.Close(), nil)

					var got = make([]float32, total)
					requireError(t, dst.DownloadInto(got), nil)
					var wantBits = referenceTransposeBits(inBits, tt.shape, tt.perm)
					for i := range got {
						if math.Float32bits(got[i]) != wantBits[i] {
							t.Fatalf("bit mismatch at %d: got 0x%08x, want 0x%08x", i, math.Float32bits(got[i]), wantBits[i])
						}
					}
				} else {
					var inBits = make([]uint16, total)
					for i := range total {
						inBits[i] = uint16((i*37 + 13) & 0xffff)
					}
					requireError(t, src.UploadBits(inBits), nil)
					var scope, err = dev.NewScope()
					requireError(t, err, nil)
					requireError(t, scope.Transpose(src, dst, tt.perm), nil)
					requireError(t, scope.Submit(ctx), nil)
					requireError(t, scope.Wait(ctx), nil)
					requireError(t, scope.Close(), nil)

					var got = make([]uint16, total)
					requireError(t, dst.DownloadBitsInto(got), nil)
					var wantBits = referenceTransposeBits(inBits, tt.shape, tt.perm)
					if !slices.Equal(got, wantBits) {
						t.Fatalf("16-bit transpose mismatch for %s %s", tt.name, dt)
					}
				}
				requireError(t, src.Close(), nil)
				requireError(t, dst.Close(), nil)
			})
		}
	}

	// Verify view execution and view-alias rejection.
	t.Run("view_and_alias_rejection", func(t *testing.T) {
		var (
			base = requireTensor(t, dev, 24)
			dst  = requireTensor(t, dev, 4, 6)
			vals = make([]float32, 24)
		)
		for i := range vals {
			vals[i] = float32(i + 1)
		}
		requireError(t, base.Upload(vals), nil)
		var viewIn, err = base.View([]int64{6, 4})
		requireError(t, err, nil)
		var viewAlias, err2 = base.View([]int64{4, 6})
		requireError(t, err2, nil)

		var scope, sErr = dev.NewScope()
		requireError(t, sErr, nil)
		if !errors.Is(scope.Transpose(viewIn, viewAlias, []int{1, 0}), ErrInvalid) {
			t.Fatalf("expected ErrInvalid for view-aliased transpose")
		}
		requireError(t, scope.Close(), nil)

		scope, sErr = dev.NewScope()
		requireError(t, sErr, nil)
		requireError(t, scope.Transpose(viewIn, dst, []int{1, 0}), nil)
		requireError(t, scope.Submit(ctx), nil)
		requireError(t, scope.Wait(ctx), nil)
		requireError(t, scope.Close(), nil)

		var got = make([]float32, 24)
		requireError(t, dst.DownloadInto(got), nil)
		var want = referenceTransposeBits(vals, []int64{6, 4}, []int{1, 0})
		if !slices.Equal(got, want) {
			t.Fatalf("view transpose mismatch: got %v, want %v", got, want)
		}
		for _, tensor := range []*Tensor{viewIn, viewAlias, base, dst} {
			requireError(t, tensor.Close(), nil)
		}
	})

	if after := native.ResourceSnapshot(); after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("resource leak: initial=%+v after=%+v", initial, after)
	}
}

func Test_MetalMatMulLeftTransposeAgainstFP64(t *testing.T) {
	var (
		dev     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = dev.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		rng     = rand.New(rand.NewPCG(1701, 1702))
	)

	type shape struct {
		name    string
		k, m, n int64
	}
	var shapes = []shape{
		{name: "1x1x1", k: 1, m: 1, n: 1},
		{name: "3x5x7", k: 3, m: 5, n: 7},
		{name: "7x7x7", k: 7, m: 7, n: 7},
		{name: "8x4x8", k: 8, m: 4, n: 8},
		{name: "15x15x15", k: 15, m: 15, n: 15},
		{name: "16x16x16", k: 16, m: 16, n: 16},
		{name: "17x19x23", k: 17, m: 19, n: 23},
		{name: "31x31x31", k: 31, m: 31, n: 31},
		{name: "64x32x64", k: 64, m: 32, n: 64},
		{name: "128x65x128", k: 128, m: 65, n: 128},
		{name: "129x33x65", k: 129, m: 33, n: 65},
	}

	for _, sc := range shapes {
		var leftData = make([]float32, sc.k*sc.m)
		var rightData = make([]float32, sc.k*sc.n)
		for i := range leftData {
			leftData[i] = float32(rng.Float64()*2 - 1)
		}
		for i := range rightData {
			rightData[i] = float32(rng.Float64()*2 - 1)
		}
		for _, dt := range []DType{Float32, Float16, BFloat16} {
			t.Run(fmt.Sprintf("%s/%s", sc.name, dt), func(t *testing.T) {
				var (
					left  = requireTensor(t, dev, sc.k, sc.m)
					right = requireTensorWithDType(t, dev, dt, sc.k, sc.n)
					dst   = requireTensor(t, dev, sc.m, sc.n)
				)
				requireError(t, left.Upload(leftData), nil)
				requireError(t, right.UploadFloat32As(rightData), nil)
				var scope, err = dev.NewScope()
				requireError(t, err, nil)
				requireError(t, scope.MatMulLeftTranspose(left, right, dst), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)

				var got = make([]float32, sc.m*sc.n)
				requireError(t, dst.DownloadInto(got), nil)
				var effectiveRight = quantizeRoundTrip(rightData, dt)
				var want = referenceMatMulLeftTransposeFP64(leftData, effectiveRight, int(sc.k), int(sc.m), int(sc.n))
				var p50, p95, p99, maxErr = nn17ErrorStats(got, want)
				var tol = 1e-5 * math.Sqrt(float64(sc.k))
				t.Logf("MatMulLeftTranspose %s/%s: p50=%.3e p95=%.3e p99=%.3e max=%.3e", sc.name, dt, p50, p95, p99, maxErr)
				if maxErr > tol {
					t.Fatalf("max error %.3e exceeds %.3e", maxErr, tol)
				}
				for _, tensor := range []*Tensor{left, right, dst} {
					requireError(t, tensor.Close(), nil)
				}
			})
		}
	}

	t.Run("special_values_and_self_transpose", func(t *testing.T) {
		var (
			left  = requireTensor(t, dev, 2, 3)
			right = requireTensor(t, dev, 2, 3)
			dst   = requireTensor(t, dev, 3, 3)
		)
		requireError(t, left.Upload([]float32{
			float32(math.NaN()), 0, float32(math.Inf(1)),
			1, 2, 3,
		}), nil)
		requireError(t, right.Upload([]float32{
			1, float32(math.Inf(1)), 1,
			1, 1, 1,
		}), nil)
		var scope, err = dev.NewScope()
		requireError(t, err, nil)
		requireError(t, scope.MatMulLeftTranspose(left, right, dst), nil)
		requireError(t, scope.Submit(ctx), nil)
		requireError(t, scope.Wait(ctx), nil)
		requireError(t, scope.Close(), nil)
		var got = make([]float32, 9)
		requireError(t, dst.DownloadInto(got), nil)
		if !math.IsNaN(float64(got[0])) {
			t.Fatalf("expected NaN propagation at [0,0], got %v", got[0])
		}
		if !math.IsNaN(float64(got[4])) {
			t.Fatalf("expected 0*Inf -> NaN at [1,1], got %v", got[4])
		}
		if !math.IsInf(float64(got[6]), 1) {
			t.Fatalf("expected +Inf at [2,0], got %v", got[6])
		}

		// Self-product A^T * A.
		requireError(t, left.Upload([]float32{1, 2, 3, 4, 5, 6}), nil)
		scope, err = dev.NewScope()
		requireError(t, err, nil)
		requireError(t, scope.MatMulLeftTranspose(left, left, dst), nil)
		requireError(t, scope.Submit(ctx), nil)
		requireError(t, scope.Wait(ctx), nil)
		requireError(t, scope.Close(), nil)
		requireError(t, dst.DownloadInto(got), nil)
		var want = referenceMatMulLeftTransposeFP64([]float32{1, 2, 3, 4, 5, 6}, []float32{1, 2, 3, 4, 5, 6}, 2, 3, 3)
		var _, _, _, maxErr = nn17ErrorStats(got, want)
		if maxErr > 1e-6 {
			t.Fatalf("A^T*A max error %.3e", maxErr)
		}
		for _, tensor := range []*Tensor{left, right, dst} {
			requireError(t, tensor.Close(), nil)
		}
	})

	if after := native.ResourceSnapshot(); after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("resource leak: initial=%+v after=%+v", initial, after)
	}
}

func Test_MetalBatchedMatMulAgainstFP64(t *testing.T) {
	var (
		dev     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = dev.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		rng     = rand.New(rand.NewPCG(1703, 1704))
	)

	type testcase struct {
		name       string
		leftShape  []int64
		rightShape []int64
		rightT     bool
	}
	var tests = []testcase{
		{name: "2d_x_2d_tiny_fallback", leftShape: []int64{3, 7}, rightShape: []int64{7, 7}},
		{name: "2d_x_2d_rt_tiny_fallback", leftShape: []int64{7, 8}, rightShape: []int64{9, 8}, rightT: true},
		{name: "2d_x_2d_tiled_min", leftShape: []int64{4, 8}, rightShape: []int64{8, 8}},
		{name: "2d_x_2d_rt_tiled_min", leftShape: []int64{8, 8}, rightShape: []int64{8, 8}, rightT: true},
		{name: "2d_x_2d_15x15x15", leftShape: []int64{15, 15}, rightShape: []int64{15, 15}},
		{name: "2d_x_2d_16x16x16_rt", leftShape: []int64{16, 16}, rightShape: []int64{16, 16}, rightT: true},
		{name: "2d_x_2d", leftShape: []int64{17, 33}, rightShape: []int64{33, 19}},
		{name: "2d_x_2d_rt_vec4", leftShape: []int64{17, 64}, rightShape: []int64{19, 64}, rightT: true},
		{name: "3d_x_3d_odd", leftShape: []int64{4, 17, 33}, rightShape: []int64{4, 33, 19}},
		{name: "3d_x_3d_rt_odd", leftShape: []int64{4, 17, 33}, rightShape: []int64{4, 19, 33}, rightT: true},
		{name: "3d_x_3d_edge_31x31x31", leftShape: []int64{2, 31, 31}, rightShape: []int64{2, 31, 31}},
		{name: "3d_x_3d_edge_33x35x65", leftShape: []int64{2, 33, 35}, rightShape: []int64{2, 35, 65}},
		{name: "3d_x_3d_rt_edge_33x35x65", leftShape: []int64{2, 33, 35}, rightShape: []int64{2, 65, 35}, rightT: true},
		{name: "3d_x_2d_shared_weight", leftShape: []int64{3, 19, 64}, rightShape: []int64{64, 48}},
		{name: "3d_x_2d_shared_weight_rt", leftShape: []int64{3, 19, 64}, rightShape: []int64{48, 64}, rightT: true},
		{name: "3d_x_2d_large_edge_65x128x128", leftShape: []int64{2, 65, 128}, rightShape: []int64{128, 128}},
		{name: "2d_x_3d_broadcast", leftShape: []int64{11, 29}, rightShape: []int64{3, 29, 13}},
		{name: "4d_attention_scores", leftShape: []int64{2, 8, 32, 64}, rightShape: []int64{2, 8, 64, 32}},
		{name: "4d_attention_scores_rt", leftShape: []int64{2, 8, 32, 64}, rightShape: []int64{2, 8, 32, 64}, rightT: true},
		{name: "4d_multi_broadcast", leftShape: []int64{2, 1, 13, 29}, rightShape: []int64{1, 3, 29, 17}},
		{name: "6d_broadcast", leftShape: []int64{2, 1, 2, 1, 5, 7}, rightShape: []int64{1, 2, 1, 3, 7, 4}},
		{name: "6d_broadcast_rt", leftShape: []int64{2, 1, 2, 1, 5, 8}, rightShape: []int64{1, 2, 1, 3, 4, 8}, rightT: true},
		{name: "6d_broadcast_tiled", leftShape: []int64{2, 1, 2, 1, 9, 17}, rightShape: []int64{1, 2, 1, 2, 17, 19}},
		{name: "6d_broadcast_tiled_rt", leftShape: []int64{2, 1, 2, 1, 9, 17}, rightShape: []int64{1, 2, 1, 2, 19, 17}, rightT: true},
	}

	for _, tt := range tests {
		var (
			lCount int64 = 1
			rCount int64 = 1
		)
		for _, d := range tt.leftShape {
			lCount *= d
		}
		for _, d := range tt.rightShape {
			rCount *= d
		}
		var (
			leftData  = make([]float32, lCount)
			rightData = make([]float32, rCount)
		)
		for i := range leftData {
			leftData[i] = float32(rng.Float64()*2 - 1)
		}
		for i := range rightData {
			rightData[i] = float32(rng.Float64()*2 - 1)
		}
		for _, dt := range []DType{Float32, Float16, BFloat16} {
			t.Run(fmt.Sprintf("%s/%s", tt.name, dt), func(t *testing.T) {
				var effectiveRight = quantizeRoundTrip(rightData, dt)
				var want, dstShape = referenceBatchedMatMulFP64(leftData, effectiveRight, tt.leftShape, tt.rightShape, tt.rightT)
				var (
					left  = requireTensor(t, dev, tt.leftShape...)
					right = requireTensorWithDType(t, dev, dt, tt.rightShape...)
					dst   = requireTensor(t, dev, dstShape...)
				)
				requireError(t, left.Upload(leftData), nil)
				requireError(t, right.UploadFloat32As(rightData), nil)
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

				var got = make([]float32, len(want))
				requireError(t, dst.DownloadInto(got), nil)
				var p50, p95, p99, maxErr = nn17ErrorStats(got, want)
				var k = tt.leftShape[len(tt.leftShape)-1]
				var tol = 1e-5 * math.Sqrt(float64(k))
				t.Logf("BatchedMatMul %s/%s: p50=%.3e p95=%.3e p99=%.3e max=%.3e", tt.name, dt, p50, p95, p99, maxErr)
				if maxErr > tol {
					t.Fatalf("max error %.3e exceeds %.3e", maxErr, tol)
				}
				for _, tensor := range []*Tensor{left, right, dst} {
					requireError(t, tensor.Close(), nil)
				}
			})
		}
	}

	if after := native.ResourceSnapshot(); after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("resource leak: initial=%+v after=%+v", initial, after)
	}
}

func Test_MetalNN17ConcurrentScopes(t *testing.T) {
	var (
		dev     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = dev.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		rng     = rand.New(rand.NewPCG(1705, 1706))
	)

	// Shared immutable inputs across concurrent workers.
	var (
		tIn  = requireTensor(t, dev, 2, 16, 4, 8)
		ltL  = requireTensor(t, dev, 32, 16)
		ltR  = requireTensorWithDType(t, dev, Float16, 32, 24)
		bmL  = requireTensor(t, dev, 2, 4, 16, 32)
		bmR  = requireTensorWithDType(t, dev, BFloat16, 2, 4, 16, 32)
		tInV = make([]float32, 2*16*4*8)
		ltLV = make([]float32, 32*16)
		ltRV = make([]float32, 32*24)
		bmLV = make([]float32, 2*4*16*32)
		bmRV = make([]float32, 2*4*16*32)
	)
	for i := range tInV {
		tInV[i] = float32(i) * 0.1
	}
	for i := range ltLV {
		ltLV[i] = float32(rng.Float64()*2 - 1)
	}
	for i := range ltRV {
		ltRV[i] = float32(rng.Float64()*2 - 1)
	}
	for i := range bmLV {
		bmLV[i] = float32(rng.Float64()*2 - 1)
	}
	for i := range bmRV {
		bmRV[i] = float32(rng.Float64()*2 - 1)
	}
	requireError(t, tIn.Upload(tInV), nil)
	requireError(t, ltL.Upload(ltLV), nil)
	requireError(t, ltR.UploadFloat32As(ltRV), nil)
	requireError(t, bmL.Upload(bmLV), nil)
	requireError(t, bmR.UploadFloat32As(bmRV), nil)

	var (
		wantT     = referenceTransposeBits(tInV, []int64{2, 16, 4, 8}, []int{0, 2, 1, 3})
		wantLT    = referenceMatMulLeftTransposeFP64(ltLV, quantizeRoundTrip(ltRV, Float16), 32, 16, 24)
		wantBM, _ = referenceBatchedMatMulFP64(bmLV, quantizeRoundTrip(bmRV, BFloat16), []int64{2, 4, 16, 32}, []int64{2, 4, 16, 32}, true)
	)

	const workers = 8
	const iters = 16
	var wg sync.WaitGroup
	var errCh = make(chan error, workers*iters)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var (
				tDst  = requireTensor(t, dev, 2, 4, 16, 8)
				ltDst = requireTensor(t, dev, 16, 24)
				bmDst = requireTensor(t, dev, 2, 4, 16, 16)
				gotT  = make([]float32, len(wantT))
				gotLT = make([]float32, len(wantLT))
				gotBM = make([]float32, len(wantBM))
			)
			defer func() {
				_ = tDst.Close()
				_ = ltDst.Close()
				_ = bmDst.Close()
			}()
			for range iters {
				var scope, err = dev.NewScope()
				if err != nil {
					errCh <- err
					return
				}
				if err = scope.Transpose(tIn, tDst, []int{0, 2, 1, 3}); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.MatMulLeftTranspose(ltL, ltR, ltDst); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.BatchedMatMulRightTranspose(bmL, bmR, bmDst); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.Submit(ctx); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.Wait(ctx); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.Close(); err != nil {
					errCh <- err
					return
				}
				if err = tDst.DownloadInto(gotT); err != nil {
					errCh <- err
					return
				}
				if err = ltDst.DownloadInto(gotLT); err != nil {
					errCh <- err
					return
				}
				if err = bmDst.DownloadInto(gotBM); err != nil {
					errCh <- err
					return
				}
				if !slices.Equal(gotT, wantT) {
					errCh <- errors.New("concurrent transpose mismatch")
					return
				}
				if _, _, _, maxErr := nn17ErrorStats(gotLT, wantLT); maxErr > 1e-4 {
					errCh <- fmt.Errorf("concurrent left-transpose matmul max err %.3e", maxErr)
					return
				}
				if _, _, _, maxErr := nn17ErrorStats(gotBM, wantBM); maxErr > 1e-4 {
					errCh <- fmt.Errorf("concurrent batched matmul max err %.3e", maxErr)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
	for _, tensor := range []*Tensor{tIn, ltL, ltR, bmL, bmR} {
		requireError(t, tensor.Close(), nil)
	}
	if after := native.ResourceSnapshot(); after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("resource leak: initial=%+v after=%+v", initial, after)
	}
}

func referenceSoftmaxLastAxisFP64(input []float64, rows, cols int) (out []float64) {
	out = make([]float64, rows*cols)
	for r := range rows {
		var (
			base = r * cols
			m    = input[base]
		)
		for c := 1; c < cols; c++ {
			if input[base+c] > m {
				m = input[base+c]
			}
		}
		var sum float64
		for c := range cols {
			var e = math.Exp(input[base+c] - m)
			out[base+c] = e
			sum += e
		}
		for c := range cols {
			out[base+c] /= sum
		}
	}
	return out
}

func referenceLayerNormFP64FromFloat64(input []float64, weight, bias []float32, rows, width int, epsilon float32) (out []float64) {
	out = make([]float64, rows*width)
	var eps = float64(epsilon)
	for r := range rows {
		var (
			base = r * width
			sum  float64
		)
		for c := range width {
			sum += input[base+c]
		}
		var (
			mean   = sum / float64(width)
			varSum float64
		)
		for c := range width {
			var d = input[base+c] - mean
			varSum += d * d
		}
		var invStd = 1.0 / math.Sqrt(varSum/float64(width)+eps)
		for c := range width {
			out[base+c] = (input[base+c]-mean)*invStd*float64(weight[c]) + float64(bias[c])
		}
	}
	return out
}

func referenceBatchedMatMulFP64Inputs(
	left []float64,
	right []float32,
	leftShape, rightShape []int64,
	rightTransposed bool,
) (out []float64) {
	var left32 = make([]float32, len(left))
	for i, v := range left {
		left32[i] = float32(v)
	}
	// Compute using FP64 left directly to avoid double-rounding.
	var (
		lRank     = len(leftShape)
		rRank     = len(rightShape)
		m         = int(leftShape[lRank-2])
		k         = int(leftShape[lRank-1])
		n         = int(rightShape[rRank-1])
		lBatch    = leftShape[:lRank-2]
		rBatch    = rightShape[:rRank-2]
		batchRank = max(len(lBatch), len(rBatch))
	)
	if rightTransposed {
		n = int(rightShape[rRank-2])
	}
	var (
		lStrides     = make([]int, batchRank)
		rStrides     = make([]int, batchRank)
		outStrides   = make([]int, batchRank)
		outBatch     = make([]int64, batchRank)
		totalBatches = 1
	)
	var ls = 1
	for i := len(lBatch) - 1; i >= 0; i-- {
		var idx = batchRank - len(lBatch) + i
		if lBatch[i] != 1 {
			lStrides[idx] = ls
			ls *= int(lBatch[i])
		}
	}
	var rs = 1
	for i := len(rBatch) - 1; i >= 0; i-- {
		var idx = batchRank - len(rBatch) + i
		if rBatch[i] != 1 {
			rStrides[idx] = rs
			rs *= int(rBatch[i])
		}
	}
	for i := range batchRank {
		var ld, rd int64 = 1, 1
		if i >= batchRank-len(lBatch) {
			ld = lBatch[i-(batchRank-len(lBatch))]
		}
		if i >= batchRank-len(rBatch) {
			rd = rBatch[i-(batchRank-len(rBatch))]
		}
		outBatch[i] = max(ld, rd)
		totalBatches *= int(outBatch[i])
	}
	var os = 1
	for i := batchRank - 1; i >= 0; i-- {
		outStrides[i] = os
		os *= int(outBatch[i])
	}
	out = make([]float64, totalBatches*m*n)
	for b := range totalBatches {
		var rem, lb, rb = b, 0, 0
		for d := range batchRank {
			var c = rem / outStrides[d]
			rem -= c * outStrides[d]
			lb += c * lStrides[d]
			rb += c * rStrides[d]
		}
		var lBase, rBase, oBase = lb * m * k, rb * k * n, b * m * n
		for row := range m {
			for col := range n {
				var sum float64
				for inner := range k {
					var rv float64
					if rightTransposed {
						rv = float64(right[rBase+col*k+inner])
					} else {
						rv = float64(right[rBase+inner*n+col])
					}
					sum += left[lBase+row*k+inner] * rv
				}
				out[oBase+row*n+col] = sum
			}
		}
	}
	_ = left32
	return out
}

func referenceBatchedMatMulFP64Both(left, right []float64, b, h, m, k, n int, rightT bool) (out []float64) {
	var total = b * h
	out = make([]float64, total*m*n)
	for batch := range total {
		var lBase, rBase, oBase = batch * m * k, batch * k * n, batch * m * n
		for row := range m {
			for col := range n {
				var sum float64
				for inner := range k {
					var rv float64
					if rightT {
						rv = right[rBase+col*k+inner]
					} else {
						rv = right[rBase+inner*n+col]
					}
					sum += left[lBase+row*k+inner] * rv
				}
				out[oBase+row*n+col] = sum
			}
		}
	}
	return out
}

func Test_MetalONNXTransformerEncoderBlockSingleScope(t *testing.T) {
	var (
		dev     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = dev.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
	)

	const (
		B   = int64(2)
		S   = int64(16)
		H   = int64(4)
		D   = int64(16)
		E   = H * D
		FFN = E * 2
		eps = float32(1e-5)
	)
	var scale = float32(1.0 / math.Sqrt(float64(D)))

	for _, wt := range []DType{Float32, Float16, BFloat16} {
		t.Run(wt.String(), func(t *testing.T) {
			var (
				rng   = rand.New(rand.NewPCG(1707, 1708))
				fillF = func(n int64, s float64) []float32 {
					var out = make([]float32, n)
					for i := range out {
						out[i] = float32((rng.Float64()*2 - 1) * s)
					}
					return out
				}
				xData    = fillF(B*S*E, 0.5)
				ln1WData = fillF(E, 0.5)
				ln1BData = fillF(E, 0.2)
				wqData   = fillF(E*E, 0.2)
				wkData   = fillF(E*E, 0.2)
				wvData   = fillF(E*E, 0.2)
				woData   = fillF(E*E, 0.2)
				boData   = fillF(E, 0.1)
				ln2WData = fillF(E, 0.5)
				ln2BData = fillF(E, 0.2)
				w1Data   = fillF(E*FFN, 0.2)
				b1Data   = fillF(FFN, 0.1)
				w2Data   = fillF(FFN*E, 0.2)
				b2Data   = fillF(E, 0.1)
			)

			// Allocate device tensors.
			var (
				x      = requireTensor(t, dev, B, S, E)
				ln1Out = requireTensor(t, dev, B, S, E)
				ln1W   = requireTensorWithDType(t, dev, wt, E)
				ln1B   = requireTensorWithDType(t, dev, wt, E)
				wq     = requireTensorWithDType(t, dev, wt, E, E)
				wk     = requireTensorWithDType(t, dev, wt, E, E)
				wv     = requireTensorWithDType(t, dev, wt, E, E)
				wo     = requireTensorWithDType(t, dev, wt, E, E)
				bo     = requireTensor(t, dev, E)
				ln2W   = requireTensorWithDType(t, dev, wt, E)
				ln2B   = requireTensorWithDType(t, dev, wt, E)
				w1     = requireTensorWithDType(t, dev, wt, E, FFN)
				b1     = requireTensor(t, dev, FFN)
				w2     = requireTensorWithDType(t, dev, wt, FFN, E)
				b2     = requireTensor(t, dev, E)

				qProj   = requireTensor(t, dev, B, S, E)
				kProj   = requireTensor(t, dev, B, S, E)
				vProj   = requireTensor(t, dev, B, S, E)
				qHeads  = requireTensor(t, dev, B, H, S, D)
				kHeads  = requireTensor(t, dev, B, H, S, D)
				vHeads  = requireTensor(t, dev, B, H, S, D)
				scores  = requireTensor(t, dev, B, H, S, S)
				scaled  = requireTensor(t, dev, B, H, S, S)
				probs   = requireTensor(t, dev, B, H, S, S)
				ctxH    = requireTensor(t, dev, B, H, S, D)
				ctxS    = requireTensor(t, dev, B, S, H, D)
				attnOut = requireTensor(t, dev, B, S, E)
				attnB   = requireTensor(t, dev, B, S, E)
				res1    = requireTensor(t, dev, B, S, E)
				ln2Out  = requireTensor(t, dev, B, S, E)
				ffn1    = requireTensor(t, dev, B, S, FFN)
				ffn1B   = requireTensor(t, dev, B, S, FFN)
				ffnAct  = requireTensor(t, dev, B, S, FFN)
				ffn2    = requireTensor(t, dev, B, S, E)
				ffn2B   = requireTensor(t, dev, B, S, E)
				finalY  = requireTensor(t, dev, B, S, E)
			)

			// Zero-copy views for [B,S,E] <-> [B,S,H,D].
			var (
				q4d, err1   = qProj.View([]int64{B, S, H, D})
				k4d, err2   = kProj.View([]int64{B, S, H, D})
				v4d, err3   = vProj.View([]int64{B, S, H, D})
				ctx3d, err4 = ctxS.View([]int64{B, S, E})
			)
			requireError(t, err1, nil)
			requireError(t, err2, nil)
			requireError(t, err3, nil)
			requireError(t, err4, nil)

			requireError(t, x.Upload(xData), nil)
			requireError(t, ln1W.UploadFloat32As(ln1WData), nil)
			requireError(t, ln1B.UploadFloat32As(ln1BData), nil)
			requireError(t, wq.UploadFloat32As(wqData), nil)
			requireError(t, wk.UploadFloat32As(wkData), nil)
			requireError(t, wv.UploadFloat32As(wvData), nil)
			requireError(t, wo.UploadFloat32As(woData), nil)
			requireError(t, bo.Upload(boData), nil)
			requireError(t, ln2W.UploadFloat32As(ln2WData), nil)
			requireError(t, ln2B.UploadFloat32As(ln2BData), nil)
			requireError(t, w1.UploadFloat32As(w1Data), nil)
			requireError(t, b1.Upload(b1Data), nil)
			requireError(t, w2.UploadFloat32As(w2Data), nil)
			requireError(t, b2.Upload(b2Data), nil)

			// Execute entire Transformer encoder block in ONE gpu.Scope!
			var beforeSubmits = native.ResourceSnapshot().SubmittedCommands
			var scope, err = dev.NewScope()
			requireError(t, err, nil)
			requireError(t, scope.LayerNorm(x, ln1W, ln1B, ln1Out, eps), nil)
			requireError(t, scope.BatchedMatMul(ln1Out, wq, qProj), nil)
			requireError(t, scope.BatchedMatMul(ln1Out, wk, kProj), nil)
			requireError(t, scope.BatchedMatMul(ln1Out, wv, vProj), nil)
			requireError(t, scope.Transpose(q4d, qHeads, []int{0, 2, 1, 3}), nil)
			requireError(t, scope.Transpose(k4d, kHeads, []int{0, 2, 1, 3}), nil)
			requireError(t, scope.Transpose(v4d, vHeads, []int{0, 2, 1, 3}), nil)
			requireError(t, scope.BatchedMatMulRightTranspose(qHeads, kHeads, scores), nil)
			requireError(t, scope.Scale(scores, scaled, scale), nil)
			requireError(t, scope.Softmax(scaled, probs), nil)
			requireError(t, scope.BatchedMatMul(probs, vHeads, ctxH), nil)
			requireError(t, scope.Transpose(ctxH, ctxS, []int{0, 2, 1, 3}), nil)
			requireError(t, scope.BatchedMatMul(ctx3d, wo, attnOut), nil)
			requireError(t, scope.AddBroadcast(attnOut, bo, attnB), nil)
			requireError(t, scope.Add(x, attnB, res1), nil)
			requireError(t, scope.LayerNorm(res1, ln2W, ln2B, ln2Out, eps), nil)
			requireError(t, scope.BatchedMatMul(ln2Out, w1, ffn1), nil)
			requireError(t, scope.AddBroadcast(ffn1, b1, ffn1B), nil)
			requireError(t, scope.ReLU(ffn1B, ffnAct), nil)
			requireError(t, scope.BatchedMatMul(ffnAct, w2, ffn2), nil)
			requireError(t, scope.AddBroadcast(ffn2, b2, ffn2B), nil)
			requireError(t, scope.Add(res1, ffn2B, finalY), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)

			if afterSubmits := native.ResourceSnapshot().SubmittedCommands; afterSubmits-beforeSubmits != 1 {
				t.Fatalf("expected exactly 1 GPU scope submit, got %d", afterSubmits-beforeSubmits)
			}

			var got = make([]float32, B*S*E)
			requireError(t, finalY.DownloadInto(got), nil)

			// Independent FP64 CPU reference.
			var x64 = make([]float64, len(xData))
			for i, v := range xData {
				x64[i] = float64(v)
			}
			var (
				rLN1   = referenceLayerNormFP64FromFloat64(x64, quantizeRoundTrip(ln1WData, wt), quantizeRoundTrip(ln1BData, wt), int(B*S), int(E), eps)
				rQ     = referenceBatchedMatMulFP64Inputs(rLN1, quantizeRoundTrip(wqData, wt), []int64{B, S, E}, []int64{E, E}, false)
				rK     = referenceBatchedMatMulFP64Inputs(rLN1, quantizeRoundTrip(wkData, wt), []int64{B, S, E}, []int64{E, E}, false)
				rV     = referenceBatchedMatMulFP64Inputs(rLN1, quantizeRoundTrip(wvData, wt), []int64{B, S, E}, []int64{E, E}, false)
				rQH    = referenceTransposeBits(rQ, []int64{B, S, H, D}, []int{0, 2, 1, 3})
				rKH    = referenceTransposeBits(rK, []int64{B, S, H, D}, []int{0, 2, 1, 3})
				rVH    = referenceTransposeBits(rV, []int64{B, S, H, D}, []int{0, 2, 1, 3})
				rScore = referenceBatchedMatMulFP64Both(rQH, rKH, int(B), int(H), int(S), int(D), int(S), true)
			)
			for i := range rScore {
				rScore[i] *= float64(scale)
			}
			var (
				rProb = referenceSoftmaxLastAxisFP64(rScore, int(B*H*S), int(S))
				rCtxH = referenceBatchedMatMulFP64Both(rProb, rVH, int(B), int(H), int(S), int(S), int(D), false)
				rCtxS = referenceTransposeBits(rCtxH, []int64{B, H, S, D}, []int{0, 2, 1, 3})
				rAttn = referenceBatchedMatMulFP64Inputs(rCtxS, quantizeRoundTrip(woData, wt), []int64{B, S, E}, []int64{E, E}, false)
				rRes1 = make([]float64, len(rAttn))
			)
			for i := range rAttn {
				rRes1[i] = x64[i] + rAttn[i] + float64(boData[i%int(E)])
			}
			var (
				rLN2  = referenceLayerNormFP64FromFloat64(rRes1, quantizeRoundTrip(ln2WData, wt), quantizeRoundTrip(ln2BData, wt), int(B*S), int(E), eps)
				rFFN1 = referenceBatchedMatMulFP64Inputs(rLN2, quantizeRoundTrip(w1Data, wt), []int64{B, S, E}, []int64{E, FFN}, false)
			)
			for i := range rFFN1 {
				var v = rFFN1[i] + float64(b1Data[i%int(FFN)])
				if v < 0 {
					v = 0
				}
				rFFN1[i] = v
			}
			var (
				rFFN2 = referenceBatchedMatMulFP64Inputs(rFFN1, quantizeRoundTrip(w2Data, wt), []int64{B, S, FFN}, []int64{FFN, E}, false)
				rOut  = make([]float64, len(rFFN2))
			)
			for i := range rFFN2 {
				rOut[i] = rRes1[i] + rFFN2[i] + float64(b2Data[i%int(E)])
			}

			var p50, p95, p99, maxErr = nn17ErrorStats(got, rOut)
			t.Logf("ONNX Transformer Encoder Block (1 Scope) %s: p50=%.3e p95=%.3e p99=%.3e max=%.3e", wt, p50, p95, p99, maxErr)
			if maxErr > 1e-5 {
				t.Fatalf("Transformer encoder block max error %.3e exceeds 1e-5", maxErr)
			}

			for _, tensor := range []*Tensor{
				q4d, k4d, v4d, ctx3d,
				x, ln1Out, ln1W, ln1B, wq, wk, wv, wo, bo, ln2W, ln2B, w1, b1, w2, b2,
				qProj, kProj, vProj, qHeads, kHeads, vHeads, scores, scaled, probs,
				ctxH, ctxS, attnOut, attnB, res1, ln2Out, ffn1, ffn1B, ffnAct, ffn2, ffn2B, finalY,
			} {
				requireError(t, tensor.Close(), nil)
			}
		})
	}

	if after := native.ResourceSnapshot(); after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("resource leak: initial=%+v after=%+v", initial, after)
	}
}

func Test_MetalMatMulLeftTransposeSpecialValues(t *testing.T) {
	var (
		dev     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = dev.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
	)

	type shapeCase struct {
		name    string
		k, m, n int64
	}
	var shapes = []shapeCase{
		{name: "fallback_2x3x3", k: 2, m: 3, n: 3},
		{name: "tiled_partial_9x8x9", k: 9, m: 8, n: 9},
		{name: "tiled_edge_35x33x65", k: 35, m: 33, n: 65},
	}

	for _, sc := range shapes {
		for _, dt := range []DType{Float32, Float16, BFloat16} {
			t.Run(fmt.Sprintf("%s/%s", sc.name, dt), func(t *testing.T) {
				var (
					leftVals  = make([]float32, sc.k*sc.m)
					rightVals = make([]float32, sc.k*sc.n)
					m         = int(sc.m)
					n         = int(sc.n)
				)
				for i := range leftVals {
					leftVals[i] = 1.0
				}
				for i := range rightVals {
					rightVals[i] = 1.0
				}
				// [0, 0]: NaN * 1 -> NaN.
				leftVals[0*m+0] = float32(math.NaN())
				// [1, 1]: 0 * +Inf -> NaN.
				leftVals[0*m+1] = 0
				rightVals[0*n+1] = float32(math.Inf(1))
				// [2, 0]: +Inf * 1 + 1 * 1 -> +Inf.
				leftVals[0*m+2] = float32(math.Inf(1))
				// [2, 2]: +Inf * 1 + 1 * (-Inf) -> NaN.
				rightVals[1*n+2] = float32(math.Inf(-1))
				// Edge element check on tiled shapes.
				if sc.m > 3 && sc.n > 3 {
					leftVals[(int(sc.k)-1)*m+(m-1)] = 2.0
					rightVals[(int(sc.k)-1)*n+(n-1)] = 3.0
				}

				var (
					left  = requireTensor(t, dev, sc.k, sc.m)
					right = requireTensorWithDType(t, dev, dt, sc.k, sc.n)
					dst   = requireTensor(t, dev, sc.m, sc.n)
				)
				requireError(t, left.Upload(leftVals), nil)
				requireError(t, right.UploadFloat32As(rightVals), nil)
				var scope, err = dev.NewScope()
				requireError(t, err, nil)
				requireError(t, scope.MatMulLeftTranspose(left, right, dst), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)

				var got = make([]float32, sc.m*sc.n)
				requireError(t, dst.DownloadInto(got), nil)

				if !math.IsNaN(float64(got[0*n+0])) {
					t.Fatalf("expected NaN at [0,0], got %v", got[0*n+0])
				}
				if !math.IsNaN(float64(got[1*n+1])) {
					t.Fatalf("expected 0*Inf -> NaN at [1,1], got %v", got[1*n+1])
				}
				if !math.IsInf(float64(got[2*n+0]), 1) {
					t.Fatalf("expected +Inf at [2,0], got %v", got[2*n+0])
				}
				if !math.IsNaN(float64(got[2*n+2])) {
					t.Fatalf("expected opposing Inf -> NaN at [2,2], got %v", got[2*n+2])
				}
				if sc.m > 3 && sc.n > 3 {
					var wantLast = float32(sc.k + 5)
					var gotLast = got[(m-1)*n+(n-1)]
					if math.Abs(float64(gotLast-wantLast)) > 1e-4 {
						t.Fatalf("edge element [%d,%d]: got %v, want %v", m-1, n-1, gotLast, wantLast)
					}
				}

				for _, tensor := range []*Tensor{left, right, dst} {
					requireError(t, tensor.Close(), nil)
				}
			})
		}
	}

	if after := native.ResourceSnapshot(); after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("resource leak: initial=%+v after=%+v", initial, after)
	}
}

func Test_MetalBatchedMatMulSpecialValues(t *testing.T) {
	var (
		dev     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = dev.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
	)

	type shapeCase struct {
		name   string
		b      int64
		m, k   int64
		n      int64
		rightT bool
		bcastR bool
	}
	var cases = []shapeCase{
		{name: "fallback_nn_2x3x2x3", b: 2, m: 3, k: 2, n: 3},
		{name: "fallback_rt_2x3x2x3", b: 2, m: 3, k: 2, n: 3, rightT: true},
		{name: "tiled_partial_nn_2x8x9x9_bcast", b: 2, m: 8, k: 9, n: 9, bcastR: true},
		{name: "tiled_partial_rt_2x8x9x9_bcast", b: 2, m: 8, k: 9, n: 9, rightT: true, bcastR: true},
		{name: "tiled_edge_nn_2x33x35x65", b: 2, m: 33, k: 35, n: 65},
		{name: "tiled_edge_rt_2x33x35x65", b: 2, m: 33, k: 35, n: 65, rightT: true},
	}

	for _, tc := range cases {
		for _, dt := range []DType{Float32, Float16, BFloat16} {
			t.Run(fmt.Sprintf("%s/%s", tc.name, dt), func(t *testing.T) {
				var (
					rBatches  = tc.b
					leftVals  = make([]float32, tc.b*tc.m*tc.k)
					m         = int(tc.m)
					k         = int(tc.k)
					n         = int(tc.n)
					leftShape = []int64{tc.b, tc.m, tc.k}
					dstShape  = []int64{tc.b, tc.m, tc.n}
				)
				if tc.bcastR {
					rBatches = 1
				}
				var rightVals = make([]float32, rBatches*tc.k*tc.n)
				var rightShape []int64
				if tc.rightT {
					if tc.bcastR {
						rightShape = []int64{tc.n, tc.k}
					} else {
						rightShape = []int64{tc.b, tc.n, tc.k}
					}
				} else {
					if tc.bcastR {
						rightShape = []int64{tc.k, tc.n}
					} else {
						rightShape = []int64{tc.b, tc.k, tc.n}
					}
				}

				for i := range leftVals {
					leftVals[i] = 1.0
				}
				for i := range rightVals {
					rightVals[i] = 1.0
				}
				setRight := func(batch, kk, col int, val float32) {
					var base = batch * k * n
					if tc.rightT {
						rightVals[base+col*k+kk] = val
					} else {
						rightVals[base+kk*n+col] = val
					}
				}

				// Batch 0 special cases:
				// [0, 0, 0]: NaN * 1 -> NaN
				leftVals[0*k+0] = float32(math.NaN())
				// [0, 1, 1]: 0 * +Inf -> NaN
				leftVals[1*k+0] = 0
				setRight(0, 0, 1, float32(math.Inf(1)))
				// [0, 2, 0]: +Inf * 1 + 1*1 -> +Inf
				leftVals[2*k+0] = float32(math.Inf(1))
				// [0, 2, 2]: +Inf * 1 + 1 * (-Inf) -> NaN
				setRight(0, 1, 2, float32(math.Inf(-1)))

				// Batch 1 finite check at last element.
				var b1Base = m * k
				leftVals[b1Base+(m-1)*k+(k-1)] = 2.0
				if !tc.bcastR {
					setRight(1, k-1, n-1, 3.0)
				} else {
					setRight(0, k-1, n-1, 3.0)
				}

				var (
					left  = requireTensor(t, dev, leftShape...)
					right = requireTensorWithDType(t, dev, dt, rightShape...)
					dst   = requireTensor(t, dev, dstShape...)
				)
				requireError(t, left.Upload(leftVals), nil)
				requireError(t, right.UploadFloat32As(rightVals), nil)
				var scope, err = dev.NewScope()
				requireError(t, err, nil)
				if tc.rightT {
					requireError(t, scope.BatchedMatMulRightTranspose(left, right, dst), nil)
				} else {
					requireError(t, scope.BatchedMatMul(left, right, dst), nil)
				}
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)

				var got = make([]float32, tc.b*tc.m*tc.n)
				requireError(t, dst.DownloadInto(got), nil)

				if !math.IsNaN(float64(got[0*n+0])) {
					t.Fatalf("expected NaN at [0,0,0], got %v", got[0*n+0])
				}
				if !math.IsNaN(float64(got[1*n+1])) {
					t.Fatalf("expected 0*Inf -> NaN at [0,1,1], got %v", got[1*n+1])
				}
				if !math.IsInf(float64(got[2*n+0]), 1) {
					t.Fatalf("expected +Inf at [0,2,0], got %v", got[2*n+0])
				}
				if !math.IsNaN(float64(got[2*n+2])) {
					t.Fatalf("expected opposing Inf -> NaN at [0,2,2], got %v", got[2*n+2])
				}
				if tc.m > 3 && tc.n > 3 {
					var wantLast = float32(tc.k + 5)
					var gotLast = got[m*n+(m-1)*n+(n-1)]
					if math.Abs(float64(gotLast-wantLast)) > 1e-4 {
						t.Fatalf("batch 1 edge element [%d,%d]: got %v, want %v", m-1, n-1, gotLast, wantLast)
					}
				}

				for _, tensor := range []*Tensor{left, right, dst} {
					requireError(t, tensor.Close(), nil)
				}
			})
		}
	}

	if after := native.ResourceSnapshot(); after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("resource leak: initial=%+v after=%+v", initial, after)
	}
}

func Test_MetalBatchedMatMulTiledConcurrentStress(t *testing.T) {
	var (
		dev     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = dev.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		rng     = rand.New(rand.NewPCG(2301, 2302))
	)

	const (
		kDim = 67
		mDim = 33
		nDim = 65
	)

	var (
		ltL  = requireTensor(t, dev, kDim, mDim)
		ltR  = requireTensorWithDType(t, dev, Float16, kDim, nDim)
		nnL  = requireTensor(t, dev, 2, mDim, kDim)
		nnR  = requireTensorWithDType(t, dev, BFloat16, kDim, nDim)
		rtL  = requireTensor(t, dev, 2, 2, mDim, kDim)
		rtR  = requireTensorWithDType(t, dev, Float32, 2, 2, nDim, kDim)
		ltLV = make([]float32, kDim*mDim)
		ltRV = make([]float32, kDim*nDim)
		nnLV = make([]float32, 2*mDim*kDim)
		nnRV = make([]float32, kDim*nDim)
		rtLV = make([]float32, 4*mDim*kDim)
		rtRV = make([]float32, 4*nDim*kDim)
	)
	for _, buf := range [][]float32{ltLV, ltRV, nnLV, nnRV, rtLV, rtRV} {
		for i := range buf {
			buf[i] = float32(rng.Float64()*2 - 1)
		}
	}
	requireError(t, ltL.Upload(ltLV), nil)
	requireError(t, ltR.UploadFloat32As(ltRV), nil)
	requireError(t, nnL.Upload(nnLV), nil)
	requireError(t, nnR.UploadFloat32As(nnRV), nil)
	requireError(t, rtL.Upload(rtLV), nil)
	requireError(t, rtR.UploadFloat32As(rtRV), nil)

	var (
		wantLT    = referenceMatMulLeftTransposeFP64(ltLV, quantizeRoundTrip(ltRV, Float16), kDim, mDim, nDim)
		wantNN, _ = referenceBatchedMatMulFP64(nnLV, quantizeRoundTrip(nnRV, BFloat16), []int64{2, mDim, kDim}, []int64{kDim, nDim}, false)
		wantRT, _ = referenceBatchedMatMulFP64(rtLV, rtRV, []int64{2, 2, mDim, kDim}, []int64{2, 2, nDim, kDim}, true)
	)

	const (
		workers = 8
		iters   = 16
	)
	var (
		wg    sync.WaitGroup
		errCh = make(chan error, workers*iters)
	)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var (
				ltDst = requireTensor(t, dev, mDim, nDim)
				nnDst = requireTensor(t, dev, 2, mDim, nDim)
				rtDst = requireTensor(t, dev, 2, 2, mDim, nDim)
				gotLT = make([]float32, len(wantLT))
				gotNN = make([]float32, len(wantNN))
				gotRT = make([]float32, len(wantRT))
			)
			defer func() {
				_ = ltDst.Close()
				_ = nnDst.Close()
				_ = rtDst.Close()
			}()
			for range iters {
				var scope, err = dev.NewScope()
				if err != nil {
					errCh <- err
					return
				}
				if err = scope.MatMulLeftTranspose(ltL, ltR, ltDst); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.BatchedMatMul(nnL, nnR, nnDst); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.BatchedMatMulRightTranspose(rtL, rtR, rtDst); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.Submit(ctx); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.Wait(ctx); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.Close(); err != nil {
					errCh <- err
					return
				}
				if err = ltDst.DownloadInto(gotLT); err != nil {
					errCh <- err
					return
				}
				if err = nnDst.DownloadInto(gotNN); err != nil {
					errCh <- err
					return
				}
				if err = rtDst.DownloadInto(gotRT); err != nil {
					errCh <- err
					return
				}
				if _, _, _, maxErr := nn17ErrorStats(gotLT, wantLT); maxErr > 1e-4 {
					errCh <- fmt.Errorf("tiled concurrent left-transpose max err %.3e", maxErr)
					return
				}
				if _, _, _, maxErr := nn17ErrorStats(gotNN, wantNN); maxErr > 1e-4 {
					errCh <- fmt.Errorf("tiled concurrent batched nn max err %.3e", maxErr)
					return
				}
				if _, _, _, maxErr := nn17ErrorStats(gotRT, wantRT); maxErr > 1e-4 {
					errCh <- fmt.Errorf("tiled concurrent batched rt max err %.3e", maxErr)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
	for _, tensor := range []*Tensor{ltL, ltR, nnL, nnR, rtL, rtR} {
		requireError(t, tensor.Close(), nil)
	}
	if after := native.ResourceSnapshot(); after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("resource leak: initial=%+v after=%+v", initial, after)
	}
}
