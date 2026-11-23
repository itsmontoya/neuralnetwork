//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func Test_MetalTensorMatMul(t *testing.T) {
	var (
		gpu     = requireMetalDeviceForMatMul(t)
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
	)
	var cases = []struct {
		name              string
		rows, inner, cols int64
		transposed, zero  bool
	}{
		{name: "scalar", rows: 1, inner: 1, cols: 1},
		{name: "zero", rows: 3, inner: 5, cols: 7, zero: true},
		{name: "tall_skinny", rows: 33, inner: 3, cols: 1},
		{name: "standard_tails", rows: 17, inner: 35, cols: 19},
		{name: "transposed_tails", rows: 31, inner: 257, cols: 33, transposed: true},
		{name: "long_contraction", rows: 2, inner: 8193, cols: 5},
		{name: "projection_decode", rows: 1, inner: 2048, cols: 8192, transposed: true},
		{name: "projection_prefill_27", rows: 27, inner: 2048, cols: 8192, transposed: true},
		{name: "projection_prefill_28", rows: 28, inner: 2048, cols: 8192, transposed: true},
		{name: "projection_prefill_29", rows: 29, inner: 2048, cols: 8192, transposed: true},
		{name: "projection_down", rows: 29, inner: 8192, cols: 2048, transposed: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var (
				left   = make([]float32, tc.rows*tc.inner)
				weight = make([]float32, tc.inner*tc.cols)
				random = rand.New(rand.NewPCG(0x6e6e03, 0x281d2000))
			)
			if !tc.zero {
				for i := range left {
					left[i] = 2*random.Float32() - 1
				}
			}
			for i := range weight {
				weight[i] = 2*random.Float32() - 1
			}
			var got = runMetalMatMul(t, gpu, tc.rows, tc.inner, tc.cols, left, weight, tc.transposed)
			checkMetalMatMul(t, tc.inner, tc.cols, left, weight, got)
		})
	}
	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("matmul leaked native resources: before %+v; after %+v", initial, after)
	}
}

func Test_MetalTensorMatMulSpecialValues(t *testing.T) {
	var (
		gpu   = requireMetalDeviceForMatMul(t)
		inf   = float32(math.Inf(1))
		nan   = float32(math.NaN())
		large = float32(math.MaxFloat32 / 4)
	)
	var cases = []struct {
		name        string
		left, right []float32
		want        float32
	}{
		{name: "nan", left: []float32{1, 2}, right: []float32{nan, 3}, want: nan},
		{name: "positive_infinity", left: []float32{1, 2}, right: []float32{inf, 3}, want: inf},
		{name: "negative_infinity", left: []float32{-1, 2}, right: []float32{inf, 3}, want: -inf},
		{name: "zero_times_infinity", left: []float32{0, 2}, right: []float32{inf, 3}, want: nan},
		{name: "opposing_infinities", left: []float32{1, 2}, right: []float32{inf, -inf}, want: nan},
		{name: "large_finite", left: []float32{large, large}, right: []float32{1, 2}, want: 3 * large},
		{name: "finite_cancellation", left: []float32{large, -large, 1e-30}, right: []float32{1, 1, 1}, want: 1e-30},
		{name: "small_normal", left: []float32{1e-30, 0}, right: []float32{1e-6, 1}, want: float32(1e-30) * float32(1e-6)},
		{name: "overflow", left: []float32{math.MaxFloat32, math.MaxFloat32}, right: []float32{1, 1}, want: inf},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, transposed := range []bool{false, true} {
				var got = runMetalMatMul(t, gpu, 1, int64(len(tc.left)), 1, tc.left, tc.right, transposed)[0]
				if got != tc.want && !(math.IsNaN(float64(got)) && math.IsNaN(float64(tc.want))) {
					t.Fatalf("right transposed %t: got %g, want %g", transposed, got, tc.want)
				}
			}
		})
	}
}

func Test_MetalTensorMatMulSharedWeightsAndOrdering(t *testing.T) {
	var (
		gpu          = requireMetalDeviceForMatMul(t)
		ctx          = context.Background()
		weight       = requireTensor(t, gpu, 2, 2)
		firstOutput  = requireTensor(t, gpu, 1, 2)
		secondOutput = requireTensor(t, gpu, 1, 2)
	)
	requireError(t, weight.Upload([]float32{1, -2, 3, 4}), nil)
	for i := 0; i < 3; i++ {
		var (
			first                     = requireScope(t, gpu)
			second                    = requireScope(t, gpu)
			firstInput                = requireTensor(t, gpu, 1, 2)
			secondInput               = requireTensor(t, gpu, 1, 2)
			intermediate              = requireTensor(t, gpu, 1, 2)
			firstResult, secondResult [2]float32
			results                   = make(chan error, 2)
		)
		requireError(t, first.Fill(firstInput, 2), nil)
		requireError(t, first.MatMulRightTranspose(firstInput, weight, intermediate), nil)
		requireError(t, first.MatMul(intermediate, weight, firstOutput), nil)
		requireError(t, second.Fill(secondInput, -1), nil)
		requireError(t, second.MatMul(secondInput, weight, secondOutput), nil)
		requireError(t, weight.Upload([]float32{0, 0, 0, 0}), ErrBusy)
		requireError(t, firstOutput.DownloadInto(firstResult[:]), ErrBusy)
		requireError(t, first.Submit(ctx), nil)
		requireError(t, second.Submit(ctx), nil)
		for _, input := range []*Tensor{firstInput, secondInput, intermediate} {
			requireError(t, input.Close(), nil)
		}
		go func() { results <- first.Wait(ctx) }()
		go func() { results <- second.Wait(ctx) }()
		requireError(t, <-results, nil)
		requireError(t, <-results, nil)
		requireError(t, first.Close(), nil)
		requireError(t, second.Close(), nil)
		requireError(t, firstOutput.DownloadInto(firstResult[:]), nil)
		requireError(t, secondOutput.DownloadInto(secondResult[:]), nil)
		if firstResult != [2]float32{40, 60} || secondResult != [2]float32{-4, -2} {
			t.Fatalf("iteration %d: results = %v and %v", i, firstResult, secondResult)
		}
	}
}

func requireMetalDeviceForMatMul(t *testing.T) (gpu *Device) {
	t.Helper()
	var err error
	gpu, err = OpenDefault()
	if errors.Is(err, ErrUnavailable) {
		t.Skip("default Metal device unavailable")
	}
	requireError(t, err, nil)
	t.Cleanup(func() { requireError(t, gpu.Close(), nil) })
	return gpu
}

// weight is supplied in one logical [N,K] order for the reference. Standard
// matmul uploads the same values in physical [K,N] order.
func runMetalMatMul(t *testing.T, gpu *Device, rows, inner, cols int64, values, weight []float32, transposed bool) (got []float32) {
	t.Helper()
	var (
		left        = requireTensor(t, gpu, rows, inner)
		destination = requireTensor(t, gpu, rows, cols)
		scope       = requireScope(t, gpu)
		ctx         = context.Background()
	)
	requireError(t, left.Upload(values), nil)
	if transposed {
		var right = requireTensor(t, gpu, cols, inner)
		requireError(t, right.Upload(weight), nil)
		requireError(t, scope.MatMulRightTranspose(left, right, destination), nil)
	} else {
		var (
			right    = requireTensor(t, gpu, inner, cols)
			physical = make([]float32, len(weight))
		)
		for col := range cols {
			for k := range inner {
				physical[k*cols+col] = weight[col*inner+k]
			}
		}
		requireError(t, right.Upload(physical), nil)
		requireError(t, scope.MatMul(left, right, destination), nil)
	}
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)
	got = make([]float32, rows*cols)
	requireError(t, destination.DownloadInto(got), nil)
	return got
}

func checkMetalMatMul(t *testing.T, inner, cols int64, left, weight, got []float32) {
	t.Helper()
	var deltas = make([]float64, len(got))
	for i, value := range got {
		var (
			row, col = int64(i) / cols, int64(i) % cols
			values   = left[row*inner : (row+1)*inner]
			weights  = weight[col*inner : (col+1)*inner]
			want     float64
		)
		for k, v := range values {
			want += float64(v) * float64(weights[k])
		}
		var delta = math.Abs(float64(value) - want)
		var limit = 4e-4 + 3e-5*math.Abs(want)
		if math.IsNaN(delta) || delta > limit {
			t.Fatalf("output [%d,%d] got %.9g, FP64 reference %.9g, absolute error %.3g exceeds %.3g", row, col, value, want, delta, limit)
		}
		deltas[i] = delta
	}
	slices.Sort(deltas)
	t.Logf("absolute error vs FP64 for %d outputs: p50 %.3g, p95 %.3g, p99 %.3g, max %.3g", len(deltas),
		deltas[(len(deltas)-1)*50/100], deltas[(len(deltas)-1)*95/100], deltas[(len(deltas)-1)*99/100], deltas[len(deltas)-1])
}
