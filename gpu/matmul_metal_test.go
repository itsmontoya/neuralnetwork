//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
	"testing"
)

func Test_MetalTensorMatMul(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
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
		{name: "row_rt_k1_n1", rows: 1, inner: 1, cols: 1, transposed: true},
		{name: "row_rt_k3_n1", rows: 1, inner: 3, cols: 1, transposed: true},
		{name: "row_rt_k4_n1", rows: 1, inner: 4, cols: 1, transposed: true},
		{name: "row_rt_k4_n3", rows: 1, inner: 4, cols: 3, transposed: true},
		{name: "row_rt_k4_n9", rows: 1, inner: 4, cols: 9, transposed: true},
		{name: "row_rt_k31_n17", rows: 1, inner: 31, cols: 17, transposed: true},
		{name: "row_rt_k32_n33", rows: 1, inner: 32, cols: 33, transposed: true},
		{name: "row_rt_k128_n15", rows: 1, inner: 128, cols: 15, transposed: true},
		{name: "row_rt_k132_n18", rows: 1, inner: 132, cols: 18, transposed: true},
		{name: "row_rt_k257_n33", rows: 1, inner: 257, cols: 33, transposed: true},
		{name: "projection_attn_kv_decode", rows: 1, inner: 2048, cols: 512, transposed: true},
		{name: "projection_attn_qo_decode", rows: 1, inner: 2048, cols: 2048, transposed: true},
		{name: "projection_decode", rows: 1, inner: 2048, cols: 8192, transposed: true},
		{name: "projection_down_decode", rows: 1, inner: 8192, cols: 2048, transposed: true},
		{name: "projection_prefill_6", rows: 6, inner: 2048, cols: 8192, transposed: true},
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

func Test_MetalTensorMatMulFloat32Paths(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
	)
	var cases = []struct {
		name              string
		rows, inner, cols int64
	}{
		{name: "m1_k1_n1", rows: 1, inner: 1, cols: 1},
		{name: "m1_k3_n7", rows: 1, inner: 3, cols: 7},
		{name: "m1_k7_n3", rows: 1, inner: 7, cols: 3},
		{name: "m1_k16_n16", rows: 1, inner: 16, cols: 16},
		{name: "m1_k31_n31", rows: 1, inner: 31, cols: 31},
		{name: "m1_k32_n32", rows: 1, inner: 32, cols: 32},
		{name: "m1_k33_n33", rows: 1, inner: 33, cols: 33},
		{name: "m2_k16_n16_fallback", rows: 2, inner: 16, cols: 16},
		{name: "m3_k33_n33_fallback", rows: 3, inner: 33, cols: 33},
		{name: "m4_k7_n33_small_k_fallback", rows: 4, inner: 7, cols: 33},
		{name: "m4_k33_n7_small_n_fallback", rows: 4, inner: 33, cols: 7},
		{name: "m4_k16_n16_tiled_min", rows: 4, inner: 16, cols: 16},
		{name: "m6_k31_n33_tiled", rows: 6, inner: 31, cols: 33},
		{name: "m7_k32_n32_tiled", rows: 7, inner: 32, cols: 32},
		{name: "m8_k33_n33_tiled", rows: 8, inner: 33, cols: 33},
		{name: "m15_k16_n31_tiled", rows: 15, inner: 16, cols: 31},
		{name: "m16_k32_n32_tiled", rows: 16, inner: 32, cols: 32},
		{name: "m17_k33_n33_tiled", rows: 17, inner: 33, cols: 33},
		{name: "m27_k257_n512_tiled", rows: 27, inner: 257, cols: 512},
		{name: "m28_k768_n512_tiled", rows: 28, inner: 768, cols: 512},
		{name: "m29_k768_n512_tiled", rows: 29, inner: 768, cols: 512},
		{name: "m31_k31_n33_tiled", rows: 31, inner: 31, cols: 33},
		{name: "m32_k32_n32_tiled", rows: 32, inner: 32, cols: 32},
		{name: "m33_k33_n65_tiled_edge", rows: 33, inner: 33, cols: 65},
		{name: "m64_k257_n129_tiled", rows: 64, inner: 257, cols: 129},
		{name: "m128_k64_n128_tiled", rows: 128, inner: 64, cols: 128},
		{name: "m32_k2048_n2048_tiled", rows: 32, inner: 2048, cols: 2048},
		{name: "m4_k8192_n512_tiled", rows: 4, inner: 8192, cols: 512},
		{name: "m4_k2048_n8192_tiled", rows: 4, inner: 2048, cols: 8192},
	}
	for _, tc := range cases {
		for _, transposed := range []bool{false, true} {
			var label = fmt.Sprintf("%s_rt_%t", tc.name, transposed)
			t.Run(label, func(t *testing.T) {
				var (
					left   = make([]float32, tc.rows*tc.inner)
					weight = make([]float32, tc.inner*tc.cols)
					random = rand.New(rand.NewPCG(0x6e6e30, uint64(tc.rows*1000003+tc.inner*1009+tc.cols)))
				)
				for i := range left {
					left[i] = 2*random.Float32() - 1
				}
				for i := range weight {
					weight[i] = 2*random.Float32() - 1
				}
				var got = runMetalMatMul(t, gpu, tc.rows, tc.inner, tc.cols, left, weight, transposed)
				checkMetalMatMul(t, tc.inner, tc.cols, left, weight, got)
			})
		}
	}
	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("fp32 matmul paths leaked native resources: before %+v; after %+v", initial, after)
	}
}

func Test_MetalTensorMatMulTiledSpecialValues(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		inf     = float32(math.Inf(1))
		nan     = float32(math.NaN())
		large   = float32(math.MaxFloat32 / 4)
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
	var shapes = []struct {
		name              string
		rows, inner, cols int64
	}{
		{name: "fallback_2x4x4", rows: 2, inner: 4, cols: 4},
		{name: "tiled_partial_8x9x9", rows: 8, inner: 9, cols: 9},
		{name: "tiled_edge_33x35x65", rows: 33, inner: 35, cols: 65},
	}
	for _, sh := range shapes {
		for _, transposed := range []bool{false, true} {
			for _, tc := range cases {
				var label = fmt.Sprintf("%s_rt_%t/%s", sh.name, transposed, tc.name)
				t.Run(label, func(t *testing.T) {
					var (
						left   = make([]float32, sh.rows*sh.inner)
						weight = make([]float32, sh.cols*sh.inner)
					)
					// Place the special-value test vector in both the first and last
					// row/column so interior tiles and boundary edge tiles are covered.
					var targetRows = []int64{0, sh.rows - 1}
					var targetCols = []int64{0, sh.cols - 1}
					for _, r := range targetRows {
						copy(left[r*sh.inner:r*sh.inner+int64(len(tc.left))], tc.left)
					}
					for _, c := range targetCols {
						copy(weight[c*sh.inner:c*sh.inner+int64(len(tc.right))], tc.right)
					}
					var got = runMetalMatMul(t, gpu, sh.rows, sh.inner, sh.cols, left, weight, transposed)
					for _, r := range targetRows {
						for _, c := range targetCols {
							var val = got[r*sh.cols+c]
							if val != tc.want && !(math.IsNaN(float64(val)) && math.IsNaN(float64(tc.want))) {
								t.Fatalf("[%d,%d]: got %g, want %g", r, c, val, tc.want)
							}
						}
					}
				})
			}
		}
	}
	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("special-value fp32 matmul leaked native resources: before %+v; after %+v", initial, after)
	}
}

func Test_MetalTensorMatMulFloat32ConcurrentStress(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		ctx     = context.Background()
		workers = 8
	)
	var modes = []struct {
		name              string
		rows, inner, cols int64
		transposed        bool
	}{
		{name: "gemv_m1_rt", rows: 1, inner: 131, cols: 67, transposed: true},
		{name: "tiled_m33_nn", rows: 33, inner: 131, cols: 67, transposed: false},
		{name: "tiled_m33_nt", rows: 33, inner: 131, cols: 67, transposed: true},
	}
	for _, mode := range modes {
		t.Run(mode.name, func(t *testing.T) {
			var (
				random  = rand.New(rand.NewPCG(0x6e6e30, uint64(mode.rows*97+mode.cols)))
				leftVal = make([]float32, mode.rows*mode.inner)
				weightV = make([]float32, mode.cols*mode.inner)
				weight  *Tensor
			)
			for i := range leftVal {
				leftVal[i] = 2*random.Float32() - 1
			}
			for i := range weightV {
				weightV[i] = 2*random.Float32() - 1
			}
			if mode.transposed {
				weight = requireTensor(t, gpu, mode.cols, mode.inner)
				requireError(t, weight.Upload(weightV), nil)
			} else {
				var physical = make([]float32, len(weightV))
				for col := range mode.cols {
					for k := range mode.inner {
						physical[k*mode.cols+col] = weightV[col*mode.inner+k]
					}
				}
				weight = requireTensor(t, gpu, mode.inner, mode.cols)
				requireError(t, weight.Upload(physical), nil)
			}

			var (
				wg   sync.WaitGroup
				errs = make(chan error, workers)
			)
			for worker := range workers {
				wg.Add(1)
				go func(workerID int) {
					defer wg.Done()
					var (
						left, errL = gpu.NewFloat32([]int64{mode.rows, mode.inner})
						out, errO  = gpu.NewFloat32([]int64{mode.rows, mode.cols})
					)
					if errL != nil {
						errs <- errL
						return
					}
					defer left.Close()
					if errO != nil {
						errs <- errO
						return
					}
					defer out.Close()
					if err := left.Upload(leftVal); err != nil {
						errs <- err
						return
					}
					for iter := range 4 {
						var scope, errS = gpu.NewScope()
						if errS != nil {
							errs <- errS
							return
						}
						var errOp error
						if mode.transposed {
							errOp = scope.MatMulRightTranspose(left, weight, out)
						} else {
							errOp = scope.MatMul(left, weight, out)
						}
						if errOp != nil {
							_ = scope.Close()
							errs <- errOp
							return
						}
						if err := scope.Submit(ctx); err != nil {
							_ = scope.Close()
							errs <- err
							return
						}
						if err := scope.Wait(ctx); err != nil {
							_ = scope.Close()
							errs <- err
							return
						}
						if err := scope.Close(); err != nil {
							errs <- err
							return
						}
						if iter == 3 {
							var got = make([]float32, mode.rows*mode.cols)
							if err := out.DownloadInto(got); err != nil {
								errs <- err
								return
							}
							for row := range mode.rows {
								for col := range mode.cols {
									var want float64
									for k := range mode.inner {
										want += float64(leftVal[row*mode.inner+k]) * float64(weightV[col*mode.inner+k])
									}
									var diff = math.Abs(float64(got[row*mode.cols+col]) - want)
									var limit = 4e-4 + 3e-5*math.Abs(want)
									if math.IsNaN(diff) || diff > limit {
										errs <- fmt.Errorf("worker %d (%d,%d): got %g, want %g (diff %g)", workerID, row, col, got[row*mode.cols+col], want, diff)
										return
									}
								}
							}
						}
					}
				}(worker)
			}
			wg.Wait()
			close(errs)
			for err := range errs {
				t.Fatal(err)
			}
		})
	}
	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("concurrent fp32 matmul leaked native resources: before %+v; after %+v", initial, after)
	}
}

func Test_MetalTensorMatMulSpecialValues(t *testing.T) {
	var (
		gpu   = requireMetalDevice(t)
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
			// Pad to K=4, N=2 to also exercise the vectorized M=1 right-transposed fast path.
			var (
				paddedLeft  = make([]float32, 4)
				paddedRight = make([]float32, 8)
			)
			copy(paddedLeft, tc.left)
			copy(paddedRight[:4], tc.right)
			copy(paddedRight[4:], tc.right)
			var gotPair = runMetalMatMul(t, gpu, 1, 4, 2, paddedLeft, paddedRight, true)
			for col, got := range gotPair {
				if got != tc.want && !(math.IsNaN(float64(got)) && math.IsNaN(float64(tc.want))) {
					t.Fatalf("vectorized M=1 right transposed col %d: got %g, want %g", col, got, tc.want)
				}
			}
		})
	}
}

func Test_MetalTensorMatMulSharedWeightsAndOrdering(t *testing.T) {
	var (
		gpu          = requireMetalDevice(t)
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
