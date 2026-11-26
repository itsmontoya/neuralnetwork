//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"sync"
	"testing"
)

func Test_MetalTensorMatMulMixed(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
	)
	type shapeCase struct {
		name              string
		rows, inner, cols int64
		transposed, zero  bool
	}
	var cases = []shapeCase{
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
		{name: "projection_gate_up_decode", rows: 1, inner: 2048, cols: 8192, transposed: true},
		{name: "projection_down_decode", rows: 1, inner: 8192, cols: 2048, transposed: true},
		{name: "projection_classifier_decode", rows: 1, inner: 2048, cols: 128256, transposed: true},
		{name: "projection_prefill_6", rows: 6, inner: 2048, cols: 8192, transposed: true},
		{name: "projection_prefill_27", rows: 27, inner: 2048, cols: 8192, transposed: true},
		{name: "projection_prefill_28", rows: 28, inner: 2048, cols: 8192, transposed: true},
		{name: "projection_prefill_29", rows: 29, inner: 2048, cols: 8192, transposed: true},
		{name: "projection_down_prefill_29", rows: 29, inner: 8192, cols: 2048, transposed: true},
		{name: "tiled_m2_k16_n16_nn", rows: 2, inner: 16, cols: 16},
		{name: "tiled_m3_k31_n31_nt", rows: 3, inner: 31, cols: 31, transposed: true},
		{name: "tiled_m4_k8_n8_nn", rows: 4, inner: 8, cols: 8},
		{name: "tiled_m4_k8_n8_nt", rows: 4, inner: 8, cols: 8, transposed: true},
		{name: "tiled_m4_k7_n7_nn", rows: 4, inner: 7, cols: 7},
		{name: "tiled_m7_k33_n65_nn", rows: 7, inner: 33, cols: 65},
		{name: "tiled_m7_k33_n65_nt", rows: 7, inner: 33, cols: 65, transposed: true},
		{name: "tiled_m15_k16_n31_nn", rows: 15, inner: 16, cols: 31},
		{name: "tiled_m16_k32_n32_nt", rows: 16, inner: 32, cols: 32, transposed: true},
		{name: "tiled_m17_k33_n33_nn", rows: 17, inner: 33, cols: 33},
		{name: "tiled_m17_k33_n33_nt", rows: 17, inner: 33, cols: 33, transposed: true},
		{name: "tiled_m31_k257_n33_nn", rows: 31, inner: 257, cols: 33},
		{name: "tiled_m32_k32_n64_nn", rows: 32, inner: 32, cols: 64},
		{name: "tiled_m32_k32_n64_nt", rows: 32, inner: 32, cols: 64, transposed: true},
		{name: "tiled_m33_k65_n65_nn", rows: 33, inner: 65, cols: 65},
		{name: "tiled_m33_k65_n65_nt", rows: 33, inner: 65, cols: 65, transposed: true},
		{name: "tiled_m64_k128_n128_nn", rows: 64, inner: 128, cols: 128},
		{name: "tiled_m64_k128_n128_nt", rows: 64, inner: 128, cols: 128, transposed: true},
		{name: "tiled_m128_k64_n128_nn", rows: 128, inner: 64, cols: 128},
		{name: "tiled_m128_k64_n128_nt", rows: 128, inner: 64, cols: 128, transposed: true},
	}

	for _, dtype := range []DType{Float16, BFloat16} {
		t.Run(dtype.String(), func(t *testing.T) {
			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					var (
						left    = make([]float32, tc.rows*tc.inner)
						weight  = make([]float32, tc.inner*tc.cols)
						dequant = make([]float32, tc.inner*tc.cols)
						random  = rand.New(rand.NewPCG(0x6e6e10, uint64(tc.rows*1000003+tc.inner*1009+tc.cols)))
						got     []float32
					)
					if !tc.zero {
						for i := range left {
							left[i] = 2*random.Float32() - 1
						}
					}
					for i := range weight {
						weight[i] = 2*random.Float32() - 1
						dequant[i] = roundTripDType(weight[i], dtype)
					}
					got = runMetalMatMulMixed(t, gpu, dtype, tc.rows, tc.inner, tc.cols, left, weight, tc.transposed)
					checkMetalMatMul(t, tc.inner, tc.cols, left, dequant, got)
				})
			}
		})
	}

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("mixed matmul leaked native resources: before %+v; after %+v", initial, after)
	}
}

func Test_MetalTensorMatMulMixedSpecialValues(t *testing.T) {
	var (
		gpu   = requireMetalDevice(t)
		inf   = float32(math.Inf(1))
		nan   = float32(math.NaN())
		large = float32(math.MaxFloat32 / 4)
	)
	type specialCase struct {
		name        string
		left, right []float32
		want        float32
	}
	var cases = []specialCase{
		{name: "nan", left: []float32{1, 2}, right: []float32{nan, 3}, want: nan},
		{name: "positive_infinity", left: []float32{1, 2}, right: []float32{inf, 3}, want: inf},
		{name: "negative_infinity", left: []float32{-1, 2}, right: []float32{inf, 3}, want: -inf},
		{name: "zero_times_infinity", left: []float32{0, 2}, right: []float32{inf, 3}, want: nan},
		{name: "opposing_infinities", left: []float32{1, 2}, right: []float32{inf, -inf}, want: nan},
		{name: "large_finite", left: []float32{large, large}, right: []float32{1, 2}, want: 3 * large},
		{name: "finite_cancellation", left: []float32{large, -large, 1e-30}, right: []float32{1, 1, 1}, want: 1e-30},
		{name: "overflow", left: []float32{math.MaxFloat32, math.MaxFloat32}, right: []float32{1, 1}, want: inf},
	}

	for _, dtype := range []DType{Float16, BFloat16} {
		t.Run(dtype.String(), func(t *testing.T) {
			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					for _, transposed := range []bool{false, true} {
						var got = runMetalMatMulMixed(t, gpu, dtype, 1, int64(len(tc.left)), 1, tc.left, tc.right, transposed)[0]
						if !sameSpecialFloat32(got, tc.want) {
							t.Fatalf("M=1 right transposed %t: got %g, want %g", transposed, got, tc.want)
						}
					}

					// Pad to K=4, N=2 to exercise the vectorized M=1 right-transposed fast path.
					var (
						paddedLeft  = make([]float32, 4)
						paddedRight = make([]float32, 8)
					)
					copy(paddedLeft, tc.left)
					copy(paddedRight[:4], tc.right)
					copy(paddedRight[4:], tc.right)
					var gotPair = runMetalMatMulMixed(t, gpu, dtype, 1, 4, 2, paddedLeft, paddedRight, true)
					for col, got := range gotPair {
						if !sameSpecialFloat32(got, tc.want) {
							t.Fatalf("vectorized M=1 right transposed col %d: got %g, want %g", col, got, tc.want)
						}
					}

					// Multi-row M=2 fallback path for both orientations.
					var multiLeft = make([]float32, 2*len(tc.left))
					copy(multiLeft[:len(tc.left)], tc.left)
					copy(multiLeft[len(tc.left):], tc.left)
					for _, transposed := range []bool{false, true} {
						var gotMulti = runMetalMatMulMixed(t, gpu, dtype, 2, int64(len(tc.left)), 1, multiLeft, tc.right, transposed)
						for row, got := range gotMulti {
							if !sameSpecialFloat32(got, tc.want) {
								t.Fatalf("M=2 right transposed %t row %d: got %g, want %g", transposed, row, got, tc.want)
							}
						}
					}

					// Tiled simdgroup path: partial tile (M=8, K=9, N=9) and multi-tile edge (M=33, K=35, N=65).
					type tiledSpec struct {
						name              string
						rows, inner, cols int64
					}
					var tiledSpecs = []tiledSpec{
						{name: "tiled_partial_8x9x9", rows: 8, inner: 9, cols: 9},
						{name: "tiled_edge_33x35x65", rows: 33, inner: 35, cols: 65},
					}
					for _, spec := range tiledSpecs {
						var (
							tiledLeft  = make([]float32, spec.rows*spec.inner)
							tiledRight = make([]float32, spec.cols*spec.inner)
						)
						for row := range spec.rows {
							copy(tiledLeft[row*spec.inner:], tc.left)
						}
						for col := range spec.cols {
							copy(tiledRight[col*spec.inner:], tc.right)
						}
						for _, transposed := range []bool{false, true} {
							var gotTiled = runMetalMatMulMixed(t, gpu, dtype, spec.rows, spec.inner, spec.cols, tiledLeft, tiledRight, transposed)
							for idx, got := range gotTiled {
								if !sameSpecialFloat32(got, tc.want) {
									t.Fatalf("%s right transposed %t idx %d: got %g, want %g", spec.name, transposed, idx, got, tc.want)
								}
							}
						}
					}
				})
			}
		})
	}
}

func Test_MetalTensorMatMulMixedSharedWeightsAndConcurrency(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		ctx     = context.Background()
	)
	for _, dtype := range []DType{Float16, BFloat16} {
		t.Run(dtype.String(), func(t *testing.T) {
			var (
				weight       = requireTensorWithDType(t, gpu, dtype, 2, 2)
				firstOutput  = requireTensor(t, gpu, 1, 2)
				secondOutput = requireTensor(t, gpu, 1, 2)
			)
			if weight.Bytes() != 8 {
				t.Fatalf("weight bytes = %d, want 8 (exact 50%% of FP32)", weight.Bytes())
			}
			requireError(t, weight.UploadFloat32As([]float32{1, -2, 3, 4}), nil)

			for i := range 3 {
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
				requireError(t, weight.UploadFloat32As([]float32{0, 0, 0, 0}), ErrBusy)
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
					t.Fatalf("iteration %d: results = %v and %v, want [40 60] and [-4 -2]", i, firstResult, secondResult)
				}
			}
		})
	}
	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("shared mixed matmul leaked native resources: before %+v; after %+v", initial, after)
	}
}

func Test_MetalTensorMatMulMixedTiledConcurrentStress(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		ctx     = context.Background()
		rows    = int64(33)
		inner   = int64(131)
		cols    = int64(67)
		workers = 8
	)
	for _, dtype := range []DType{Float16, BFloat16} {
		for _, transposed := range []bool{false, true} {
			var label = fmt.Sprintf("%s_rt_%t", dtype, transposed)
			t.Run(label, func(t *testing.T) {
				var (
					random  = rand.New(rand.NewPCG(0x6e6e21, uint64(dtype)*17))
					leftVal = make([]float32, rows*inner)
					weightV = make([]float32, cols*inner)
					dequant = make([]float32, cols*inner)
					weight  *Tensor
				)
				for i := range leftVal {
					leftVal[i] = 2*random.Float32() - 1
				}
				for i := range weightV {
					weightV[i] = 2*random.Float32() - 1
					dequant[i] = roundTripDType(weightV[i], dtype)
				}
				if transposed {
					weight = requireTensorWithDType(t, gpu, dtype, cols, inner)
					requireError(t, weight.UploadFloat32As(weightV), nil)
				} else {
					var physical = make([]float32, len(weightV))
					for col := range cols {
						for k := range inner {
							physical[k*cols+col] = weightV[col*inner+k]
						}
					}
					weight = requireTensorWithDType(t, gpu, dtype, inner, cols)
					requireError(t, weight.UploadFloat32As(physical), nil)
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
							left, errL = gpu.NewFloat32([]int64{rows, inner})
							out, errO  = gpu.NewFloat32([]int64{rows, cols})
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
							if transposed {
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
								var got = make([]float32, rows*cols)
								if err := out.DownloadInto(got); err != nil {
									errs <- err
									return
								}
								for row := range rows {
									for col := range cols {
										var want float64
										for k := range inner {
											want += float64(leftVal[row*inner+k]) * float64(dequant[col*inner+k])
										}
										var diff = math.Abs(float64(got[row*cols+col]) - want)
										if diff > 1e-3 {
											errs <- fmt.Errorf("worker %d (%d,%d): got %g, want %g (diff %g)", workerID, row, col, got[row*cols+col], want, diff)
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
	}
	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("concurrent tiled mixed matmul leaked native resources: before %+v; after %+v", initial, after)
	}
}

func Test_MetalFFNMixedPrecision(t *testing.T) {
	var (
		gpu          = requireMetalDevice(t)
		native       = gpu.state.backend.(metalBackend).runtime
		initial      = native.ResourceSnapshot()
		weightValues = ffnWeightValues()
	)
	for _, dtype := range []DType{Float16, BFloat16} {
		t.Run(dtype.String(), func(t *testing.T) {
			var (
				weights        = newFFNWeightsWithDType(t, gpu, dtype, ffnHidden, ffnIntermediate)
				dequantWeights [3][]float32
				totalWeightB   uint64
			)
			for i, w := range weights {
				requireError(t, w.UploadFloat32As(weightValues[i]), nil)
				totalWeightB += w.Bytes()
				dequantWeights[i] = make([]float32, len(weightValues[i]))
				for j, v := range weightValues[i] {
					dequantWeights[i][j] = roundTripDType(v, dtype)
				}
			}
			var wantWeightBytes = uint64(3 * ffnHidden * ffnIntermediate * 2)
			if totalWeightB != wantWeightBytes {
				t.Fatalf("FFN %s weight bytes = %d, want %d", dtype, totalWeightB, wantWeightBytes)
			}

			for _, rows := range ffnRowCounts {
				t.Run(fmt.Sprintf("M%d", rows), func(t *testing.T) {
					var (
						input   = ffnInputs(rows)
						want    = ffnReference(input, dequantWeights, ffnHidden, ffnIntermediate)
						scratch = newFFNScratch(t, gpu, rows, ffnHidden, ffnIntermediate)
						output  = make([]float32, len(input))
						scope   = requireScope(t, gpu)
						ctx     = context.Background()
					)
					requireError(t, scratch.input.Upload(input), nil)
					requireError(t, scratch.encode(scope, weights), nil)
					requireError(t, scope.Submit(ctx), nil)
					requireError(t, scope.Wait(ctx), nil)
					requireError(t, scope.Close(), nil)
					requireError(t, scratch.output.DownloadInto(output), nil)
					checkFFN(t, output, want)
					logFFNError(t, output, want)
				})
			}
		})
	}
	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("mixed FFN leaked native resources: before %+v; after %+v", initial, after)
	}
}

func runMetalMatMulMixed(
	t *testing.T,
	gpu *Device,
	dtype DType,
	rows, inner, cols int64,
	values, weight []float32,
	transposed bool,
) (got []float32) {
	t.Helper()
	var (
		left        = requireTensor(t, gpu, rows, inner)
		destination = requireTensor(t, gpu, rows, cols)
		scope       = requireScope(t, gpu)
		ctx         = context.Background()
	)
	requireError(t, left.Upload(values), nil)
	if transposed {
		var right = requireTensorWithDType(t, gpu, dtype, cols, inner)
		if right.Bytes() != uint64(cols*inner)*dtype.Bytes() {
			t.Fatalf("right bytes = %d, want %d", right.Bytes(), uint64(cols*inner)*dtype.Bytes())
		}
		requireError(t, right.UploadFloat32As(weight), nil)
		requireError(t, scope.MatMulRightTranspose(left, right, destination), nil)
	} else {
		var (
			right    = requireTensorWithDType(t, gpu, dtype, inner, cols)
			physical = make([]float32, len(weight))
		)
		if right.Bytes() != uint64(cols*inner)*dtype.Bytes() {
			t.Fatalf("right bytes = %d, want %d", right.Bytes(), uint64(cols*inner)*dtype.Bytes())
		}
		for col := range cols {
			for k := range inner {
				physical[k*cols+col] = weight[col*inner+k]
			}
		}
		requireError(t, right.UploadFloat32As(physical), nil)
		requireError(t, scope.MatMul(left, right, destination), nil)
	}
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)
	got = make([]float32, rows*cols)
	requireError(t, destination.DownloadInto(got), nil)
	return got
}

func roundTripDType(value float32, dtype DType) (out float32) {
	switch dtype {
	case Float16:
		return Float16ToFloat32(Float32ToFloat16(value))
	case BFloat16:
		return BFloat16ToFloat32(Float32ToBFloat16(value))
	default:
		return value
	}
}

func sameSpecialFloat32(got, want float32) (ok bool) {
	return got == want || (math.IsNaN(float64(got)) && math.IsNaN(float64(want)))
}

func newFFNWeightsWithDType(t testing.TB, device *Device, dtype DType, hidden, intermediate int64) (weights [3]*Tensor) {
	t.Helper()
	return [3]*Tensor{
		requireTensorWithDType(t, device, dtype, intermediate, hidden),
		requireTensorWithDType(t, device, dtype, intermediate, hidden),
		requireTensorWithDType(t, device, dtype, hidden, intermediate),
	}
}
