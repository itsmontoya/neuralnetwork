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

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func batchNormReferenceFP64(input, scale, bias, mean, variance []float32, shape []int64, epsilon float32) (out []float64) {
	var (
		channels = int(shape[1])
		spatial  = 1
	)
	for _, dim := range shape[2:] {
		spatial *= int(dim)
	}
	out = make([]float64, len(input))
	for i, x := range input {
		var (
			c     = (i / spatial) % channels
			denom = math.Sqrt(float64(variance[c]) + float64(epsilon))
		)
		out[i] = float64(scale[c])*(float64(x)-float64(mean[c]))/denom + float64(bias[c])
	}
	return out
}

func checkBatchNormAgainstFP64(t *testing.T, label string, got []float32, want []float64, tolerance float64) {
	t.Helper()
	var (
		deltas = make([]float64, 0, len(got))
		maxRel float64
	)
	for i, value := range got {
		var (
			actual   = float64(value)
			expected = want[i]
		)
		if math.IsNaN(expected) || math.IsInf(expected, 0) {
			if actual == expected || (math.IsNaN(expected) && math.IsNaN(actual)) {
				continue
			}
			t.Fatalf("%s element %d: got %g, FP64 reference %g", label, i, actual, expected)
		}
		var (
			delta = math.Abs(actual - expected)
			limit = tolerance + tolerance*math.Abs(expected)
		)
		if math.IsNaN(delta) || delta > limit {
			t.Fatalf("%s element %d: got %.9g, FP64 reference %.9g, abs error %.3g exceeds %.3g",
				label, i, actual, expected, delta, limit)
		}
		deltas = append(deltas, delta)
		if math.Abs(expected) >= 1e-6 {
			maxRel = max(maxRel, delta/math.Abs(expected))
		}
	}
	if len(deltas) == 0 {
		return
	}
	slices.Sort(deltas)
	var last = len(deltas) - 1
	t.Logf("%s FP64 abs error (%d finite): p50=%.3g p95=%.3g p99=%.3g max=%.3g maxRel=%.3g",
		label, len(deltas), deltas[last*50/100], deltas[last*95/100], deltas[last*99/100], deltas[last], maxRel)
}

func Test_MetalBatchNormAgainstFP64AndTensor(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		shapes  = [][]int64{
			{1, 1},
			{4, 7},
			{32, 64},
			{2, 5, 17},
			{4, 16, 64},
			{1, 3, 5, 7},
			{2, 8, 4, 4},
			{2, 32, 7, 7},
			{4, 16, 14, 14},
			{1, 256, 14, 14},
			{2, 3, 2, 2, 2},
		}
	)

	for _, shape := range shapes {
		for _, eps := range []float32{0, 1e-5} {
			t.Run(fmt.Sprintf("%v/eps=%g", shape, eps), func(t *testing.T) {
				var (
					channels = int(shape[1])
					input    = requireTensor(t, gpu, shape...)
					dst      = requireTensor(t, gpu, shape...)
					scale    = requireTensor(t, gpu, int64(channels))
					bias     = requireTensor(t, gpu, int64(channels))
					mean     = requireTensor(t, gpu, int64(channels))
					variance = requireTensor(t, gpu, int64(channels))
					count    = int(input.state.storage.count)
					random   = rand.New(rand.NewPCG(0x4e4e26b, uint64(count)+uint64(channels)))
					xData    = make([]float32, count)
					sData    = make([]float32, channels)
					bData    = make([]float32, channels)
					mData    = make([]float32, channels)
					vData    = make([]float32, channels)
					got      = make([]float32, count)
				)
				for i := range xData {
					xData[i] = 10*random.Float32() - 5
				}
				for c := range channels {
					sData[c] = 2*random.Float32() - 1
					bData[c] = 2*random.Float32() - 1
					mData[c] = 4*random.Float32() - 2
					vData[c] = 0.05 + 4*random.Float32()
				}
				requireError(t, input.Upload(xData), nil)
				requireError(t, scale.Upload(sData), nil)
				requireError(t, bias.Upload(bData), nil)
				requireError(t, mean.Upload(mData), nil)
				requireError(t, variance.Upload(vData), nil)

				var scope = requireScope(t, gpu)
				requireError(t, scope.BatchNorm(input, scale, bias, mean, variance, dst, eps), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)
				requireError(t, dst.DownloadInto(got), nil)

				var wantFP64 = batchNormReferenceFP64(xData, sData, bData, mData, vData, shape, eps)
				checkBatchNormAgainstFP64(t, fmt.Sprintf("BatchNorm%v", shape), got, wantFP64, 1e-5)

				// Cross-check against CPU tensor.BatchNormalizationInto.
				var (
					cpuIn, err  = tensor.FromFloat32(shape, xData)
					cpuSc, err2 = tensor.FromFloat32([]int64{int64(channels)}, sData)
					cpuBi, err3 = tensor.FromFloat32([]int64{int64(channels)}, bData)
					cpuMn, err4 = tensor.FromFloat32([]int64{int64(channels)}, mData)
					cpuVr, err5 = tensor.FromFloat32([]int64{int64(channels)}, vData)
					cpuOut, e6  = tensor.FromFloat32(shape, make([]float32, count))
				)
				for _, e := range []error{err, err2, err3, err4, err5, e6} {
					requireError(t, e, nil)
				}
				requireError(t, tensor.BatchNormalizationInto(cpuIn, cpuSc, cpuBi, cpuMn, cpuVr, cpuOut, eps), nil)
				var cpuVals, errVals = cpuOut.Float32Values()
				requireError(t, errVals, nil)
				for i := range count {
					var diff = math.Abs(float64(got[i]) - float64(cpuVals[i]))
					if diff > 1e-5*(1+math.Abs(float64(cpuVals[i]))) {
						t.Fatalf("BatchNorm vs tensor.BatchNormalizationInto mismatch at %d: gpu=%.9g cpu=%.9g (diff=%.3g)",
							i, got[i], cpuVals[i], diff)
					}
				}
			})
		}
	}

	t.Run("special_values", func(t *testing.T) {
		var (
			inf = float32(math.Inf(1))
			nan = float32(math.NaN())
		)
		// Test both scalar (spatial=5) and vec4 (spatial=8) paths with special values.
		for _, spatial := range []int64{5, 8} {
			t.Run(fmt.Sprintf("spatial=%d", spatial), func(t *testing.T) {
				var (
					shape    = []int64{1, 4, spatial}
					count    = int(4 * spatial)
					input    = requireTensor(t, gpu, shape...)
					dst      = requireTensor(t, gpu, shape...)
					scale    = requireTensor(t, gpu, 4)
					bias     = requireTensor(t, gpu, 4)
					mean     = requireTensor(t, gpu, 4)
					variance = requireTensor(t, gpu, 4)
					xData    = make([]float32, count)
					sData    = []float32{1, 2, 0.5, 1}
					bData    = []float32{0, -1, 0.25, 0}
					mData    = []float32{0, 1, -2, 0}
					vData    = []float32{1, 0, 4, 1}
					got      = make([]float32, count)
				)
				// Channel 0: normal + special values (0, 1000, -1000, +Inf, -Inf/NaN)
				// Channel 1: zero variance with eps=0 -> (x-1)/0 -> ±Inf or NaN
				// Channel 2: normal values
				// Channel 3: NaN propagation
				for c := range 4 {
					for s := range int(spatial) {
						var idx = c*int(spatial) + s
						switch c {
						case 0:
							switch s {
							case 0:
								xData[idx] = 0
							case 1:
								xData[idx] = 1000
							case 2:
								xData[idx] = -1000
							case 3:
								xData[idx] = inf
							case 4:
								xData[idx] = -inf
							default:
								xData[idx] = float32(s)
							}
						case 1:
							if s == 0 {
								xData[idx] = 1 // 0 / 0 -> NaN
							} else if s%2 == 1 {
								xData[idx] = 2 // +Inf
							} else {
								xData[idx] = 0 // -Inf
							}
						case 2:
							xData[idx] = float32(s) - 2.5
						case 3:
							xData[idx] = nan
						}
					}
				}
				requireError(t, input.Upload(xData), nil)
				requireError(t, scale.Upload(sData), nil)
				requireError(t, bias.Upload(bData), nil)
				requireError(t, mean.Upload(mData), nil)
				requireError(t, variance.Upload(vData), nil)

				var scope = requireScope(t, gpu)
				requireError(t, scope.BatchNorm(input, scale, bias, mean, variance, dst, 0), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)
				requireError(t, dst.DownloadInto(got), nil)

				var wantFP64 = batchNormReferenceFP64(xData, sData, bData, mData, vData, shape, 0)
				checkBatchNormAgainstFP64(t, fmt.Sprintf("BatchNormSpecial(spatial=%d)", spatial), got, wantFP64, 1e-5)
			})
		}
	})

	t.Run("view_alias_rejected", func(t *testing.T) {
		var (
			input    = requireTensor(t, gpu, 1, 4, 2, 2)
			scale    = requireTensor(t, gpu, 4)
			bias     = requireTensor(t, gpu, 4)
			mean     = requireTensor(t, gpu, 4)
			variance = requireTensor(t, gpu, 4)
			alias, e = input.View([]int64{1, 4, 2, 2})
		)
		requireError(t, e, nil)
		defer alias.Close()
		requireError(t, input.Upload(make([]float32, 16)), nil)
		requireError(t, scale.Upload(make([]float32, 4)), nil)
		requireError(t, bias.Upload(make([]float32, 4)), nil)
		requireError(t, mean.Upload(make([]float32, 4)), nil)
		requireError(t, variance.Upload(make([]float32, 4)), nil)
		var scope = requireScope(t, gpu)
		requireError(t, scope.BatchNorm(input, scale, bias, mean, variance, alias, 1e-5), ErrInvalid)
		requireError(t, scope.Close(), nil)
	})

	t.Run("concurrent_scopes", func(t *testing.T) {
		const (
			workers    = 8
			iterations = 12
		)
		var (
			shape    = []int64{2, 16, 8, 8}
			count    = 2 * 16 * 8 * 8
			input    = requireTensor(t, gpu, shape...)
			scale    = requireTensor(t, gpu, 16)
			bias     = requireTensor(t, gpu, 16)
			mean     = requireTensor(t, gpu, 16)
			variance = requireTensor(t, gpu, 16)
			random   = rand.New(rand.NewPCG(0x4e4e26d, 7))
			xData    = make([]float32, count)
			sData    = make([]float32, 16)
			bData    = make([]float32, 16)
			mData    = make([]float32, 16)
			vData    = make([]float32, 16)
			wg       sync.WaitGroup
			errCh    = make(chan error, workers)
		)
		for i := range xData {
			xData[i] = 6*random.Float32() - 3
		}
		for c := range 16 {
			sData[c] = 1.5*random.Float32() + 0.1
			bData[c] = random.Float32() - 0.5
			mData[c] = 2*random.Float32() - 1
			vData[c] = 0.25 + 2*random.Float32()
		}
		requireError(t, input.Upload(xData), nil)
		requireError(t, scale.Upload(sData), nil)
		requireError(t, bias.Upload(bData), nil)
		requireError(t, mean.Upload(mData), nil)
		requireError(t, variance.Upload(vData), nil)
		var want = batchNormReferenceFP64(xData, sData, bData, mData, vData, shape, 1e-5)

		for worker := range workers {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				var (
					dst = requireTensor(t, gpu, shape...)
					got = make([]float32, count)
				)
				for range iterations {
					var scope, err = gpu.NewScope()
					if err != nil {
						errCh <- fmt.Errorf("worker %d NewScope: %w", id, err)
						return
					}
					if err = scope.BatchNorm(input, scale, bias, mean, variance, dst, 1e-5); err != nil {
						_ = scope.Close()
						errCh <- fmt.Errorf("worker %d BatchNorm: %w", id, err)
						return
					}
					if err = scope.Submit(ctx); err != nil {
						_ = scope.Close()
						errCh <- fmt.Errorf("worker %d Submit: %w", id, err)
						return
					}
					if err = scope.Wait(ctx); err != nil {
						_ = scope.Close()
						errCh <- fmt.Errorf("worker %d Wait: %w", id, err)
						return
					}
					if err = scope.Close(); err != nil {
						errCh <- fmt.Errorf("worker %d Close: %w", id, err)
						return
					}
					if err = dst.DownloadInto(got); err != nil {
						errCh <- fmt.Errorf("worker %d Download: %w", id, err)
						return
					}
					for i := range count {
						if math.Abs(float64(got[i])-want[i]) > 1e-5*(1+math.Abs(want[i])) {
							errCh <- fmt.Errorf("worker %d BatchNorm[%d]=%g want %g", id, i, got[i], want[i])
							return
						}
					}
				}
			}(worker)
		}
		wg.Wait()
		close(errCh)
		for err := range errCh {
			t.Fatal(err)
		}
	})

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("BatchNorm tests leaked native resources: before %+v; after %+v", initial, after)
	}
}
