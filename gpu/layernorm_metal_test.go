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

func referenceLayerNormFP64(input, weight, bias []float32, rows, width int, epsilon float32) (out []float64) {
	out = make([]float64, rows*width)
	var eps = float64(epsilon)
	for r := range rows {
		var (
			base = r * width
			sum  float64
		)
		for c := range width {
			sum += float64(input[base+c])
		}
		var (
			mean   = sum / float64(width)
			varSum float64
		)
		for c := range width {
			var d = float64(input[base+c]) - mean
			varSum += d * d
		}
		var invStd = 1.0 / math.Sqrt(varSum/float64(width)+eps)
		for c := range width {
			var v = (float64(input[base+c]) - mean) * invStd * float64(weight[c])
			if bias != nil {
				v += float64(bias[c])
			}
			out[base+c] = v
		}
	}
	return out
}

func layerNormFP64ErrorStats(got []float32, want []float64) (p50, p95, p99, maxErr float64) {
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

func Test_MetalLayerNormAgainstFP64(t *testing.T) {
	var (
		dev     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = dev.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		rowsSet = []int64{1, 6, 27, 28, 29, 32}
		widths  = []int64{1, 3, 17, 32, 33, 64, 65, 128, 257, 768, 1024, 2048, 4096}
		dtypes  = []DType{Float32, Float16, BFloat16}
		eps     = float32(1e-5)
	)

	t.Run("2D_sweep", func(t *testing.T) {
		for _, w := range widths {
			for _, m := range rowsSet {
				if w >= 2048 && (m == 6 || m == 27 || m == 28) {
					continue
				}
				for _, wt := range dtypes {
					for _, withBias := range []bool{false, true} {
						var bt = wt
						if w == 257 && withBias && wt == Float16 {
							bt = Float32
						}
						if w == 768 && withBias && wt == BFloat16 {
							bt = Float16
						}
						var name = fmt.Sprintf("M%d_W%d_w%s_b%v", m, w, wt, withBias)
						t.Run(name, func(t *testing.T) {
							var (
								input  = requireTensor(t, dev, m, w)
								output = requireTensor(t, dev, m, w)
								weight = requireTensorWithDType(t, dev, wt, w)
								bias   *Tensor
								scope  = requireScope(t, dev)
								inData = attentionValues(m*w, uint64(w*31+m), 1)
								wData  = attentionValues(w, uint64(w*37+m), 1)
								bData  []float32
								effW   = dequantizeForRef(wData, wt)
								effB   []float32
							)
							requireError(t, input.Upload(inData), nil)
							requireError(t, weight.UploadFloat32As(wData), nil)
							if withBias {
								bias = requireTensorWithDType(t, dev, bt, w)
								bData = attentionValues(w, uint64(w*43+m), 1)
								requireError(t, bias.UploadFloat32As(bData), nil)
								effB = dequantizeForRef(bData, bt)
							}
							requireError(t, scope.LayerNorm(input, weight, bias, output, eps), nil)
							requireError(t, scope.Submit(ctx), nil)
							requireError(t, scope.Wait(ctx), nil)
							requireError(t, scope.Close(), nil)

							var (
								got                    = make([]float32, m*w)
								want                   = referenceLayerNormFP64(inData, effW, effB, int(m), int(w), eps)
								p50, p95, p99, maxDiff float64
							)
							requireError(t, output.DownloadInto(got), nil)
							p50, p95, p99, maxDiff = layerNormFP64ErrorStats(got, want)
							if maxDiff > 1e-5 {
								t.Fatalf("%s max abs error = %g (p50=%g p95=%g p99=%g)", name, maxDiff, p50, p95, p99)
							}
							if (w == 257 && m == 29) ||
								(w == 768 && (m == 1 || m == 32)) ||
								(w == 2048 && (m == 1 || m == 32)) ||
								(w == 4096 && m == 32) {
								t.Logf("LayerNorm %s (bt=%s) FP64 abs err: p50=%g p95=%g p99=%g max=%g",
									name, bt, p50, p95, p99, maxDiff)
							}
						})
					}
				}
			}
		}
	})

	t.Run("ranks_1D_to_4D", func(t *testing.T) {
		var shapes = [][]int64{
			{257},
			{3, 64},
			{2, 5, 65},
			{2, 3, 4, 32},
		}
		for _, shape := range shapes {
			t.Run(fmt.Sprint(shape), func(t *testing.T) {
				var (
					input  = requireTensor(t, dev, shape...)
					output = requireTensor(t, dev, shape...)
					width  = shape[len(shape)-1]
					rows   = int(input.state.storage.count / uint64(width))
					weight = requireTensorWithDType(t, dev, Float16, width)
					bias   = requireTensorWithDType(t, dev, BFloat16, width)
					inData = attentionValues(int64(rows)*width, uint64(width*19), 1)
					wData  = attentionValues(width, uint64(width*23), 1)
					bData  = attentionValues(width, uint64(width*29), 1)
					scope  = requireScope(t, dev)
				)
				requireError(t, input.Upload(inData), nil)
				requireError(t, weight.UploadFloat32As(wData), nil)
				requireError(t, bias.UploadFloat32As(bData), nil)
				requireError(t, scope.LayerNorm(input, weight, bias, output, eps), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)

				var (
					got                    = make([]float32, len(inData))
					want                   = referenceLayerNormFP64(inData, dequantizeForRef(wData, Float16), dequantizeForRef(bData, BFloat16), rows, int(width), eps)
					p50, p95, p99, maxDiff float64
				)
				requireError(t, output.DownloadInto(got), nil)
				p50, p95, p99, maxDiff = layerNormFP64ErrorStats(got, want)
				if maxDiff > 1e-5 {
					t.Fatalf("shape %v max abs err = %g (p50=%g p95=%g p99=%g)", shape, maxDiff, p50, p95, p99)
				}
			})
		}
	})

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("resource leak: before=%+v after=%+v", initial, after)
	}
}

func Test_MetalLayerNormEdgeCases(t *testing.T) {
	var (
		dev = requireMetalDevice(t)
		ctx = context.Background()
		eps = float32(1e-5)
	)

	t.Run("constant_rows_and_specials", func(t *testing.T) {
		var (
			rows   int64 = 7
			width  int64 = 16
			input        = requireTensor(t, dev, rows, width)
			output       = requireTensor(t, dev, rows, width)
			weight       = requireTensor(t, dev, width)
			bias         = requireTensor(t, dev, width)
			data         = make([]float32, rows*width)
			wData        = make([]float32, width)
			bData        = make([]float32, width)
			scope        = requireScope(t, dev)
		)
		for i := range width {
			wData[i] = float32(i+1) * 0.25
			bData[i] = float32(i) - 3.5
		}
		// Row 0: all zeros (var == 0)
		// Row 1: constant 5.0 (var == 0)
		for i := range width {
			data[width+i] = 5.0
		}
		// Row 2: very small values (1e-18)
		for i := range width {
			data[2*width+i] = float32(int(i)-8) * 1e-18
		}
		// Row 3: large finite values (1e15)
		for i := range width {
			data[3*width+i] = float32(int(i)-8) * 1e15
		}
		// Row 4: contains NaN
		for i := range width {
			data[4*width+i] = float32(i)
		}
		data[4*width+3] = float32(math.NaN())
		// Row 5: contains +Inf
		for i := range width {
			data[5*width+i] = float32(i)
		}
		data[5*width+1] = float32(math.Inf(1))
		// Row 6: contains -Inf
		for i := range width {
			data[6*width+i] = float32(i)
		}
		data[6*width+2] = float32(math.Inf(-1))

		requireError(t, input.Upload(data), nil)
		requireError(t, weight.Upload(wData), nil)
		requireError(t, bias.Upload(bData), nil)
		requireError(t, scope.LayerNorm(input, weight, bias, output, eps), nil)
		requireError(t, scope.Submit(ctx), nil)
		requireError(t, scope.Wait(ctx), nil)
		requireError(t, scope.Close(), nil)

		var got = make([]float32, rows*width)
		requireError(t, output.DownloadInto(got), nil)

		// Rows 0..3: finite match against FP64 reference.
		var want = referenceLayerNormFP64(data[:4*width], wData, bData, 4, int(width), eps)
		var _, _, _, maxDiff = layerNormFP64ErrorStats(got[:4*width], want)
		if maxDiff > 1e-5 {
			t.Fatalf("finite edge rows max abs err = %g", maxDiff)
		}
		// Rows 4..6: NaN / ±Inf produce NaN across the entire row.
		for row := int64(4); row < rows; row++ {
			for col := range width {
				var v = got[row*width+col]
				if !math.IsNaN(float64(v)) {
					t.Fatalf("special row %d col %d = %v, want NaN", row, col, v)
				}
			}
		}
	})

	t.Run("large_offset_catastrophic_cancellation", func(t *testing.T) {
		for _, width := range []int64{256, 257} {
			t.Run(fmt.Sprintf("W%d", width), func(t *testing.T) {
				var (
					rows   int64 = 4
					input        = requireTensor(t, dev, rows, width)
					output       = requireTensor(t, dev, rows, width)
					weight       = requireTensor(t, dev, width)
					bias         = requireTensor(t, dev, width)
					rng          = rand.New(rand.NewPCG(0x4e4e16, uint64(width)))
					data         = make([]float32, rows*width)
					wData        = make([]float32, width)
					bData        = make([]float32, width)
					scope        = requireScope(t, dev)
				)
				for i := range data {
					// 1e5 + small_delta would catastrophically cancel under single-pass E[x^2] - E[x]^2.
					data[i] = 1e5 + (rng.Float32()*2.0 - 1.0)
				}
				for i := range width {
					wData[i] = 0.5 + rng.Float32()
					bData[i] = rng.Float32() - 0.5
				}
				requireError(t, input.Upload(data), nil)
				requireError(t, weight.Upload(wData), nil)
				requireError(t, bias.Upload(bData), nil)
				requireError(t, scope.LayerNorm(input, weight, bias, output, eps), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)

				var (
					got                    = make([]float32, rows*width)
					want                   = referenceLayerNormFP64(data, wData, bData, int(rows), int(width), eps)
					p50, p95, p99, maxDiff float64
				)
				requireError(t, output.DownloadInto(got), nil)
				p50, p95, p99, maxDiff = layerNormFP64ErrorStats(got, want)
				t.Logf("LayerNorm cancellation 1e5+delta W=%d FP64 abs err: p50=%g p95=%g p99=%g max=%g",
					width, p50, p95, p99, maxDiff)
				if maxDiff > 1e-4 {
					t.Fatalf("cancellation W=%d max abs err = %g (p50=%g p95=%g p99=%g)", width, maxDiff, p50, p95, p99)
				}
			})
		}
	})

	t.Run("view_execution_and_alias_rejection", func(t *testing.T) {
		var (
			rawIn   = requireTensor(t, dev, 2, 3, 16)
			rawDst  = requireTensor(t, dev, 2, 3, 16)
			weight  = requireTensor(t, dev, 16)
			bias    = requireTensor(t, dev, 16)
			inData  = attentionValues(96, 1601, 1)
			wData   = attentionValues(16, 1602, 1)
			bData   = attentionValues(16, 1603, 1)
			inView  *Tensor
			dstView *Tensor
			err     error
		)
		requireError(t, rawIn.Upload(inData), nil)
		requireError(t, weight.Upload(wData), nil)
		requireError(t, bias.Upload(bData), nil)
		inView, err = rawIn.View([]int64{6, 16})
		requireError(t, err, nil)
		defer inView.Close()
		dstView, err = rawDst.View([]int64{6, 16})
		requireError(t, err, nil)
		defer dstView.Close()

		// Alias through view must be rejected.
		var badScope = requireScope(t, dev)
		var aliasView *Tensor
		aliasView, err = rawIn.View([]int64{6, 16})
		requireError(t, err, nil)
		defer aliasView.Close()
		requireError(t, badScope.LayerNorm(inView, weight, bias, aliasView, eps), ErrInvalid)
		requireError(t, badScope.Close(), nil)

		// Disjoint views succeed and match FP64 reference.
		var scope = requireScope(t, dev)
		requireError(t, scope.LayerNorm(inView, weight, bias, dstView, eps), nil)
		requireError(t, scope.Submit(ctx), nil)
		requireError(t, scope.Wait(ctx), nil)
		requireError(t, scope.Close(), nil)

		var (
			got  = make([]float32, 96)
			want = referenceLayerNormFP64(inData, wData, bData, 6, 16, eps)
		)
		requireError(t, rawDst.DownloadInto(got), nil)
		var _, _, _, maxDiff = layerNormFP64ErrorStats(got, want)
		if maxDiff > 1e-5 {
			t.Fatalf("view LayerNorm max abs err = %g", maxDiff)
		}
	})
}

func Test_MetalLayerNormConcurrent(t *testing.T) {
	const (
		workers    = 8
		iterations = 16
		rows       = 29
	)
	var (
		dev    = requireMetalDevice(t)
		ctx    = context.Background()
		eps    = float32(1e-5)
		widths = []int64{33, 64, 257, 768, 2048}
	)
	for _, width := range widths {
		t.Run(fmt.Sprintf("W%d", width), func(t *testing.T) {
			var (
				weight = requireTensorWithDType(t, dev, Float16, width)
				bias   = requireTensorWithDType(t, dev, BFloat16, width)
				wData  = attentionValues(width, uint64(width*71), 1)
				bData  = attentionValues(width, uint64(width*73), 1)
				effW   = dequantizeForRef(wData, Float16)
				effB   = dequantizeForRef(bData, BFloat16)
				wg     sync.WaitGroup
				errCh  = make(chan error, workers)
			)
			requireError(t, weight.UploadFloat32As(wData), nil)
			requireError(t, bias.UploadFloat32As(bData), nil)

			for worker := range workers {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()
					var (
						inData = attentionValues(rows*width, uint64(id*1009+int(width)), 1)
						want   = referenceLayerNormFP64(inData, effW, effB, rows, int(width), eps)
						got    = make([]float32, rows*width)
						input  *Tensor
						output *Tensor
						err    error
					)
					if input, err = dev.NewFloat32([]int64{rows, width}); err != nil {
						errCh <- err
						return
					}
					defer input.Close()
					if output, err = dev.NewFloat32([]int64{rows, width}); err != nil {
						errCh <- err
						return
					}
					defer output.Close()
					if err = input.Upload(inData); err != nil {
						errCh <- err
						return
					}
					for iter := range iterations {
						var scope *Scope
						if scope, err = dev.NewScope(); err != nil {
							errCh <- err
							return
						}
						if err = scope.LayerNorm(input, weight, bias, output, eps); err != nil {
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
						if err = output.DownloadInto(got); err != nil {
							errCh <- err
							return
						}
						var _, _, _, maxDiff = layerNormFP64ErrorStats(got, want)
						if maxDiff > 1e-5 {
							errCh <- fmt.Errorf("worker %d iter %d W=%d max abs err = %g", id, iter, width, maxDiff)
							return
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
	}
}
