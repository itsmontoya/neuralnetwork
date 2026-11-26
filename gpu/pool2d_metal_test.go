//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"math"
	"sort"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func refMaxPool2DFP64(input []float32, shape []int64, cfg Pool2DConfig) (out []float64) {
	var (
		outShape, _ = Pool2DShape(shape, cfg)
		n           = int(shape[0])
		c           = int(shape[1])
		inH         = int(shape[2])
		inW         = int(shape[3])
		outH        = int(outShape[2])
		outW        = int(outShape[3])
	)
	out = make([]float64, n*c*outH*outW)
	var idx int
	for b := range n {
		for ch := range c {
			var base = (b*c + ch) * inH * inW
			for oh := range outH {
				var startH = int64(oh)*cfg.Strides[0] - cfg.Pads[0]
				for ow := range outW {
					var (
						startW = int64(ow)*cfg.Strides[1] - cfg.Pads[1]
						best   = math.Inf(-1)
					)
					for kh := range cfg.Kernel[0] {
						var ih = startH + kh*cfg.Dilations[0]
						if ih < 0 || ih >= int64(inH) {
							continue
						}
						for kw := range cfg.Kernel[1] {
							var iw = startW + kw*cfg.Dilations[1]
							if iw < 0 || iw >= int64(inW) {
								continue
							}
							var sample = float64(input[base+int(ih)*inW+int(iw)])
							if sample > best || math.IsNaN(sample) {
								best = sample
							}
						}
					}
					out[idx] = best
					idx++
				}
			}
		}
	}
	return out
}

func refAvgPool2DFP64(input []float32, shape []int64, cfg Pool2DConfig) (out []float64) {
	var (
		outShape, _ = Pool2DShape(shape, cfg)
		n           = int(shape[0])
		c           = int(shape[1])
		inH         = int(shape[2])
		inW         = int(shape[3])
		outH        = int(outShape[2])
		outW        = int(outShape[3])
	)
	out = make([]float64, n*c*outH*outW)
	var idx int
	for b := range n {
		for ch := range c {
			var base = (b*c + ch) * inH * inW
			for oh := range outH {
				var startH = int64(oh)*cfg.Strides[0] - cfg.Pads[0]
				for ow := range outW {
					var (
						startW = int64(ow)*cfg.Strides[1] - cfg.Pads[1]
						sum    float64
						count  int64
					)
					for kh := range cfg.Kernel[0] {
						var ih = startH + kh*cfg.Dilations[0]
						if ih < 0 || ih >= int64(inH) {
							continue
						}
						for kw := range cfg.Kernel[1] {
							var iw = startW + kw*cfg.Dilations[1]
							if iw < 0 || iw >= int64(inW) {
								continue
							}
							sum += float64(input[base+int(ih)*inW+int(iw)])
							count++
						}
					}
					if cfg.CountIncludePad {
						count = cfg.Kernel[0] * cfg.Kernel[1]
					}
					out[idx] = sum / float64(count)
					idx++
				}
			}
		}
	}
	return out
}

func Test_MetalScopePool2DParity(t *testing.T) {
	var (
		dev     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = dev.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
	)

	type testcase struct {
		name  string
		shape []int64
		cfg   Pool2DConfig
		seed  uint64
	}
	var cases = []testcase{
		{
			name:  "2x2_stride2_even",
			shape: []int64{2, 4, 8, 8},
			cfg:   Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1}},
			seed:  271,
		},
		{
			name:  "2x2_stride2_odd_floor",
			shape: []int64{2, 3, 7, 7},
			cfg:   Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1}},
			seed:  272,
		},
		{
			name:  "2x2_stride2_odd_ceil_excl_pad",
			shape: []int64{2, 3, 7, 7},
			cfg:   Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1}, CeilMode: true},
			seed:  273,
		},
		{
			name:  "2x2_stride2_odd_ceil_incl_pad",
			shape: []int64{2, 3, 7, 7},
			cfg:   Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1}, CeilMode: true, CountIncludePad: true},
			seed:  274,
		},
		{
			name:  "3x3_stride2_pad1_resnet",
			shape: []int64{2, 16, 28, 28},
			cfg:   Pool2DConfig{Kernel: [2]int64{3, 3}, Strides: [2]int64{2, 2}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}},
			seed:  275,
		},
		{
			name:  "asymmetric_pad_excl_pad",
			shape: []int64{2, 5, 9, 10},
			cfg:   Pool2DConfig{Kernel: [2]int64{3, 2}, Strides: [2]int64{2, 1}, Pads: [4]int64{1, 2, 0, 1}, Dilations: [2]int64{1, 1}},
			seed:  276,
		},
		{
			name:  "asymmetric_pad_incl_pad",
			shape: []int64{2, 5, 9, 10},
			cfg:   Pool2DConfig{Kernel: [2]int64{3, 2}, Strides: [2]int64{2, 1}, Pads: [4]int64{1, 2, 0, 1}, Dilations: [2]int64{1, 1}, CountIncludePad: true},
			seed:  277,
		},
		{
			name:  "dilated_2x3_excl_pad",
			shape: []int64{2, 4, 11, 13},
			cfg:   Pool2DConfig{Kernel: [2]int64{3, 3}, Strides: [2]int64{1, 2}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{2, 3}},
			seed:  278,
		},
		{
			name:  "dilated_2x2_ceil_incl_pad",
			shape: []int64{2, 3, 10, 9},
			cfg:   Pool2DConfig{Kernel: [2]int64{2, 3}, Strides: [2]int64{2, 2}, Pads: [4]int64{1, 0, 1, 1}, Dilations: [2]int64{2, 2}, CeilMode: true, CountIncludePad: true},
			seed:  279,
		},
		{
			name:  "global_pool_7x7",
			shape: []int64{4, 32, 7, 7},
			cfg:   Pool2DConfig{Kernel: [2]int64{7, 7}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}},
			seed:  280,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var (
				outShape, shapeErr = Pool2DShape(tc.shape, tc.cfg)
				inCount            = tc.shape[0] * tc.shape[1] * tc.shape[2] * tc.shape[3]
				outCount           = outShape[0] * outShape[1] * outShape[2] * outShape[3]
				inData             = attentionValues(inCount, tc.seed, 1)
				inTensor, e0       = dev.NewFloat32(tc.shape)
				maxOut, e1         = dev.NewFloat32(outShape)
				avgOut, e2         = dev.NewFloat32(outShape)
			)
			requireError(t, shapeErr, nil)
			requireError(t, e0, nil)
			requireError(t, e1, nil)
			requireError(t, e2, nil)
			defer func() {
				requireError(t, inTensor.Close(), nil)
				requireError(t, maxOut.Close(), nil)
				requireError(t, avgOut.Close(), nil)
			}()
			requireError(t, inTensor.Upload(inData), nil)

			var scope, err = dev.NewScope()
			requireError(t, err, nil)
			requireError(t, scope.MaxPool2D(inTensor, maxOut, tc.cfg), nil)
			requireError(t, scope.AvgPool2D(inTensor, avgOut, tc.cfg), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)

			var (
				gotMax = make([]float32, outCount)
				gotAvg = make([]float32, outCount)
			)
			requireError(t, maxOut.DownloadInto(gotMax), nil)
			requireError(t, avgOut.DownloadInto(gotAvg), nil)

			var (
				refMax64 = refMaxPool2DFP64(inData, tc.shape, tc.cfg)
				refAvg64 = refAvgPool2DFP64(inData, tc.shape, tc.cfg)
				avgErrs  = make([]float64, 0, outCount)
			)
			for i := range gotMax {
				var (
					aMax = float64(gotMax[i])
					eMax = refMax64[i]
				)
				if !(aMax == eMax || (math.IsNaN(aMax) && math.IsNaN(eMax))) {
					t.Fatalf("MaxPool2D mismatch at %d: got %g want %g", i, gotMax[i], eMax)
				}
				var (
					aAvg = float64(gotAvg[i])
					eAvg = refAvg64[i]
				)
				if math.IsNaN(eAvg) || math.IsInf(eAvg, 0) {
					if !(aAvg == eAvg || (math.IsNaN(aAvg) && math.IsNaN(eAvg))) {
						t.Fatalf("AvgPool2D special mismatch at %d: got %g want %g", i, gotAvg[i], eAvg)
					}
					continue
				}
				var dAvg = math.Abs(aAvg - eAvg)
				avgErrs = append(avgErrs, dAvg)
				if math.IsNaN(dAvg) || dAvg > 1e-5 {
					t.Fatalf("AvgPool2D mismatch at %d: got %g want %g (diff %g)", i, gotAvg[i], eAvg, dAvg)
				}
			}
			sort.Float64s(avgErrs)
			var n = len(avgErrs)
			t.Logf("%s: MaxPool2D maxErr=0 (exact), AvgPool2D FP64 abs err p50=%.3e p95=%.3e p99=%.3e max=%.3e (N=%d)",
				tc.name, avgErrs[n/2], avgErrs[(n*95)/100], avgErrs[(n*99)/100], avgErrs[n-1], n)

			// Compare bit-exact against tensor.MaxPool2DInto and (for unit dilation) tensor.AveragePool2DInto.
			var (
				cpuIn, c1     = tensor.FromFloat32(tc.shape, inData)
				cpuMaxOut, c2 = tensor.FromFloat32(outShape, make([]float32, outCount))
				cpuCfg        = tensor.Pool2DConfig{
					Kernel:          tc.cfg.Kernel,
					Strides:         tc.cfg.Strides,
					Dilations:       tc.cfg.Dilations,
					Pads:            tc.cfg.Pads,
					CeilMode:        tc.cfg.CeilMode,
					CountIncludePad: tc.cfg.CountIncludePad,
				}
			)
			requireError(t, c1, nil)
			requireError(t, c2, nil)
			requireError(t, tensor.MaxPool2DInto(cpuIn, cpuMaxOut, cpuCfg), nil)
			var cpuMaxVals, c3 = cpuMaxOut.Float32Values()
			requireError(t, c3, nil)
			for i := range gotMax {
				if math.Float32bits(gotMax[i]) != math.Float32bits(cpuMaxVals[i]) {
					t.Fatalf("MaxPool2D bit mismatch vs tensor at %d: got %g (%x) want %g (%x)",
						i, gotMax[i], math.Float32bits(gotMax[i]), cpuMaxVals[i], math.Float32bits(cpuMaxVals[i]))
				}
			}
			if tc.cfg.Dilations == [2]int64{1, 1} {
				var cpuAvgOut, c4 = tensor.FromFloat32(outShape, make([]float32, outCount))
				requireError(t, c4, nil)
				requireError(t, tensor.AveragePool2DInto(cpuIn, cpuAvgOut, cpuCfg), nil)
				var cpuAvgVals, c5 = cpuAvgOut.Float32Values()
				requireError(t, c5, nil)
				for i := range gotAvg {
					var sameNaN = math.IsNaN(float64(gotAvg[i])) && math.IsNaN(float64(cpuAvgVals[i]))
					if !sameNaN && math.Float32bits(gotAvg[i]) != math.Float32bits(cpuAvgVals[i]) {
						t.Fatalf("AvgPool2D bit mismatch vs tensor at %d: got %g (%x) want %g (%x)",
							i, gotAvg[i], math.Float32bits(gotAvg[i]), cpuAvgVals[i], math.Float32bits(cpuAvgVals[i]))
					}
				}
			}
		})
	}

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("native resource leak: buffers %d->%d scopes %d->%d",
			initial.LiveBuffers, after.LiveBuffers, initial.LiveScopes, after.LiveScopes)
	}
}

func Test_MetalScopePool2DSpecialValues(t *testing.T) {
	var (
		dev     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = dev.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
	)

	t.Run("all_negative_no_zero_clamp", func(t *testing.T) {
		var (
			in, e1  = dev.NewFloat32([]int64{1, 1, 2, 2})
			out, e2 = dev.NewFloat32([]int64{1, 1, 1, 1})
			cfg     = Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1}}
		)
		requireError(t, e1, nil)
		requireError(t, e2, nil)
		defer func() {
			requireError(t, in.Close(), nil)
			requireError(t, out.Close(), nil)
		}()
		requireError(t, in.Upload([]float32{-10.5, -3.25, -100.0, -7.5}), nil)
		var scope, err = dev.NewScope()
		requireError(t, err, nil)
		requireError(t, scope.MaxPool2D(in, out, cfg), nil)
		requireError(t, scope.Submit(ctx), nil)
		requireError(t, scope.Wait(ctx), nil)
		requireError(t, scope.Close(), nil)
		var got = make([]float32, 1)
		requireError(t, out.DownloadInto(got), nil)
		if got[0] != -3.25 {
			t.Fatalf("MaxPool2D negative inputs = %g, want -3.25", got[0])
		}
	})

	t.Run("nan_and_inf_propagation", func(t *testing.T) {
		var (
			nan    = float32(math.NaN())
			posInf = float32(math.Inf(1))
			negInf = float32(math.Inf(-1))
			// 4 independent 2x2 windows in [1, 4, 2, 2]:
			// ch0: NaN before +Inf -> Max=NaN, Avg=NaN
			// ch1: +Inf before NaN -> Max=NaN, Avg=NaN
			// ch2: -Inf with finite -> Max=-2.0, Avg=-Inf
			// ch3: +Inf and -Inf -> Max=+Inf, Avg=NaN
			data = []float32{
				nan, posInf, 1, 2,
				posInf, nan, 1, 2,
				negInf, -2, -5, -3,
				posInf, negInf, 1, 2,
			}
			in, e1     = dev.NewFloat32([]int64{1, 4, 2, 2})
			maxOut, e2 = dev.NewFloat32([]int64{1, 4, 1, 1})
			avgOut, e3 = dev.NewFloat32([]int64{1, 4, 1, 1})
			cfg        = Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1}}
		)
		requireError(t, e1, nil)
		requireError(t, e2, nil)
		requireError(t, e3, nil)
		defer func() {
			requireError(t, in.Close(), nil)
			requireError(t, maxOut.Close(), nil)
			requireError(t, avgOut.Close(), nil)
		}()
		requireError(t, in.Upload(data), nil)
		var scope, err = dev.NewScope()
		requireError(t, err, nil)
		requireError(t, scope.MaxPool2D(in, maxOut, cfg), nil)
		requireError(t, scope.AvgPool2D(in, avgOut, cfg), nil)
		requireError(t, scope.Submit(ctx), nil)
		requireError(t, scope.Wait(ctx), nil)
		requireError(t, scope.Close(), nil)
		var (
			gotMax = make([]float32, 4)
			gotAvg = make([]float32, 4)
		)
		requireError(t, maxOut.DownloadInto(gotMax), nil)
		requireError(t, avgOut.DownloadInto(gotAvg), nil)

		if !math.IsNaN(float64(gotMax[0])) || !math.IsNaN(float64(gotMax[1])) ||
			gotMax[2] != -2 || !math.IsInf(float64(gotMax[3]), 1) {
			t.Fatalf("unexpected MaxPool2D special values: %v", gotMax)
		}
		if !math.IsNaN(float64(gotAvg[0])) || !math.IsNaN(float64(gotAvg[1])) ||
			!math.IsInf(float64(gotAvg[2]), -1) || !math.IsNaN(float64(gotAvg[3])) {
			t.Fatalf("unexpected AvgPool2D special values: %v", gotAvg)
		}
	})

	t.Run("empty_window_in_padding", func(t *testing.T) {
		var (
			in, e1         = dev.NewFloat32([]int64{1, 1, 1, 1})
			maxOut, e2     = dev.NewFloat32([]int64{1, 1, 4, 4})
			avgExclOut, e3 = dev.NewFloat32([]int64{1, 1, 4, 4})
			avgInclOut, e4 = dev.NewFloat32([]int64{1, 1, 4, 4})
			cfgExcl        = Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{1, 1}, Pads: [4]int64{2, 2, 2, 2}, Dilations: [2]int64{1, 1}}
			cfgIncl        = Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{1, 1}, Pads: [4]int64{2, 2, 2, 2}, Dilations: [2]int64{1, 1}, CountIncludePad: true}
		)
		requireError(t, e1, nil)
		requireError(t, e2, nil)
		requireError(t, e3, nil)
		requireError(t, e4, nil)
		defer func() {
			requireError(t, in.Close(), nil)
			requireError(t, maxOut.Close(), nil)
			requireError(t, avgExclOut.Close(), nil)
			requireError(t, avgInclOut.Close(), nil)
		}()
		requireError(t, in.Upload([]float32{8.0}), nil)

		var scope, err = dev.NewScope()
		requireError(t, err, nil)
		requireError(t, scope.MaxPool2D(in, maxOut, cfgExcl), nil)
		requireError(t, scope.AvgPool2D(in, avgExclOut, cfgExcl), nil)
		requireError(t, scope.AvgPool2D(in, avgInclOut, cfgIncl), nil)
		requireError(t, scope.Submit(ctx), nil)
		requireError(t, scope.Wait(ctx), nil)
		requireError(t, scope.Close(), nil)

		var (
			gotMax     = make([]float32, 16)
			gotAvgExcl = make([]float32, 16)
			gotAvgIncl = make([]float32, 16)
		)
		requireError(t, maxOut.DownloadInto(gotMax), nil)
		requireError(t, avgExclOut.DownloadInto(gotAvgExcl), nil)
		requireError(t, avgInclOut.DownloadInto(gotAvgIncl), nil)

		if !math.IsInf(float64(gotMax[0]), -1) {
			t.Fatalf("empty MaxPool2D window = %g, want -Inf", gotMax[0])
		}
		if !math.IsNaN(float64(gotAvgExcl[0])) {
			t.Fatalf("empty AvgPool2D(CountIncludePad=false) window = %g, want NaN", gotAvgExcl[0])
		}
		if gotAvgIncl[0] != 0 {
			t.Fatalf("empty AvgPool2D(CountIncludePad=true) window = %g, want 0", gotAvgIncl[0])
		}
		if gotMax[5] != 8.0 || gotAvgExcl[5] != 8.0 || gotAvgIncl[5] != 2.0 {
			t.Fatalf("window (1,1): max=%g avgExcl=%g avgIncl=%g, want 8, 8, 2", gotMax[5], gotAvgExcl[5], gotAvgIncl[5])
		}
	})

	t.Run("view_alias_rejected_on_metal", func(t *testing.T) {
		var (
			in, e1 = dev.NewFloat32([]int64{1, 2, 4, 4})
			view   *Tensor
			err    error
			cfg    = Pool2DConfig{Kernel: [2]int64{1, 1}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}}
		)
		requireError(t, e1, nil)
		defer func() {
			if view != nil {
				requireError(t, view.Close(), nil)
			}
			requireError(t, in.Close(), nil)
		}()
		requireError(t, in.Upload(make([]float32, 32)), nil)
		view, err = in.View([]int64{1, 2, 4, 4})
		requireError(t, err, nil)
		var scope, scopeErr = dev.NewScope()
		requireError(t, scopeErr, nil)
		defer scope.Close()
		requireError(t, scope.MaxPool2D(in, view, cfg), ErrInvalid)
		requireError(t, scope.AvgPool2D(in, view, cfg), ErrInvalid)
	})

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("native resource leak: buffers %d->%d scopes %d->%d",
			initial.LiveBuffers, after.LiveBuffers, initial.LiveScopes, after.LiveScopes)
	}
}

func Test_MetalScopePool2DConcurrentStress(t *testing.T) {
	var (
		dev         = requireMetalDevice(t)
		ctx         = context.Background()
		native      = dev.state.backend.(metalBackend).runtime
		initial     = native.ResourceSnapshot()
		shape       = []int64{2, 8, 14, 14}
		cfg         = Pool2DConfig{Kernel: [2]int64{3, 3}, Strides: [2]int64{2, 2}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}}
		outShape, _ = Pool2DShape(shape, cfg)
		inCount     = shape[0] * shape[1] * shape[2] * shape[3]
		outCount    = outShape[0] * outShape[1] * outShape[2] * outShape[3]
		inData      = attentionValues(inCount, 285, 1)
		inTensor, e = dev.NewFloat32(shape)
		wantMax     = refMaxPool2DFP64(inData, shape, cfg)
		wantAvg     = refAvgPool2DFP64(inData, shape, cfg)
	)
	requireError(t, e, nil)
	requireError(t, inTensor.Upload(inData), nil)

	const (
		workers    = 8
		iterations = 16
	)
	var (
		wg   sync.WaitGroup
		errs = make(chan error, workers*iterations)
	)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var (
				maxOut, e1 = dev.NewFloat32(outShape)
				avgOut, e2 = dev.NewFloat32(outShape)
			)
			if e1 != nil {
				errs <- e1
				return
			}
			defer maxOut.Close()
			if e2 != nil {
				errs <- e2
				return
			}
			defer avgOut.Close()
			var (
				gotMax = make([]float32, outCount)
				gotAvg = make([]float32, outCount)
			)
			for range iterations {
				var scope, err = dev.NewScope()
				if err != nil {
					errs <- err
					return
				}
				if err = scope.MaxPool2D(inTensor, maxOut, cfg); err != nil {
					_ = scope.Close()
					errs <- err
					return
				}
				if err = scope.AvgPool2D(inTensor, avgOut, cfg); err != nil {
					_ = scope.Close()
					errs <- err
					return
				}
				if err = scope.Submit(ctx); err != nil {
					_ = scope.Close()
					errs <- err
					return
				}
				if err = scope.Wait(ctx); err != nil {
					_ = scope.Close()
					errs <- err
					return
				}
				if err = scope.Close(); err != nil {
					errs <- err
					return
				}
				if err = maxOut.DownloadInto(gotMax); err != nil {
					errs <- err
					return
				}
				if err = avgOut.DownloadInto(gotAvg); err != nil {
					errs <- err
					return
				}
				for i := range gotMax {
					if float64(gotMax[i]) != wantMax[i] || math.Abs(float64(gotAvg[i])-wantAvg[i]) > 1e-5 {
						errs <- ErrInvalid
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("concurrent Pool2D worker failed: %v", err)
	}

	requireError(t, inTensor.Close(), nil)
	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("native resource leak: buffers %d->%d scopes %d->%d",
			initial.LiveBuffers, after.LiveBuffers, initial.LiveScopes, after.LiveScopes)
	}
}
