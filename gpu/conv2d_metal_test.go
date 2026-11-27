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

func refConv2DFP64(input []float32, inShape []int64, weights []float32, wShape []int64, bias []float32, cfg Conv2DConfig) (out []float64) {
	var (
		outShape, _   = Conv2DShape(inShape, wShape, cfg)
		n             = int(inShape[0])
		inC           = int(inShape[1])
		inH           = int(inShape[2])
		inW           = int(inShape[3])
		outC          = int(wShape[0])
		inCPerGroup   = int(wShape[1])
		kH            = int(wShape[2])
		kW            = int(wShape[3])
		outH          = int(outShape[2])
		outW          = int(outShape[3])
		outCPerGroup  = outC / int(cfg.Groups)
		inSpatial     = inH * inW
		kernelSpatial = kH * kW
	)
	out = make([]float64, n*outC*outH*outW)
	var idx int
	for b := range n {
		for oc := range outC {
			var (
				g       = oc / outCPerGroup
				icStart = g * inCPerGroup
			)
			for oh := range outH {
				var startH = int64(oh)*cfg.Strides[0] - cfg.Pads[0]
				for ow := range outW {
					var (
						startW = int64(ow)*cfg.Strides[1] - cfg.Pads[1]
						sum    float64
					)
					for ic := range inCPerGroup {
						var (
							inChanBase = (b*inC + icStart + ic) * inSpatial
							wChanBase  = (oc*inCPerGroup + ic) * kernelSpatial
						)
						for kh := range kH {
							var ih = startH + int64(kh)*cfg.Dilations[0]
							if ih < 0 || ih >= int64(inH) {
								continue
							}
							for kw := range kW {
								var iw = startW + int64(kw)*cfg.Dilations[1]
								if iw < 0 || iw >= int64(inW) {
									continue
								}
								sum += float64(input[inChanBase+int(ih)*inW+int(iw)]) * float64(weights[wChanBase+kh*kW+kw])
							}
						}
					}
					if bias != nil {
						sum += float64(bias[oc])
					}
					out[idx] = sum
					idx++
				}
			}
		}
	}
	return out
}

func roundToDType(values []float32, dtype DType) (out []float32) {
	out = make([]float32, len(values))
	switch dtype {
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

func conv2DErrorStats(got []float32, ref []float64) (p50, p95, p99, maxErr float64) {
	var errs = make([]float64, len(got))
	for i := range got {
		var d = math.Abs(float64(got[i]) - ref[i])
		errs[i] = d
		if d > maxErr {
			maxErr = d
		}
	}
	sort.Float64s(errs)
	if len(errs) == 0 {
		return 0, 0, 0, 0
	}
	return errs[len(errs)*50/100], errs[len(errs)*95/100], errs[len(errs)*99/100], maxErr
}

func Test_Conv2D_Metal_Parity(t *testing.T) {
	var (
		dev = requireMetalDevice(t)
		ctx = context.Background()
	)
	type testcase struct {
		name    string
		inShape []int64
		wShape  []int64
		cfg     Conv2DConfig
		withB   bool
	}
	var cases = []testcase{
		{
			name:    "1x1_direct_small",
			inShape: []int64{1, 4, 3, 3},
			wShape:  []int64{5, 4, 1, 1},
			cfg:     Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1},
			withB:   true,
		},
		{
			name:    "1x1_implicit_gemm",
			inShape: []int64{2, 33, 7, 7},
			wShape:  []int64{65, 33, 1, 1},
			cfg:     Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1},
			withB:   true,
		},
		{
			name:    "3x3_implicit_gemm_odd",
			inShape: []int64{3, 11, 9, 10},
			wShape:  []int64{33, 11, 3, 3},
			cfg:     Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 1},
			withB:   true,
		},
		{
			name:    "3x3_no_bias_resnet_style",
			inShape: []int64{2, 32, 14, 14},
			wShape:  []int64{64, 32, 3, 3},
			cfg:     Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 1},
			withB:   false,
		},
		{
			name:    "5x5_strided_asymmetric_pad",
			inShape: []int64{2, 8, 15, 17},
			wShape:  []int64{16, 8, 5, 5},
			cfg:     Conv2DConfig{Strides: [2]int64{2, 3}, Pads: [4]int64{2, 1, 1, 2}, Dilations: [2]int64{1, 1}, Groups: 1},
			withB:   true,
		},
		{
			name:    "7x7_resnet_stem",
			inShape: []int64{2, 3, 28, 28},
			wShape:  []int64{32, 3, 7, 7},
			cfg:     Conv2DConfig{Strides: [2]int64{2, 2}, Pads: [4]int64{3, 3, 3, 3}, Dilations: [2]int64{1, 1}, Groups: 1},
			withB:   true,
		},
		{
			name:    "dilated_3x3",
			inShape: []int64{2, 10, 15, 13},
			wShape:  []int64{18, 10, 3, 3},
			cfg:     Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{2, 3, 2, 3}, Dilations: [2]int64{2, 3}, Groups: 1},
			withB:   true,
		},
		{
			name:    "grouped_g2_implicit_gemm",
			inShape: []int64{2, 16, 8, 8},
			wShape:  []int64{24, 8, 3, 3},
			cfg:     Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 2},
			withB:   true,
		},
		{
			name:    "grouped_g4_direct_fallback",
			inShape: []int64{2, 12, 7, 7},
			wShape:  []int64{16, 3, 3, 3},
			cfg:     Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 4},
			withB:   true,
		},
		{
			name:    "depthwise_g_eq_c",
			inShape: []int64{2, 16, 11, 11},
			wShape:  []int64{16, 1, 3, 3},
			cfg:     Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 16},
			withB:   true,
		},
		{
			name:    "edge_1x1_spatial_out",
			inShape: []int64{1, 12, 3, 3},
			wShape:  []int64{16, 12, 3, 3},
			cfg:     Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1},
			withB:   true,
		},
	}

	var dtypes = []DType{Float32, Float16, BFloat16}

	for _, tc := range cases {
		var (
			outShape, shapeErr = Conv2DShape(tc.inShape, tc.wShape, tc.cfg)
			inCount            = tc.inShape[0] * tc.inShape[1] * tc.inShape[2] * tc.inShape[3]
			wCount             = tc.wShape[0] * tc.wShape[1] * tc.wShape[2] * tc.wShape[3]
			outCount           = outShape[0] * outShape[1] * outShape[2] * outShape[3]
			inData             = attentionValues(inCount, 301, 1)
			wData              = attentionValues(wCount, 409, 2)
			bData              []float32
		)
		requireError(t, shapeErr, nil)
		if tc.withB {
			bData = attentionValues(tc.wShape[0], 503, 3)
		}

		for _, dt := range dtypes {
			t.Run(tc.name+"/"+dt.String(), func(t *testing.T) {
				var (
					inTensor  = requireTensor(t, dev, tc.inShape...)
					wTensor   = requireTensorWithDType(t, dev, dt, tc.wShape...)
					outTensor = requireTensor(t, dev, outShape...)
					bTensor   *Tensor
				)
				requireError(t, inTensor.Upload(inData), nil)
				if dt == Float32 {
					requireError(t, wTensor.Upload(wData), nil)
				} else {
					requireError(t, wTensor.UploadFloat32As(wData), nil)
				}
				if tc.withB {
					bTensor = requireTensorWithDType(t, dev, dt, tc.wShape[0])
					if dt == Float32 {
						requireError(t, bTensor.Upload(bData), nil)
					} else {
						requireError(t, bTensor.UploadFloat32As(bData), nil)
					}
				}

				var scope, err = dev.NewScope()
				requireError(t, err, nil)
				defer scope.Close()
				requireError(t, scope.Conv2D(inTensor, wTensor, bTensor, outTensor, tc.cfg), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)

				var got = make([]float32, outCount)
				requireError(t, outTensor.DownloadInto(got), nil)

				var (
					effW   = roundToDType(wData, dt)
					effB   []float32
					refF64 []float64
				)
				if tc.withB {
					effB = roundToDType(bData, dt)
				}
				refF64 = refConv2DFP64(inData, tc.inShape, effW, tc.wShape, effB, tc.cfg)
				var p50, p95, p99, maxErr = conv2DErrorStats(got, refF64)
				var tol = 5e-5
				if maxErr > tol {
					t.Fatalf("%s/%s: max abs err vs FP64 = %e > %e (p50=%e p95=%e p99=%e)", tc.name, dt, maxErr, tol, p50, p95, p99)
				}
				t.Logf("%s/%s: n=%d p50=%.3e p95=%.3e p99=%.3e max=%.3e", tc.name, dt, outCount, p50, p95, p99, maxErr)

				// Also cross-check FP32 against tensor.Conv2DInto.
				if dt == Float32 {
					var (
						cpuIn, e1  = tensor.FromFloat32(tc.inShape, inData)
						cpuW, e2   = tensor.FromFloat32(tc.wShape, wData)
						cpuOut, e3 = tensor.FromFloat32(outShape, make([]float32, outCount))
						cpuB       *tensor.Tensor
					)
					requireError(t, e1, nil)
					requireError(t, e2, nil)
					requireError(t, e3, nil)
					if tc.withB {
						var eb error
						cpuB, eb = tensor.FromFloat32([]int64{tc.wShape[0]}, bData)
						requireError(t, eb, nil)
					}
					requireError(t, tensor.Conv2DInto(cpuIn, cpuW, cpuB, cpuOut, tensor.Conv2DConfig{
						Strides:   tc.cfg.Strides,
						Dilations: tc.cfg.Dilations,
						Pads:      tc.cfg.Pads,
						Groups:    tc.cfg.Groups,
					}), nil)
					var cpuVals, e4 = cpuOut.Float32Values()
					requireError(t, e4, nil)
					var maxCPU float64
					for i := range got {
						var d = math.Abs(float64(got[i]) - float64(cpuVals[i]))
						if d > maxCPU {
							maxCPU = d
						}
					}
					if maxCPU > tol {
						t.Fatalf("%s/f32: max abs err vs tensor.Conv2DInto = %e > %e", tc.name, maxCPU, tol)
					}
				}
			})
		}
	}
}

func Test_Conv2D_Metal_SpecialValues(t *testing.T) {
	var (
		dev = requireMetalDevice(t)
		ctx = context.Background()
		nan = float32(math.NaN())
		inf = float32(math.Inf(1))
		neg = float32(math.Inf(-1))
	)

	// Test both implicit-GEMM (M_gemm>=8, N_gemm>=8, K_gemm>=8 with odd K not div by 32)
	// and direct fallback (M_gemm<8 or N_gemm<8).
	type specialCase struct {
		name    string
		inShape []int64
		wShape  []int64
		cfg     Conv2DConfig
	}
	var cases = []specialCase{
		{
			name:    "implicit_gemm_odd_K_padded",
			inShape: []int64{1, 3, 4, 4}, // M_gemm=16, N_gemm=10, K_gemm=27 (not div by 32)
			wShape:  []int64{10, 3, 3, 3},
			cfg:     Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 1},
		},
		{
			name:    "direct_fallback_padded",
			inShape: []int64{1, 2, 3, 3}, // M_gemm=9, N_gemm=4 < 8
			wShape:  []int64{4, 2, 3, 3},
			cfg:     Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 1},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var (
				outShape, shapeErr = Conv2DShape(tc.inShape, tc.wShape, tc.cfg)
				inCount            = tc.inShape[0] * tc.inShape[1] * tc.inShape[2] * tc.inShape[3]
				wCount             = tc.wShape[0] * tc.wShape[1] * tc.wShape[2] * tc.wShape[3]
				outCount           = outShape[0] * outShape[1] * outShape[2] * outShape[3]
				inData             = make([]float32, inCount)
				wData              = make([]float32, wCount)
			)
			requireError(t, shapeErr, nil)
			for i := range inData {
				inData[i] = 1.0
			}
			for i := range wData {
				wData[i] = 1.0
			}
			// Inject NaN, +Inf, and -Inf at distinct spatial pixels.
			inData[0] = nan
			inData[int(tc.inShape[3])-1] = inf
			inData[int(tc.inShape[2]*tc.inShape[3])-1] = neg

			var (
				inTensor  = requireTensor(t, dev, tc.inShape...)
				wTensor   = requireTensor(t, dev, tc.wShape...)
				outTensor = requireTensor(t, dev, outShape...)
			)
			requireError(t, inTensor.Upload(inData), nil)
			requireError(t, wTensor.Upload(wData), nil)

			var scope, err = dev.NewScope()
			requireError(t, err, nil)
			defer scope.Close()
			requireError(t, scope.Conv2D(inTensor, wTensor, nil, outTensor, tc.cfg), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)

			var got = make([]float32, outCount)
			requireError(t, outTensor.DownloadInto(got), nil)

			var ref = refConv2DFP64(inData, tc.inShape, wData, tc.wShape, nil, tc.cfg)
			for i := range got {
				var g = float64(got[i])
				var r = ref[i]
				if math.IsNaN(r) {
					if !math.IsNaN(g) {
						t.Fatalf("idx %d: got %v, want NaN", i, g)
					}
					continue
				}
				if math.IsInf(r, 0) {
					if !math.IsInf(g, int(math.Copysign(1, r))) {
						t.Fatalf("idx %d: got %v, want %v", i, g, r)
					}
					continue
				}
				if math.IsNaN(g) || math.IsInf(g, 0) || math.Abs(g-r) > 1e-4 {
					t.Fatalf("idx %d: got %v, want %v", i, g, r)
				}
			}
		})
	}
}

func Test_Conv2D_Metal_ViewAliasRejected(t *testing.T) {
	var (
		dev  = requireMetalDevice(t)
		in   = requireTensor(t, dev, 1, 8, 4, 4)
		w    = requireTensor(t, dev, 8, 8, 3, 3)
		cfg  = Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		v, e = in.View([]int64{1, 8, 4, 4})
	)
	requireError(t, e, nil)
	defer v.Close()
	requireError(t, in.Upload(attentionValues(1*8*4*4, 17, 1)), nil)
	requireError(t, w.Upload(attentionValues(8*8*3*3, 19, 2)), nil)

	var scope, err = dev.NewScope()
	requireError(t, err, nil)
	defer scope.Close()
	requireError(t, scope.Conv2D(in, w, nil, v, cfg), ErrInvalid)
}

func Test_Conv2D_Metal_ResNetBottleneck_SingleScope(t *testing.T) {
	var (
		dev   = requireMetalDevice(t)
		ctx   = context.Background()
		n     = int64(2)
		cIn   = int64(32)
		cMid  = int64(16)
		h, w  = int64(14), int64(14)
		eps   = float32(1e-5)
		cfg1  = Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		cfg3  = Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		pCfg  = Pool2DConfig{Kernel: [2]int64{3, 3}, Strides: [2]int64{2, 2}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}}
		inDat = attentionValues(n*cIn*h*w, 701, 1)
		w1Dat = attentionValues(cMid*cIn*1*1, 709, 2)
		w2Dat = attentionValues(cMid*cMid*3*3, 719, 3)
		w3Dat = attentionValues(cIn*cMid*1*1, 727, 4)
	)
	makeBN := func(ch int64, seed uint64) (sc, bi, mn, va []float32) {
		sc = attentionValues(ch, seed, 1)
		bi = attentionValues(ch, seed+1, 2)
		mn = attentionValues(ch, seed+2, 3)
		va = make([]float32, ch)
		for i := range va {
			va[i] = 0.5 + float32(i%7)*0.1
		}
		return sc, bi, mn, va
	}
	var (
		sc1, bi1, mn1, va1 = makeBN(cMid, 801)
		sc2, bi2, mn2, va2 = makeBN(cMid, 811)
		sc3, bi3, mn3, va3 = makeBN(cIn, 821)
	)

	// Allocate GPU tensors.
	var (
		inT    = requireTensor(t, dev, n, cIn, h, w)
		w1T    = requireTensor(t, dev, cMid, cIn, 1, 1)
		w2T    = requireTensor(t, dev, cMid, cMid, 3, 3)
		w3T    = requireTensor(t, dev, cIn, cMid, 1, 1)
		sc1T   = requireTensor(t, dev, cMid)
		bi1T   = requireTensor(t, dev, cMid)
		mn1T   = requireTensor(t, dev, cMid)
		va1T   = requireTensor(t, dev, cMid)
		sc2T   = requireTensor(t, dev, cMid)
		bi2T   = requireTensor(t, dev, cMid)
		mn2T   = requireTensor(t, dev, cMid)
		va2T   = requireTensor(t, dev, cMid)
		sc3T   = requireTensor(t, dev, cIn)
		bi3T   = requireTensor(t, dev, cIn)
		mn3T   = requireTensor(t, dev, cIn)
		va3T   = requireTensor(t, dev, cIn)
		c1Out  = requireTensor(t, dev, n, cMid, h, w)
		bn1Out = requireTensor(t, dev, n, cMid, h, w)
		r1Out  = requireTensor(t, dev, n, cMid, h, w)
		c2Out  = requireTensor(t, dev, n, cMid, h, w)
		bn2Out = requireTensor(t, dev, n, cMid, h, w)
		r2Out  = requireTensor(t, dev, n, cMid, h, w)
		c3Out  = requireTensor(t, dev, n, cIn, h, w)
		bn3Out = requireTensor(t, dev, n, cIn, h, w)
		addOut = requireTensor(t, dev, n, cIn, h, w)
		r3Out  = requireTensor(t, dev, n, cIn, h, w)
		pOut   = requireTensor(t, dev, n, cIn, h/2, w/2)
	)
	requireError(t, inT.Upload(inDat), nil)
	requireError(t, w1T.Upload(w1Dat), nil)
	requireError(t, w2T.Upload(w2Dat), nil)
	requireError(t, w3T.Upload(w3Dat), nil)
	requireError(t, sc1T.Upload(sc1), nil)
	requireError(t, bi1T.Upload(bi1), nil)
	requireError(t, mn1T.Upload(mn1), nil)
	requireError(t, va1T.Upload(va1), nil)
	requireError(t, sc2T.Upload(sc2), nil)
	requireError(t, bi2T.Upload(bi2), nil)
	requireError(t, mn2T.Upload(mn2), nil)
	requireError(t, va2T.Upload(va2), nil)
	requireError(t, sc3T.Upload(sc3), nil)
	requireError(t, bi3T.Upload(bi3), nil)
	requireError(t, mn3T.Upload(mn3), nil)
	requireError(t, va3T.Upload(va3), nil)

	// Execute full 11-op ResNet bottleneck in ONE gpu.Scope.
	var scope, err = dev.NewScope()
	requireError(t, err, nil)
	defer scope.Close()
	requireError(t, scope.Conv2D(inT, w1T, nil, c1Out, cfg1), nil)
	requireError(t, scope.BatchNorm(c1Out, sc1T, bi1T, mn1T, va1T, bn1Out, eps), nil)
	requireError(t, scope.ReLU(bn1Out, r1Out), nil)
	requireError(t, scope.Conv2D(r1Out, w2T, nil, c2Out, cfg3), nil)
	requireError(t, scope.BatchNorm(c2Out, sc2T, bi2T, mn2T, va2T, bn2Out, eps), nil)
	requireError(t, scope.ReLU(bn2Out, r2Out), nil)
	requireError(t, scope.Conv2D(r2Out, w3T, nil, c3Out, cfg1), nil)
	requireError(t, scope.BatchNorm(c3Out, sc3T, bi3T, mn3T, va3T, bn3Out, eps), nil)
	requireError(t, scope.Add(bn3Out, inT, addOut), nil)
	requireError(t, scope.ReLU(addOut, r3Out), nil)
	requireError(t, scope.MaxPool2D(r3Out, pOut, pCfg), nil)
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)

	var got = make([]float32, n*cIn*(h/2)*(w/2))
	requireError(t, pOut.DownloadInto(got), nil)

	// CPU reference using neuralnetwork/tensor.
	mustTensor := func(shape []int64, data []float32) *tensor.Tensor {
		var out, e = tensor.FromFloat32(shape, data)
		requireError(t, e, nil)
		return out
	}
	var (
		cInT  = mustTensor([]int64{n, cIn, h, w}, inDat)
		cW1   = mustTensor([]int64{cMid, cIn, 1, 1}, w1Dat)
		cW2   = mustTensor([]int64{cMid, cMid, 3, 3}, w2Dat)
		cW3   = mustTensor([]int64{cIn, cMid, 1, 1}, w3Dat)
		cTmp1 = mustTensor([]int64{n, cMid, h, w}, make([]float32, n*cMid*h*w))
		cTmp2 = mustTensor([]int64{n, cMid, h, w}, make([]float32, n*cMid*h*w))
		cTmp3 = mustTensor([]int64{n, cIn, h, w}, make([]float32, n*cIn*h*w))
		cTmp4 = mustTensor([]int64{n, cIn, h, w}, make([]float32, n*cIn*h*w))
		cPool = mustTensor([]int64{n, cIn, h / 2, w / 2}, make([]float32, n*cIn*(h/2)*(w/2)))
		tCfg1 = tensor.Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		tCfg3 = tensor.Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		tPCfg = tensor.Pool2DConfig{Kernel: [2]int64{3, 3}, Strides: [2]int64{2, 2}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}}
	)
	requireError(t, tensor.Conv2DInto(cInT, cW1, nil, cTmp1, tCfg1), nil)
	requireError(t, tensor.BatchNormalizationInto(cTmp1, mustTensor([]int64{cMid}, sc1), mustTensor([]int64{cMid}, bi1), mustTensor([]int64{cMid}, mn1), mustTensor([]int64{cMid}, va1), cTmp2, eps), nil)
	requireError(t, tensor.ReLUInto(cTmp2, cTmp1), nil)
	requireError(t, tensor.Conv2DInto(cTmp1, cW2, nil, cTmp2, tCfg3), nil)
	requireError(t, tensor.BatchNormalizationInto(cTmp2, mustTensor([]int64{cMid}, sc2), mustTensor([]int64{cMid}, bi2), mustTensor([]int64{cMid}, mn2), mustTensor([]int64{cMid}, va2), cTmp1, eps), nil)
	requireError(t, tensor.ReLUInto(cTmp1, cTmp2), nil)
	requireError(t, tensor.Conv2DInto(cTmp2, cW3, nil, cTmp3, tCfg1), nil)
	requireError(t, tensor.BatchNormalizationInto(cTmp3, mustTensor([]int64{cIn}, sc3), mustTensor([]int64{cIn}, bi3), mustTensor([]int64{cIn}, mn3), mustTensor([]int64{cIn}, va3), cTmp4, eps), nil)
	requireError(t, tensor.AddInto(cTmp4, cInT, cTmp3), nil)
	requireError(t, tensor.ReLUInto(cTmp3, cTmp4), nil)
	requireError(t, tensor.MaxPool2DInto(cTmp4, cPool, tPCfg), nil)

	var want, e5 = cPool.Float32Values()
	requireError(t, e5, nil)
	var maxErr float64
	for i := range got {
		var d = math.Abs(float64(got[i]) - float64(want[i]))
		if d > maxErr {
			maxErr = d
		}
	}
	if maxErr > 1e-4 {
		t.Fatalf("ResNet bottleneck 1-Scope max abs err vs CPU = %e > 1e-4", maxErr)
	}
	t.Logf("ResNet bottleneck 1-Scope max abs err vs CPU = %.3e", maxErr)
}

func Test_Conv2D_Metal_ConcurrentStress(t *testing.T) {
	var (
		dev      = requireMetalDevice(t)
		ctx      = context.Background()
		inShape  = []int64{2, 16, 10, 10}
		wShape   = []int64{24, 16, 3, 3}
		cfg      = Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		outShape = []int64{2, 24, 10, 10}
		inData   = attentionValues(2*16*10*10, 907, 1)
		wData    = attentionValues(24*16*3*3, 911, 2)
		bData    = attentionValues(24, 919, 3)
		ref      = refConv2DFP64(inData, inShape, wData, wShape, bData, cfg)
		sharedIn = requireTensor(t, dev, inShape...)
		sharedW  = requireTensor(t, dev, wShape...)
		sharedB  = requireTensor(t, dev, 24)
	)
	requireError(t, sharedIn.Upload(inData), nil)
	requireError(t, sharedW.Upload(wData), nil)
	requireError(t, sharedB.Upload(bData), nil)

	const workers = 8
	const iters = 16
	var wg sync.WaitGroup
	var errCh = make(chan error, workers)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var dst, err = dev.NewFloat32(outShape)
			if err != nil {
				errCh <- err
				return
			}
			defer dst.Close()
			var got = make([]float32, 2*24*10*10)
			for range iters {
				var scope, sErr = dev.NewScope()
				if sErr != nil {
					errCh <- sErr
					return
				}
				if e := scope.Conv2D(sharedIn, sharedW, sharedB, dst, cfg); e != nil {
					_ = scope.Close()
					errCh <- e
					return
				}
				if e := scope.Submit(ctx); e != nil {
					_ = scope.Close()
					errCh <- e
					return
				}
				if e := scope.Wait(ctx); e != nil {
					_ = scope.Close()
					errCh <- e
					return
				}
				if e := scope.Close(); e != nil {
					errCh <- e
					return
				}
				if e := dst.DownloadInto(got); e != nil {
					errCh <- e
					return
				}
				var _, _, _, maxErr = conv2DErrorStats(got, ref)
				if maxErr > 5e-5 {
					errCh <- ErrInvalid
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for e := range errCh {
		t.Fatalf("concurrent Conv2D worker failed: %v", e)
	}
}
