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

func Test_MetalActivationsAgainstFP64(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		inf     = float32(math.Inf(1))
		nan     = float32(math.NaN())
		negZero = float32(math.Copysign(0, -1))
		edges   = []float32{
			0, negZero, 1e-40, -1e-40, math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
			1e-30, -1e-30, 1e-6, -1e-6, 0.5, -0.5, 1, -1, 9.5, -9.5, 10, -10, 20, -20,
			80, -80, 88, -88, 100, -100, 1000, -1000,
			math.MaxFloat32, -math.MaxFloat32, inf, -inf, nan,
		}
	)
	var operations = []struct {
		name      string
		encode    func(*Scope, *Tensor, *Tensor) error
		reference func(float64) float64
		tolerance float64
	}{
		{
			name:   "relu",
			encode: (*Scope).ReLU,
			reference: func(x float64) float64 {
				if math.IsNaN(x) {
					return x
				}
				if x > 0 {
					return x
				}
				return 0
			},
			tolerance: 0,
		},
		{
			name:   "sigmoid",
			encode: (*Scope).Sigmoid,
			reference: func(x float64) float64 {
				if math.IsNaN(x) {
					return x
				}
				if x >= 0 {
					return 1 / (1 + math.Exp(-x))
				}
				var e = math.Exp(x)
				return e / (1 + e)
			},
			tolerance: 2e-6,
		},
		{
			name:   "tanh",
			encode: (*Scope).Tanh,
			reference: func(x float64) float64 {
				if math.IsNaN(x) {
					return x
				}
				return math.Tanh(x)
			},
			tolerance: 2e-6,
		},
	}

	var shapes = [][]int64{
		{1},
		{3},
		{17},
		{31},
		{32},
		{33},
		{2, 64},
		{3, 5, 17},
		{1, 257},
		{2, 2, 4, 33},
		{4096},
	}
	for _, shape := range shapes {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			var (
				input  = requireTensor(t, gpu, shape...)
				count  = int(input.state.storage.count)
				random = rand.New(rand.NewPCG(0x4e4e14, uint64(count)))
				values = make([]float32, count)
			)
			copy(values, edges)
			for i := len(edges); i < count; i++ {
				values[i] = 40*random.Float32() - 20
			}
			requireError(t, input.Upload(values), nil)

			for _, op := range operations {
				t.Run(op.name, func(t *testing.T) {
					var (
						destination = requireTensor(t, gpu, shape...)
						scope       = requireScope(t, gpu)
						got         = make([]float32, count)
						want        = make([]float64, count)
					)
					requireError(t, op.encode(scope, input, destination), nil)
					requireError(t, scope.Submit(ctx), nil)
					requireError(t, scope.Wait(ctx), nil)
					requireError(t, scope.Close(), nil)
					requireError(t, destination.DownloadInto(got), nil)
					for i, v := range values {
						want[i] = op.reference(float64(v))
					}
					checkAgainstFP64(t, op.name, got, want, op.tolerance)
				})
			}
		})
	}

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("activation tests leaked native resources: before %+v; after %+v", initial, after)
	}
}

func Test_MetalSoftmaxAgainstFP64(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
	)

	// Test last-axis Softmax across odd/prime/power-of-two/vocab widths.
	var lastAxisShapes = [][]int64{
		{1},
		{3},
		{17},
		{31},
		{32},
		{33},
		{4, 64},
		{3, 128},
		{2, 257},
		{2, 1024},
		{4, 4096},
		{1, 128256},
		{2, 3, 5, 17},
	}
	for _, shape := range lastAxisShapes {
		t.Run(fmt.Sprintf("last/%v", shape), func(t *testing.T) {
			var (
				input       = requireTensor(t, gpu, shape...)
				destination = requireTensor(t, gpu, shape...)
				axisDest    = requireTensor(t, gpu, shape...)
				count       = int(input.state.storage.count)
				random      = rand.New(rand.NewPCG(0x50f714, uint64(count)))
				values      = make([]float32, count)
			)
			for i := range values {
				values[i] = 16*random.Float32() - 8
			}
			requireError(t, input.Upload(values), nil)

			var scope = requireScope(t, gpu)
			requireError(t, scope.Softmax(input, destination), nil)
			requireError(t, scope.SoftmaxAxis(input, axisDest, len(shape)-1), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)

			var (
				got     = make([]float32, count)
				gotAxis = make([]float32, count)
				want    = cpuSoftmaxFP64(values, shape, len(shape)-1)
			)
			requireError(t, destination.DownloadInto(got), nil)
			requireError(t, axisDest.DownloadInto(gotAxis), nil)
			checkSoftmaxAgainstFP64(t, fmt.Sprintf("softmax%v", shape), got, want, 2e-6)
			if !slices.Equal(got, gotAxis) {
				t.Fatalf("Softmax and SoftmaxAxis(rank-1) differed for shape %v", shape)
			}
		})
	}

	// Test SoftmaxAxis across every axis of 2D, 3D, and 4D tensors (covering inner > 1 and inner == 1).
	var axisShapes = [][]int64{
		{1, 1},
		{3, 5},
		{17, 33},
		{2, 3, 4},
		{3, 17, 5},
		{2, 4, 3, 8},
		{2, 5, 7, 13},
	}
	for _, shape := range axisShapes {
		for axis := range len(shape) {
			t.Run(fmt.Sprintf("axis/%v/a%d", shape, axis), func(t *testing.T) {
				var (
					input       = requireTensor(t, gpu, shape...)
					destination = requireTensor(t, gpu, shape...)
					count       = int(input.state.storage.count)
					random      = rand.New(rand.NewPCG(0x50f7a0+uint64(axis), uint64(count)))
					values      = make([]float32, count)
				)
				for i := range values {
					values[i] = 12*random.Float32() - 6
				}
				requireError(t, input.Upload(values), nil)

				var scope = requireScope(t, gpu)
				requireError(t, scope.SoftmaxAxis(input, destination, axis), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)

				var (
					got  = make([]float32, count)
					want = cpuSoftmaxFP64(values, shape, axis)
				)
				requireError(t, destination.DownloadInto(got), nil)
				checkSoftmaxAgainstFP64(t, fmt.Sprintf("softmaxAxis%v[%d]", shape, axis), got, want, 2e-6)
			})
		}
	}

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("softmax tests leaked native resources: before %+v; after %+v", initial, after)
	}
}

func Test_MetalSoftmaxEdgeCases(t *testing.T) {
	var (
		gpu = requireMetalDevice(t)
		ctx = context.Background()
		inf = float32(math.Inf(1))
		nan = float32(math.NaN())
	)

	type edgeCase struct {
		name  string
		shape []int64
		axis  int
		input []float32
	}
	var cases = []edgeCase{
		{
			name:  "axisSize1_finite",
			shape: []int64{4, 1},
			axis:  1,
			input: []float32{-1000, 0, 42.5, 1000},
		},
		{
			name:  "axisSize1_special",
			shape: []int64{3, 1},
			axis:  1,
			input: []float32{nan, inf, -inf},
		},
		{
			name:  "large_shift_float4",
			shape: []int64{2, 4},
			axis:  1,
			input: []float32{1000, 1001, 999, 1000.5, -1000, -999, -1001, -999.5},
		},
		{
			name:  "large_shift_scalar",
			shape: []int64{2, 3},
			axis:  1,
			input: []float32{1000, 1001, 999, -1000, -999, -1001},
		},
		{
			name:  "masked_neg_inf_float4",
			shape: []int64{2, 4},
			axis:  1,
			input: []float32{1, -inf, 2, -inf, -inf, -inf, 3, -inf},
		},
		{
			name:  "masked_neg_inf_scalar",
			shape: []int64{2, 3},
			axis:  1,
			input: []float32{1, -inf, 2, -inf, 5, -inf},
		},
		{
			name:  "nan_row_float4",
			shape: []int64{2, 4},
			axis:  1,
			input: []float32{1, nan, 2, 3, 0, 1, 2, 3},
		},
		{
			name:  "nan_row_scalar",
			shape: []int64{2, 3},
			axis:  1,
			input: []float32{1, nan, 2, 0, 1, 2},
		},
		{
			name:  "pos_inf_and_all_neg_inf_float4",
			shape: []int64{2, 4},
			axis:  1,
			input: []float32{1, inf, 2, 3, -inf, -inf, -inf, -inf},
		},
		{
			name:  "strided_special_values",
			shape: []int64{3, 4},
			axis:  0,
			input: []float32{
				1, 1000, -inf, nan,
				-inf, 1001, -inf, 1,
				2, 999, -inf, 2,
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var (
				input       = requireTensor(t, gpu, tc.shape...)
				destination = requireTensor(t, gpu, tc.shape...)
				got         = make([]float32, len(tc.input))
				want        = cpuSoftmaxFP64(tc.input, tc.shape, tc.axis)
				scope       = requireScope(t, gpu)
			)
			requireError(t, input.Upload(tc.input), nil)
			requireError(t, scope.SoftmaxAxis(input, destination, tc.axis), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)
			requireError(t, destination.DownloadInto(got), nil)
			checkSoftmaxAgainstFP64(t, tc.name, got, want, 2e-6)
		})
	}
}

func Test_MetalSoftmaxAndActivationsViewAliasAndConcurrency(t *testing.T) {
	var (
		gpu      = requireMetalDevice(t)
		ctx      = context.Background()
		shape4   = []int64{8, 256}
		shapeOdd = []int64{8, 257}
		shape3D  = []int64{4, 17, 8}
		in4      = requireTensor(t, gpu, shape4...)
		inOdd    = requireTensor(t, gpu, shapeOdd...)
		in3D     = requireTensor(t, gpu, shape3D...)
		view4, e = in4.View([]int64{8, 256})
	)
	requireError(t, e, nil)
	defer func() { requireError(t, view4.Close(), nil) }()

	var (
		data4   = make([]float32, 8*256)
		dataOdd = make([]float32, 8*257)
		data3D  = make([]float32, 4*17*8)
		random  = rand.New(rand.NewPCG(0x14c00c, 0x2026))
	)
	for i := range data4 {
		data4[i] = 10*random.Float32() - 5
	}
	for i := range dataOdd {
		dataOdd[i] = 10*random.Float32() - 5
	}
	for i := range data3D {
		data3D[i] = 10*random.Float32() - 5
	}
	requireError(t, in4.Upload(data4), nil)
	requireError(t, inOdd.Upload(dataOdd), nil)
	requireError(t, in3D.Upload(data3D), nil)

	// View-alias rejection on real Metal device.
	var aliasScope = requireScope(t, gpu)
	requireError(t, aliasScope.ReLU(in4, view4), ErrInvalid)
	requireError(t, aliasScope.Close(), nil)
	aliasScope = requireScope(t, gpu)
	requireError(t, aliasScope.Softmax(in4, view4), ErrInvalid)
	requireError(t, aliasScope.Close(), nil)

	// Concurrent multi-scope stress test to verify zero WAR threadgroup races.
	var (
		want4   = cpuSoftmaxFP64(data4, shape4, 1)
		wantOdd = cpuSoftmaxFP64(dataOdd, shapeOdd, 1)
		want3D  = cpuSoftmaxFP64(data3D, shape3D, 1)
		wg      sync.WaitGroup
		errCh   = make(chan error, 8)
	)
	for worker := range 8 {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			var (
				out4   = requireTensor(t, gpu, shape4...)
				outOdd = requireTensor(t, gpu, shapeOdd...)
				out3D  = requireTensor(t, gpu, shape3D...)
				actOut = requireTensor(t, gpu, shape4...)
				got4   = make([]float32, len(data4))
				gotOdd = make([]float32, len(dataOdd))
				got3D  = make([]float32, len(data3D))
			)
			for range 12 {
				var scope, err = gpu.NewScope()
				if err != nil {
					errCh <- err
					return
				}
				if err = scope.Softmax(in4, out4); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.Softmax(inOdd, outOdd); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.SoftmaxAxis(in3D, out3D, 1); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.ReLU(in4, actOut); err != nil {
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
				if err = out4.DownloadInto(got4); err != nil {
					errCh <- err
					return
				}
				if err = outOdd.DownloadInto(gotOdd); err != nil {
					errCh <- err
					return
				}
				if err = out3D.DownloadInto(got3D); err != nil {
					errCh <- err
					return
				}
				if err = verifySoftmaxQuiet(got4, want4, 2e-6); err != nil {
					errCh <- fmt.Errorf("worker %d float4 softmax: %w", id, err)
					return
				}
				if err = verifySoftmaxQuiet(gotOdd, wantOdd, 2e-6); err != nil {
					errCh <- fmt.Errorf("worker %d scalar softmax: %w", id, err)
					return
				}
				if err = verifySoftmaxQuiet(got3D, want3D, 2e-6); err != nil {
					errCh <- fmt.Errorf("worker %d strided softmax: %w", id, err)
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
}

func Test_MetalResidentMLPSingleScope(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		rows    = int64(8)
		inDim   = int64(64)
		hidden  = int64(128)
		classes = int64(33)
		random  = rand.New(rand.NewPCG(0x14a1b2, 0x3344))
		xData   = make([]float32, rows*inDim)
		w1Data  = make([]float32, hidden*inDim)
		w2Data  = make([]float32, classes*hidden)
	)
	for i := range xData {
		xData[i] = 2*random.Float32() - 1
	}
	for i := range w1Data {
		w1Data[i] = (2*random.Float32() - 1) / float32(math.Sqrt(float64(inDim)))
	}
	for i := range w2Data {
		w2Data[i] = (2*random.Float32() - 1) / float32(math.Sqrt(float64(hidden)))
	}

	var (
		x      = requireTensor(t, gpu, rows, inDim)
		w1     = requireTensor(t, gpu, hidden, inDim)
		w2     = requireTensor(t, gpu, classes, hidden)
		h1     = requireTensor(t, gpu, rows, hidden)
		h1Relu = requireTensor(t, gpu, rows, hidden)
		logits = requireTensor(t, gpu, rows, classes)
		probs  = requireTensor(t, gpu, rows, classes)
	)
	requireError(t, x.Upload(xData), nil)
	requireError(t, w1.Upload(w1Data), nil)
	requireError(t, w2.Upload(w2Data), nil)

	var before = native.ResourceSnapshot()
	var scope = requireScope(t, gpu)
	requireError(t, scope.MatMulRightTranspose(x, w1, h1), nil)
	requireError(t, scope.ReLU(h1, h1Relu), nil)
	requireError(t, scope.MatMulRightTranspose(h1Relu, w2, logits), nil)
	requireError(t, scope.Softmax(logits, probs), nil)
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)

	var after = native.ResourceSnapshot()
	if after.SubmittedCommands != before.SubmittedCommands+1 || after.CompletedCommands != before.CompletedCommands+1 {
		t.Fatalf("resident MLP did not execute in 1 scope submit: before=%+v after=%+v", before, after)
	}

	var got = make([]float32, rows*classes)
	requireError(t, probs.DownloadInto(got), nil)

	// Independent FP64 reference: MatMulRightTranspose -> ReLU -> MatMulRightTranspose -> Softmax.
	var refH1 = make([]float64, rows*hidden)
	for r := range rows {
		for c := range hidden {
			var sum float64
			for k := range inDim {
				sum += float64(xData[r*inDim+k]) * float64(w1Data[c*inDim+k])
			}
			if sum > 0 {
				refH1[r*hidden+c] = sum
			}
		}
	}
	var refLogits = make([]float32, rows*classes)
	for r := range rows {
		for c := range classes {
			var sum float64
			for k := range hidden {
				sum += refH1[r*hidden+k] * float64(w2Data[c*hidden+k])
			}
			refLogits[r*classes+c] = float32(sum)
		}
	}
	var want = cpuSoftmaxFP64(refLogits, []int64{rows, classes}, 1)
	checkSoftmaxAgainstFP64(t, "resident_mlp", got, want, 1e-5)

	// Verify cleanup leaves zero leaks.
	for _, tensor := range []*Tensor{x, w1, w2, h1, h1Relu, logits, probs} {
		requireError(t, tensor.Close(), nil)
	}
	var final = native.ResourceSnapshot()
	if final.LiveBuffers != initial.LiveBuffers || final.LiveBufferBytes != initial.LiveBufferBytes || final.LiveScopes != initial.LiveScopes {
		t.Fatalf("resident MLP leaked native resources: before=%+v after=%+v", initial, final)
	}
}

func cpuSoftmaxFP64(input []float32, shape []int64, axis int) (output []float64) {
	var (
		outer    = 1
		axisSize = int(shape[axis])
		inner    = 1
	)
	for i := range axis {
		outer *= int(shape[i])
	}
	for i := axis + 1; i < len(shape); i++ {
		inner *= int(shape[i])
	}
	output = make([]float64, len(input))
	for o := range outer {
		for inIdx := range inner {
			var (
				base   = (o*axisSize)*inner + inIdx
				rowMax = math.Inf(-1)
			)
			for a := range axisSize {
				var v = float64(input[base+a*inner])
				if math.IsNaN(v) {
					rowMax = math.NaN()
					break
				}
				if v > rowMax {
					rowMax = v
				}
			}
			var sum float64
			for a := range axisSize {
				var e = math.Exp(float64(input[base+a*inner]) - rowMax)
				output[base+a*inner] = e
				sum += e
			}
			for a := range axisSize {
				output[base+a*inner] /= sum
			}
		}
	}
	return output
}

func checkAgainstFP64(t *testing.T, label string, got []float32, want []float64, tolerance float64) {
	t.Helper()
	const smallestNormal = 0x1p-126
	var (
		deltas      = make([]float64, 0, len(got))
		maxRelative float64
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
			limit = tolerance * math.Abs(expected)
		)
		if expected != 0 && math.Abs(expected) < smallestNormal {
			limit = smallestNormal
		}
		if math.IsNaN(delta) || delta > limit {
			t.Fatalf("%s element %d: got %.9g, FP64 reference %.9g, abs error %.3g exceeds %.3g",
				label, i, actual, expected, delta, limit)
		}
		deltas = append(deltas, delta)
		if math.Abs(expected) >= smallestNormal {
			maxRelative = max(maxRelative, delta/math.Abs(expected))
		}
	}
	if len(deltas) == 0 {
		return
	}
	slices.Sort(deltas)
	var last = len(deltas) - 1
	t.Logf("%s FP64 abs error (%d finite): p50=%.3g p95=%.3g p99=%.3g max=%.3g maxRel=%.3g",
		label, len(deltas), deltas[last*50/100], deltas[last*95/100], deltas[last*99/100], deltas[last], maxRelative)
}

func checkSoftmaxAgainstFP64(t *testing.T, label string, got []float32, want []float64, tolerance float64) {
	t.Helper()
	var deltas = make([]float64, 0, len(got))
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
		var delta = math.Abs(actual - expected)
		if math.IsNaN(delta) || delta > tolerance {
			t.Fatalf("%s element %d: got %.9g, FP64 reference %.9g, abs error %.3g exceeds %.3g",
				label, i, actual, expected, delta, tolerance)
		}
		deltas = append(deltas, delta)
	}
	if len(deltas) == 0 {
		return
	}
	slices.Sort(deltas)
	var last = len(deltas) - 1
	t.Logf("%s FP64 abs error (%d finite): p50=%.3g p95=%.3g p99=%.3g max=%.3g",
		label, len(deltas), deltas[last*50/100], deltas[last*95/100], deltas[last*99/100], deltas[last])
}

func verifySoftmaxQuiet(got []float32, want []float64, tolerance float64) (err error) {
	for i, value := range got {
		var (
			actual   = float64(value)
			expected = want[i]
		)
		if math.IsNaN(expected) {
			if !math.IsNaN(actual) {
				return fmt.Errorf("element %d: got %g, want NaN", i, actual)
			}
			continue
		}
		var delta = math.Abs(actual - expected)
		if math.IsNaN(delta) || delta > tolerance {
			return fmt.Errorf("element %d: got %.9g, want %.9g (delta %.3g > %.3g)", i, actual, expected, delta, tolerance)
		}
	}
	return nil
}
