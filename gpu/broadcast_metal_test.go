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

func Test_MetalSubDivScaleAgainstFP64(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		inf     = float32(math.Inf(1))
		nan     = float32(math.NaN())
		negZero = float32(math.Copysign(0, -1))
		edgesA  = []float32{
			0, negZero, 1, -1, 0, 1, -1, inf, -inf, inf, -inf, inf, nan, 1,
			1e-30, -1e-30, 1e30, -1e30, math.MaxFloat32, -math.MaxFloat32,
			math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
		}
		edgesB = []float32{
			0, 0, 0, 0, negZero, -0.0, 2, inf, inf, -inf, -inf, 0, 1, nan,
			1e-30, 1e-30, 1e30, -1e30, 2, 2,
			2, 2,
		}
	)

	var shapes = [][]int64{
		{1},
		{3},
		{17},
		{31},
		{32},
		{33},
		{2, 64},
		{3, 5, 17},
		{2, 2, 4, 33},
		{4096},
	}

	type binaryOp struct {
		name      string
		encode    func(*Scope, *Tensor, *Tensor, *Tensor) error
		reference func(float64, float64) float64
		tolerance float64
	}
	var binaryOps = []binaryOp{
		{
			name:   "sub",
			encode: (*Scope).Sub,
			reference: func(a, b float64) float64 {
				return clampFP32Overflow(a - b)
			},
			tolerance: 1e-6,
		},
		{
			name:   "div",
			encode: (*Scope).Div,
			reference: func(a, b float64) float64 {
				return clampFP32Overflow(a / b)
			},
			tolerance: 1e-6,
		},
	}

	for _, op := range binaryOps {
		t.Run(op.name, func(t *testing.T) {
			for _, shape := range shapes {
				var (
					count  = elementCount(shape)
					left   = requireTensor(t, gpu, shape...)
					right  = requireTensor(t, gpu, shape...)
					output = requireTensor(t, gpu, shape...)
					inA    = make([]float32, count)
					inB    = make([]float32, count)
					got    = make([]float32, count)
					random = rand.New(rand.NewPCG(0x15a1, uint64(count)))
				)
				for i := range count {
					inA[i] = float32((random.Float64()*2 - 1) * 50)
					inB[i] = float32((random.Float64()*2 - 1) * 50)
					if op.name == "div" && math.Abs(float64(inB[i])) < 0.05 {
						inB[i] = 0.25
					}
				}
				if count >= len(edgesA) {
					copy(inA, edgesA)
					copy(inB, edgesB)
				}
				requireError(t, left.Upload(inA), nil)
				requireError(t, right.Upload(inB), nil)

				var scope = requireScope(t, gpu)
				requireError(t, op.encode(scope, left, right, output), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)

				requireError(t, output.DownloadInto(got), nil)
				var want = make([]float64, count)
				for i := range count {
					want[i] = op.reference(float64(inA[i]), float64(inB[i]))
				}
				checkAgainstFP64(t, fmt.Sprintf("%s %v", op.name, shape), got, want, op.tolerance)
				requireError(t, left.Close(), nil)
				requireError(t, right.Close(), nil)
				requireError(t, output.Close(), nil)
			}
		})
	}

	t.Run("scale", func(t *testing.T) {
		var scales = []float32{
			0, negZero, 1, -1, 0.125, float32(1 / math.Sqrt(64)), -3.5, 1e10, inf, -inf, nan,
		}
		for _, scale := range scales {
			var (
				count  = 4096
				input  = requireTensor(t, gpu, 4096)
				output = requireTensor(t, gpu, 4096)
				inData = make([]float32, count)
				got    = make([]float32, count)
				random = rand.New(rand.NewPCG(0x15c0, uint64(math.Float32bits(scale))))
			)
			for i := range count {
				inData[i] = float32((random.Float64()*2 - 1) * 20)
			}
			copy(inData, edgesA)
			requireError(t, input.Upload(inData), nil)

			var scope = requireScope(t, gpu)
			requireError(t, scope.Scale(input, output, scale), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)

			requireError(t, output.DownloadInto(got), nil)
			var want = make([]float64, count)
			for i := range count {
				var x = float64(inData[i])
				// Metal GPUs flush subnormal FP32 inputs to zero (FTZ); subnormal
				// preservation is explicitly unspecified by the gpu contract.
				if math.Abs(x) < 0x1p-126 && !math.IsNaN(x) {
					x = 0
				}
				want[i] = clampFP32Overflow(x * float64(scale))
			}
			checkAgainstFP64(t, fmt.Sprintf("scale(%v)", scale), got, want, 1e-6)
			requireError(t, input.Close(), nil)
			requireError(t, output.Close(), nil)
		}
	})

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("native resource leak: before=%+v after=%+v", initial, after)
	}
}

func Test_MetalBroadcastOpsAgainstFP64(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		inf     = float32(math.Inf(1))
		nan     = float32(math.NaN())
	)

	type broadcastCase struct {
		name  string
		left  []int64
		right []int64
		want  []int64
	}
	var cases = []broadcastCase{
		{name: "same_shape_2D", left: []int64{4, 17}, right: []int64{4, 17}, want: []int64{4, 17}},
		{name: "same_layout_diff_rank", left: []int64{2, 3, 5}, right: []int64{1, 2, 3, 5}, want: []int64{1, 2, 3, 5}},
		{name: "unit_1x1_and_1", left: []int64{1, 1}, right: []int64{1}, want: []int64{1, 1}},
		{name: "row_bias_2D_right", left: []int64{32, 257}, right: []int64{257}, want: []int64{32, 257}},
		{name: "row_bias_2D_1xN", left: []int64{17, 64}, right: []int64{1, 64}, want: []int64{17, 64}},
		{name: "row_bias_2D_left", left: []int64{257}, right: []int64{32, 257}, want: []int64{32, 257}},
		{name: "row_bias_small_width", left: []int64{128, 3}, right: []int64{3}, want: []int64{128, 3}},
		{name: "nd_last_axis_coalesced", left: []int64{2, 3, 5, 17}, right: []int64{1, 1, 1, 17}, want: []int64{2, 3, 5, 17}},
		{name: "scalar_right_1", left: []int64{3, 7, 5}, right: []int64{1}, want: []int64{3, 7, 5}},
		{name: "scalar_right_1x1x1", left: []int64{3, 7, 5}, right: []int64{1, 1, 1}, want: []int64{3, 7, 5}},
		{name: "scalar_left_1", left: []int64{1}, right: []int64{3, 7, 5}, want: []int64{3, 7, 5}},
		{name: "column_bias_right", left: []int64{19, 33}, right: []int64{19, 1}, want: []int64{19, 33}},
		{name: "column_bias_left", left: []int64{19, 1}, right: []int64{19, 33}, want: []int64{19, 33}},
		{name: "4D_multidirectional", left: []int64{2, 1, 4, 1}, right: []int64{1, 3, 1, 5}, want: []int64{2, 3, 4, 5}},
		{name: "6D_multidirectional", left: []int64{2, 1, 3, 1, 2, 1}, right: []int64{1, 2, 1, 2, 1, 3}, want: []int64{2, 2, 3, 2, 2, 3}},
	}

	type opSpec struct {
		name      string
		encode    func(*Scope, *Tensor, *Tensor, *Tensor) error
		reference func(float64, float64) float64
	}
	var ops = []opSpec{
		{name: "add", encode: (*Scope).AddBroadcast, reference: func(a, b float64) float64 { return a + b }},
		{name: "sub", encode: (*Scope).SubBroadcast, reference: func(a, b float64) float64 { return a - b }},
		{name: "mul", encode: (*Scope).MulBroadcast, reference: func(a, b float64) float64 { return a * b }},
		{name: "div", encode: (*Scope).DivBroadcast, reference: func(a, b float64) float64 { return a / b }},
	}

	for _, op := range ops {
		t.Run(op.name, func(t *testing.T) {
			for caseIdx, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					var (
						leftCount  = elementCount(tc.left)
						rightCount = elementCount(tc.right)
						outCount   = elementCount(tc.want)
						leftData   = make([]float32, leftCount)
						rightData  = make([]float32, rightCount)
						got        = make([]float32, outCount)
						random     = rand.New(rand.NewPCG(0x15b0+uint64(caseIdx), uint64(leftCount+rightCount)))
					)
					for i := range leftCount {
						leftData[i] = float32((random.Float64()*2 - 1) * 10)
					}
					for i := range rightCount {
						rightData[i] = float32((random.Float64()*2 - 1) * 10)
						if op.name == "div" && math.Abs(float64(rightData[i])) < 0.05 {
							rightData[i] = 0.5
						}
					}

					var (
						left   = requireTensor(t, gpu, tc.left...)
						right  = requireTensor(t, gpu, tc.right...)
						output = requireTensor(t, gpu, tc.want...)
						scope  = requireScope(t, gpu)
					)
					requireError(t, left.Upload(leftData), nil)
					requireError(t, right.Upload(rightData), nil)
					requireError(t, op.encode(scope, left, right, output), nil)
					requireError(t, scope.Submit(ctx), nil)
					requireError(t, scope.Wait(ctx), nil)
					requireError(t, scope.Close(), nil)

					requireError(t, output.DownloadInto(got), nil)
					var want = referenceBroadcastFP64(leftData, tc.left, rightData, tc.right, tc.want, op.reference)
					checkAgainstFP64(t, fmt.Sprintf("%s/%s", op.name, tc.name), got, want, 1e-6)
					requireError(t, left.Close(), nil)
					requireError(t, right.Close(), nil)
					requireError(t, output.Close(), nil)
				})
			}
		})
	}

	// Verify IEEE-754 special values across broadcast paths (0/0, 1/0, -1/0, Inf-Inf, 0*Inf, NaN).
	t.Run("special_values", func(t *testing.T) {
		var (
			leftVals  = []float32{0, 1, -1, inf, -inf, nan, 2, -3}
			rightVals = []float32{0, inf, -inf, nan}
			left      = requireTensor(t, gpu, 2, 4)
			right     = requireTensor(t, gpu, 4)
			out       = requireTensor(t, gpu, 2, 4)
			got       = make([]float32, 8)
		)
		requireError(t, left.Upload(leftVals), nil)
		requireError(t, right.Upload(rightVals), nil)
		for _, op := range ops {
			var scope = requireScope(t, gpu)
			requireError(t, op.encode(scope, left, right, out), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)

			requireError(t, out.DownloadInto(got), nil)
			var want = referenceBroadcastFP64(leftVals, []int64{2, 4}, rightVals, []int64{4}, []int64{2, 4}, op.reference)
			checkAgainstFP64(t, "special/"+op.name, got, want, 1e-6)
		}
		requireError(t, left.Close(), nil)
		requireError(t, right.Close(), nil)
		requireError(t, out.Close(), nil)
	})

	// Verify same input storage via Tensor.View is allowed as both inputs,
	// while destination aliasing via Tensor.View is rejected.
	t.Run("view_sharing_and_alias_rejection", func(t *testing.T) {
		var (
			base     = requireTensor(t, gpu, 4)
			colView  *Tensor
			rowView  *Tensor
			out      = requireTensor(t, gpu, 4, 4)
			outView  *Tensor
			err      error
			scope    = requireScope(t, gpu)
			baseData = []float32{1, 2, 3, 4}
			got      = make([]float32, 16)
		)
		requireError(t, base.Upload(baseData), nil)
		colView, err = base.View([]int64{4, 1})
		requireError(t, err, nil)
		rowView, err = base.View([]int64{1, 4})
		requireError(t, err, nil)
		outView, err = base.View([]int64{1, 4})
		requireError(t, err, nil)

		requireError(t, scope.AddBroadcast(colView, rowView, out), nil)
		requireError(t, scope.Submit(ctx), nil)
		requireError(t, scope.Wait(ctx), nil)
		requireError(t, scope.Close(), nil)

		requireError(t, out.DownloadInto(got), nil)
		var want = referenceBroadcastFP64(baseData, []int64{4, 1}, baseData, []int64{1, 4}, []int64{4, 4}, func(a, b float64) float64 { return a + b })
		checkAgainstFP64(t, "outer_sum_views", got, want, 0)

		var badScope = requireScope(t, gpu)
		requireError(t, badScope.AddBroadcast(rowView, rowView, outView), ErrInvalid)
		requireError(t, badScope.Close(), nil)

		for _, tensor := range []*Tensor{colView, rowView, outView, base, out} {
			requireError(t, tensor.Close(), nil)
		}
	})

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("native resource leak: before=%+v after=%+v", initial, after)
	}
}

func Test_MetalBroadcastConcurrencyStress(t *testing.T) {
	const (
		workers    = 8
		iterations = 12
		rows       = 16
		cols       = 128
	)
	var (
		gpu       = requireMetalDevice(t)
		ctx       = context.Background()
		native    = gpu.state.backend.(metalBackend).runtime
		initial   = native.ResourceSnapshot()
		sharedA   = requireTensor(t, gpu, rows, cols)
		sharedB   = requireTensor(t, gpu, cols)
		sharedCol = requireTensor(t, gpu, rows, 1)
		dataA     = attentionValues(rows*cols, 151, 1)
		dataB     = attentionValues(cols, 152, 1)
		dataCol   = attentionValues(rows, 153, 1)
	)
	for i := range dataB {
		if math.Abs(float64(dataB[i])) < 0.1 {
			dataB[i] = 0.5
		}
	}
	requireError(t, sharedA.Upload(dataA), nil)
	requireError(t, sharedB.Upload(dataB), nil)
	requireError(t, sharedCol.Upload(dataCol), nil)

	var (
		wantRow = referenceBroadcastFP64(dataA, []int64{rows, cols}, dataB, []int64{cols}, []int64{rows, cols}, func(a, b float64) float64 { return a + b })
		wantCol = referenceBroadcastFP64(dataA, []int64{rows, cols}, dataCol, []int64{rows, 1}, []int64{rows, cols}, func(a, b float64) float64 { return a - b })
		wantDiv = referenceBroadcastFP64(dataA, []int64{rows, cols}, dataB, []int64{cols}, []int64{rows, cols}, func(a, b float64) float64 { return a / b })
		errCh   = make(chan error, workers)
		wg      sync.WaitGroup
	)

	for w := range workers {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			var (
				out1 = requireTensor(t, gpu, rows, cols)
				out2 = requireTensor(t, gpu, rows, cols)
				out3 = requireTensor(t, gpu, rows, cols)
				out4 = requireTensor(t, gpu, rows, cols)
				got1 = make([]float32, rows*cols)
				got2 = make([]float32, rows*cols)
				got3 = make([]float32, rows*cols)
				got4 = make([]float32, rows*cols)
			)
			defer func() {
				_ = out1.Close()
				_ = out2.Close()
				_ = out3.Close()
				_ = out4.Close()
			}()
			for iter := range iterations {
				var scope, err = gpu.NewScope()
				if err != nil {
					errCh <- err
					return
				}
				if err = scope.AddBroadcast(sharedA, sharedB, out1); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.SubBroadcast(sharedA, sharedCol, out2); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.DivBroadcast(sharedA, sharedB, out3); err != nil {
					_ = scope.Close()
					errCh <- err
					return
				}
				if err = scope.Scale(out1, out4, 0.5); err != nil {
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
				if err = out1.DownloadInto(got1); err != nil {
					errCh <- err
					return
				}
				if err = out2.DownloadInto(got2); err != nil {
					errCh <- err
					return
				}
				if err = out3.DownloadInto(got3); err != nil {
					errCh <- err
					return
				}
				if err = out4.DownloadInto(got4); err != nil {
					errCh <- err
					return
				}
				for i := range got1 {
					if math.Abs(float64(got1[i])-wantRow[i]) > 1e-5 ||
						math.Abs(float64(got2[i])-wantCol[i]) > 1e-5 ||
						math.Abs(float64(got3[i])-wantDiv[i]) > 1e-5 ||
						math.Abs(float64(got4[i])-0.5*wantRow[i]) > 1e-5 {
						errCh <- fmt.Errorf("worker %d iter %d idx %d mismatch", worker, iter, i)
						return
					}
				}
			}
		}(w)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
	requireError(t, sharedA.Close(), nil)
	requireError(t, sharedB.Close(), nil)
	requireError(t, sharedCol.Close(), nil)

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("native resource leak after concurrency stress: before=%+v after=%+v", initial, after)
	}
}

func referenceBroadcastFP64(
	left []float32,
	leftShape []int64,
	right []float32,
	rightShape []int64,
	outShape []int64,
	op func(float64, float64) float64,
) (out []float64) {
	var (
		rank       = len(outShape)
		total      = elementCount(outShape)
		outStrides = make([]int, rank)
		lStrides   = make([]int, rank)
		rStrides   = make([]int, rank)
		stride     = 1
	)
	for i := rank - 1; i >= 0; i-- {
		outStrides[i] = stride
		stride *= int(outShape[i])
	}
	stride = 1
	for i := len(leftShape) - 1; i >= 0; i-- {
		var dst = rank - len(leftShape) + i
		if leftShape[i] == 1 {
			lStrides[dst] = 0
		} else {
			lStrides[dst] = stride
			stride *= int(leftShape[i])
		}
	}
	stride = 1
	for i := len(rightShape) - 1; i >= 0; i-- {
		var dst = rank - len(rightShape) + i
		if rightShape[i] == 1 {
			rStrides[dst] = 0
		} else {
			rStrides[dst] = stride
			stride *= int(rightShape[i])
		}
	}

	out = make([]float64, total)
	for idx := range total {
		var (
			rem  = idx
			lIdx = 0
			rIdx = 0
		)
		for d := range rank {
			var coord = rem / outStrides[d]
			rem -= coord * outStrides[d]
			lIdx += coord * lStrides[d]
			rIdx += coord * rStrides[d]
		}
		out[idx] = clampFP32Overflow(op(float64(left[lIdx]), float64(right[rIdx])))
	}
	return out
}

func clampFP32Overflow(v float64) (out float64) {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return v
	}
	if v > math.MaxFloat32 {
		return math.Inf(1)
	}
	if v < -math.MaxFloat32 {
		return math.Inf(-1)
	}
	return v
}

func elementCount(shape []int64) (count int) {
	count = 1
	for _, dim := range shape {
		count *= int(dim)
	}
	return count
}
