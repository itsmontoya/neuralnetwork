package gpu

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
)

func Test_ScopeArithmeticAndBroadcastValidation(t *testing.T) {
	var (
		backend       fakeBackend
		device        = newDevice(&backend, 256)
		other         = newDevice(&backend, 256)
		left          = requireTensor(t, device, 2, 3)
		right         = requireTensor(t, device, 2, 3)
		bias          = requireTensor(t, device, 3)
		badBias       = requireTensor(t, device, 4)
		uninitialized = requireTensor(t, device, 2, 3)
		destination   = requireTensor(t, device, 2, 3)
		flat          = requireTensor(t, device, 6)
		rank7         = requireTensor(t, device, 1, 1, 1, 1, 1, 1, 1)
		foreign       = requireTensor(t, other, 2, 3)
		closed        = requireTensor(t, device, 2, 3)
		f16Tensor     = requireTensorWithDType(t, device, Float16, 2, 3)
		alias         = *left
		view, err     = left.View([]int64{2, 3})
		scope         = requireScope(t, device)
	)
	requireError(t, err, nil)
	requireError(t, left.Upload([]float32{1, 2, 3, 4, 5, 6}), nil)
	requireError(t, right.Upload([]float32{6, 5, 4, 3, 2, 1}), nil)
	requireError(t, bias.Upload([]float32{1, 2, 3}), nil)
	requireError(t, badBias.Upload([]float32{1, 2, 3, 4}), nil)
	requireError(t, flat.Upload([]float32{1, 2, 3, 4, 5, 6}), nil)
	requireError(t, rank7.Upload([]float32{1}), nil)
	requireError(t, foreign.Upload([]float32{1, 2, 3, 4, 5, 6}), nil)
	requireError(t, f16Tensor.UploadFloat32As([]float32{1, 2, 3, 4, 5, 6}), nil)
	requireError(t, closed.Close(), nil)

	var tests = []struct {
		name string
		run  func() error
		want error
	}{
		// Backward compatibility: Add and Mul still reject mismatched shapes.
		{name: "add mismatched broadcast shape", run: func() error { return scope.Add(left, bias, destination) }, want: ErrInvalid},
		{name: "mul mismatched broadcast shape", run: func() error { return scope.Mul(left, bias, destination) }, want: ErrInvalid},

		// Same-shape Sub and Div.
		{name: "sub missing left", run: func() error { return scope.Sub(nil, right, destination) }, want: ErrInvalid},
		{name: "sub missing right", run: func() error { return scope.Sub(left, nil, destination) }, want: ErrInvalid},
		{name: "sub missing destination", run: func() error { return scope.Sub(left, right, nil) }, want: ErrInvalid},
		{name: "sub mismatched shape", run: func() error { return scope.Sub(left, bias, destination) }, want: ErrInvalid},
		{name: "sub non-float32", run: func() error { return scope.Sub(f16Tensor, right, destination) }, want: ErrInvalid},
		{name: "sub aliased view", run: func() error { return scope.Sub(left, right, view) }, want: ErrInvalid},
		{name: "sub uninitialized", run: func() error { return scope.Sub(uninitialized, right, destination) }, want: ErrUninitialized},
		{name: "sub closed", run: func() error { return scope.Sub(closed, right, destination) }, want: ErrClosed},

		{name: "div missing left", run: func() error { return scope.Div(nil, right, destination) }, want: ErrInvalid},
		{name: "div mismatched shape", run: func() error { return scope.Div(left, bias, destination) }, want: ErrInvalid},
		{name: "div non-float32", run: func() error { return scope.Div(left, f16Tensor, destination) }, want: ErrInvalid},
		{name: "div aliased view", run: func() error { return scope.Div(left, right, view) }, want: ErrInvalid},
		{name: "div uninitialized", run: func() error { return scope.Div(left, uninitialized, destination) }, want: ErrUninitialized},
		{name: "div closed", run: func() error { return scope.Div(left, closed, destination) }, want: ErrClosed},

		// Scale.
		{name: "scale missing input", run: func() error { return scope.Scale(nil, destination, 2) }, want: ErrInvalid},
		{name: "scale zero handle", run: func() error { return scope.Scale(&Tensor{}, destination, 2) }, want: ErrInvalid},
		{name: "scale missing destination", run: func() error { return scope.Scale(left, nil, 2) }, want: ErrInvalid},
		{name: "scale different shape", run: func() error { return scope.Scale(left, flat, 2) }, want: ErrInvalid},
		{name: "scale non-float32 input", run: func() error { return scope.Scale(f16Tensor, destination, 2) }, want: ErrInvalid},
		{name: "scale non-float32 destination", run: func() error { return scope.Scale(left, f16Tensor, 2) }, want: ErrInvalid},
		{name: "scale foreign", run: func() error { return scope.Scale(foreign, destination, 2) }, want: ErrInvalid},
		{name: "scale aliased handle", run: func() error { return scope.Scale(left, &alias, 2) }, want: ErrInvalid},
		{name: "scale aliased view", run: func() error { return scope.Scale(left, view, 2) }, want: ErrInvalid},
		{name: "scale uninitialized", run: func() error { return scope.Scale(uninitialized, destination, 2) }, want: ErrUninitialized},
		{name: "scale closed", run: func() error { return scope.Scale(closed, destination, 2) }, want: ErrClosed},

		// Broadcast binary ops.
		{name: "add broadcast missing left", run: func() error { return scope.AddBroadcast(nil, bias, destination) }, want: ErrInvalid},
		{name: "add broadcast missing right", run: func() error { return scope.AddBroadcast(left, nil, destination) }, want: ErrInvalid},
		{name: "add broadcast missing destination", run: func() error { return scope.AddBroadcast(left, bias, nil) }, want: ErrInvalid},
		{name: "add broadcast incompatible shape", run: func() error { return scope.AddBroadcast(left, badBias, destination) }, want: ErrInvalid},
		{name: "add broadcast wrong destination shape", run: func() error { return scope.AddBroadcast(left, bias, flat) }, want: ErrInvalid},
		{name: "add broadcast rank above 6", run: func() error { return scope.AddBroadcast(rank7, bias, destination) }, want: ErrInvalid},
		{name: "add broadcast non-float32", run: func() error { return scope.AddBroadcast(f16Tensor, bias, destination) }, want: ErrInvalid},
		{name: "add broadcast foreign", run: func() error { return scope.AddBroadcast(foreign, bias, destination) }, want: ErrInvalid},
		{name: "add broadcast aliased handle", run: func() error { return scope.AddBroadcast(left, bias, &alias) }, want: ErrInvalid},
		{name: "add broadcast aliased view", run: func() error { return scope.AddBroadcast(left, bias, view) }, want: ErrInvalid},
		{name: "add broadcast uninitialized", run: func() error { return scope.AddBroadcast(uninitialized, bias, destination) }, want: ErrUninitialized},
		{name: "add broadcast closed", run: func() error { return scope.AddBroadcast(closed, bias, destination) }, want: ErrClosed},

		{name: "sub broadcast incompatible shape", run: func() error { return scope.SubBroadcast(left, badBias, destination) }, want: ErrInvalid},
		{name: "mul broadcast wrong destination shape", run: func() error { return scope.MulBroadcast(left, bias, flat) }, want: ErrInvalid},
		{name: "div broadcast aliased view", run: func() error { return scope.DivBroadcast(left, bias, view) }, want: ErrInvalid},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requireError(t, tt.run(), tt.want)
		})
	}
	requireError(t, scope.Close(), nil)
	requireError(t, view.Close(), nil)
	for _, tensor := range []*Tensor{left, right, bias, badBias, uninitialized, destination, flat, rank7, foreign, f16Tensor} {
		requireError(t, tensor.Close(), nil)
	}
	requireError(t, device.Close(), nil)
	requireError(t, other.Close(), nil)
}

func Test_BuildBroadcastSpecAndFastPathRouting(t *testing.T) {
	var tests = []struct {
		name           string
		left           []int64
		right          []int64
		wantOut        []int64
		wantContiguous bool
		wantRank       uint32
		wantDims       [6]uint32
		wantLeft       [6]uint32
		wantRight      [6]uint32
	}{
		{
			name:           "identical shapes",
			left:           []int64{4, 8},
			right:          []int64{4, 8},
			wantOut:        []int64{4, 8},
			wantContiguous: true,
		},
		{
			name:           "leading unit dim same layout",
			left:           []int64{2, 3, 4},
			right:          []int64{1, 2, 3, 4},
			wantOut:        []int64{1, 2, 3, 4},
			wantContiguous: true,
		},
		{
			name:           "both unit tensors",
			left:           []int64{1, 1},
			right:          []int64{1},
			wantOut:        []int64{1, 1},
			wantContiguous: true,
		},
		{
			name:      "2D row bias right",
			left:      []int64{32, 4096},
			right:     []int64{4096},
			wantOut:   []int64{32, 4096},
			wantRank:  2,
			wantDims:  [6]uint32{32, 4096},
			wantLeft:  [6]uint32{4096, 1},
			wantRight: [6]uint32{0, 1},
		},
		{
			name:      "4D last-axis bias coalesced to 2D",
			left:      []int64{2, 4, 8, 4096},
			right:     []int64{1, 1, 1, 4096},
			wantOut:   []int64{2, 4, 8, 4096},
			wantRank:  2,
			wantDims:  [6]uint32{64, 4096},
			wantLeft:  [6]uint32{4096, 1},
			wantRight: [6]uint32{0, 1},
		},
		{
			name:      "scalar right broadcast",
			left:      []int64{3, 4, 5},
			right:     []int64{1},
			wantOut:   []int64{3, 4, 5},
			wantRank:  1,
			wantDims:  [6]uint32{60},
			wantLeft:  [6]uint32{1},
			wantRight: [6]uint32{0},
		},
		{
			name:      "scalar left broadcast",
			left:      []int64{1, 1},
			right:     []int64{3, 4, 5},
			wantOut:   []int64{3, 4, 5},
			wantRank:  1,
			wantDims:  [6]uint32{60},
			wantLeft:  [6]uint32{0},
			wantRight: [6]uint32{1},
		},
		{
			name:      "column bias",
			left:      []int64{4, 3},
			right:     []int64{4, 1},
			wantOut:   []int64{4, 3},
			wantRank:  2,
			wantDims:  [6]uint32{4, 3},
			wantLeft:  [6]uint32{3, 1},
			wantRight: [6]uint32{1, 0},
		},
		{
			name:      "4D multidirectional",
			left:      []int64{2, 1, 4, 1},
			right:     []int64{1, 3, 1, 5},
			wantOut:   []int64{2, 3, 4, 5},
			wantRank:  4,
			wantDims:  [6]uint32{2, 3, 4, 5},
			wantLeft:  [6]uint32{4, 0, 1, 0},
			wantRight: [6]uint32{0, 5, 0, 1},
		},
		{
			name:      "6D multidirectional",
			left:      []int64{2, 1, 3, 1, 2, 1},
			right:     []int64{1, 2, 1, 2, 1, 3},
			wantOut:   []int64{2, 2, 3, 2, 2, 3},
			wantRank:  6,
			wantDims:  [6]uint32{2, 2, 3, 2, 2, 3},
			wantLeft:  [6]uint32{6, 0, 2, 0, 1, 0},
			wantRight: [6]uint32{0, 6, 0, 3, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out, outRank, spec, contiguous, err = buildBroadcastSpec(tt.left, tt.right)
			requireError(t, err, nil)
			if !slices.Equal(out[:outRank], tt.wantOut) {
				t.Fatalf("outShape = %v, want %v", out[:outRank], tt.wantOut)
			}
			if contiguous != tt.wantContiguous {
				t.Fatalf("contiguous = %t, want %t", contiguous, tt.wantContiguous)
			}
			if contiguous {
				return
			}
			if spec.Rank != tt.wantRank || spec.Dims != tt.wantDims ||
				spec.LeftStrides != tt.wantLeft || spec.RightStrides != tt.wantRight {
				t.Fatalf("spec = %+v, want rank=%d dims=%v left=%v right=%v",
					spec, tt.wantRank, tt.wantDims, tt.wantLeft, tt.wantRight)
			}
		})
	}
}

func Test_ScopeArithmeticAndBroadcastDispatchAndLimits(t *testing.T) {
	var (
		backend fakeBackend
		device  = newDevice(&backend, 1<<40)
		a       = requireTensor(t, device, 2, 4)
		b       = requireTensor(t, device, 2, 4)
		bias    = requireTensor(t, device, 4)
		dst     = requireTensor(t, device, 2, 4)
		huge    = requireTensor(t, device, 1<<16, 1<<16)
		hugeDst = requireTensor(t, device, 1<<16, 1<<16)
		scope   = requireScope(t, device)
		ctx     = context.Background()
	)
	requireError(t, a.Upload([]float32{1, 2, 3, 4, 5, 6, 7, 8}), nil)
	requireError(t, b.Upload([]float32{2, 2, 2, 2, 2, 2, 2, 2}), nil)
	requireError(t, bias.Upload([]float32{10, 20, 30, 40}), nil)

	// Exceeding uint32 element count fails validation.
	requireError(t, scope.Sub(huge, huge, hugeDst), ErrInvalid)
	requireError(t, scope.Div(huge, huge, hugeDst), ErrInvalid)
	requireError(t, scope.Scale(huge, hugeDst, 2), ErrInvalid)
	requireError(t, scope.AddBroadcast(huge, bias, hugeDst), ErrInvalid)

	// Identical-shape AddBroadcast routes to contiguous elementwise fast path.
	requireError(t, scope.Sub(a, b, dst), nil)
	requireError(t, scope.Div(a, b, dst), nil)
	requireError(t, scope.Scale(a, dst, float32(math.Inf(1))), nil)
	requireError(t, scope.AddBroadcast(a, b, dst), nil)
	requireError(t, scope.AddBroadcast(a, bias, dst), nil)
	requireError(t, scope.SubBroadcast(a, bias, dst), nil)
	requireError(t, scope.MulBroadcast(a, bias, dst), nil)
	requireError(t, scope.DivBroadcast(a, bias, dst), nil)
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)

	if got := backend.elementwise.Load(); got != 3 {
		t.Fatalf("elementwise dispatches = %d, want 3 (Sub, Div, identical-shape AddBroadcast)", got)
	}
	if got := backend.scales.Load(); got != 1 {
		t.Fatalf("scale dispatches = %d, want 1", got)
	}
	if got := backend.broadcasts.Load(); got != 4 {
		t.Fatalf("broadcast dispatches = %d, want 4", got)
	}

	// Backend encode error poisons scope.
	var (
		errBoom  = errors.New("encode boom")
		badScope = requireScope(t, device)
	)
	badScope.state.commands.(*fakeCommandBuffer).encodeErr = errBoom
	requireError(t, badScope.AddBroadcast(a, bias, dst), errBoom)
	requireError(t, badScope.Submit(ctx), ErrInvalid)
	requireError(t, badScope.Close(), nil)

	for _, tensor := range []*Tensor{a, b, bias, dst, huge, hugeDst} {
		requireError(t, tensor.Close(), nil)
	}
	requireError(t, device.Close(), nil)
}
