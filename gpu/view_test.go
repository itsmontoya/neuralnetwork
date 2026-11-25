package gpu

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
)

func Test_TensorViewShapesAndMetadata(t *testing.T) {
	var (
		b      fakeBackend
		d      = newDevice(&b, 4096)
		dtypes = []DType{Float32, Float16, BFloat16}
	)
	defer func() {
		requireError(t, d.Close(), nil)
		if b.liveBuffers.Load() != 0 || b.liveScopes.Load() != 0 {
			t.Fatalf("leaked resources: buffers=%d scopes=%d", b.liveBuffers.Load(), b.liveScopes.Load())
		}
	}()

	for _, dtype := range dtypes {
		t.Run(dtype.String(), func(t *testing.T) {
			var (
				base  = requireTensorWithDType(t, d, dtype, 24)
				shape = []int64{2, 3, 4}
			)
			if got := b.liveBuffers.Load(); got != 1 {
				t.Fatalf("live buffers after base alloc = %d, want 1", got)
			}

			var v3d, err = base.View(shape)
			requireError(t, err, nil)
			if got := b.liveBuffers.Load(); got != 1 {
				t.Fatalf("live buffers after 3D view = %d, want 1", got)
			}

			// Mutating caller input or returned Shape() must not affect the view.
			shape[0] = 99
			v3d.Shape()[2] = 77
			if !slices.Equal(v3d.Shape(), []int64{2, 3, 4}) {
				t.Fatalf("3D view shape mutated: %v", v3d.Shape())
			}
			if !slices.Equal(base.Shape(), []int64{24}) {
				t.Fatalf("base shape changed: %v", base.Shape())
			}
			if v3d.DType() != dtype || v3d.Bytes() != base.Bytes() {
				t.Fatalf("3D view metadata = (%v, %d), want (%v, %d)", v3d.DType(), v3d.Bytes(), dtype, base.Bytes())
			}

			// Chained views across 3D -> 4D -> 2D -> 1D.
			var v4d *Tensor
			if v4d, err = v3d.View([]int64{2, 1, 3, 4}); err != nil {
				t.Fatalf("4D chained view failed: %v", err)
			}
			var v2d *Tensor
			if v2d, err = v4d.View([]int64{6, 4}); err != nil {
				t.Fatalf("2D chained view failed: %v", err)
			}
			var v1d *Tensor
			if v1d, err = v2d.View([]int64{24}); err != nil {
				t.Fatalf("1D chained view failed: %v", err)
			}
			if got := b.liveBuffers.Load(); got != 1 {
				t.Fatalf("live buffers after chained views = %d, want 1", got)
			}
			if !slices.Equal(v4d.Shape(), []int64{2, 1, 3, 4}) ||
				!slices.Equal(v2d.Shape(), []int64{6, 4}) ||
				!slices.Equal(v1d.Shape(), []int64{24}) {
				t.Fatalf("unexpected chained shapes: 4D=%v 2D=%v 1D=%v", v4d.Shape(), v2d.Shape(), v1d.Shape())
			}

			// Metadata remains readable after Close; Close is idempotent.
			var wantBytes = base.Bytes()
			for _, handle := range []*Tensor{v1d, v2d, v4d, v3d, base} {
				requireError(t, handle.Close(), nil)
				requireError(t, handle.Close(), nil)
				if handle.DType() != dtype || handle.Bytes() != wantBytes {
					t.Fatalf("closed handle metadata = (%v, %d), want (%v, %d)", handle.DType(), handle.Bytes(), dtype, wantBytes)
				}
			}
			if got := b.liveBuffers.Load(); got != 0 {
				t.Fatalf("live buffers after closing all handles = %d, want 0", got)
			}
		})
	}
}

func Test_TensorViewValidationErrors(t *testing.T) {
	var (
		b fakeBackend
		d = newDevice(&b, 4096)
	)
	defer func() {
		requireError(t, d.Close(), nil)
		if b.liveBuffers.Load() != 0 {
			t.Fatalf("leaked buffers: %d", b.liveBuffers.Load())
		}
	}()

	var nilTensor *Tensor
	var _, err = nilTensor.View([]int64{1})
	requireError(t, err, ErrClosed)

	var zeroTensor Tensor
	_, err = zeroTensor.View([]int64{1})
	requireError(t, err, ErrClosed)

	var f32 = requireTensor(t, d, 2, 6)
	type testcase struct {
		name  string
		shape []int64
	}
	var cases = []testcase{
		{name: "rank-0 empty", shape: []int64{}},
		{name: "rank-0 nil", shape: nil},
		{name: "zero dim", shape: []int64{12, 0}},
		{name: "negative dim", shape: []int64{-3, 4}},
		{name: "mismatched smaller", shape: []int64{2, 5}},
		{name: "mismatched larger", shape: []int64{3, 5}},
		{name: "overflow", shape: []int64{math.MaxInt64, 2}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var view *Tensor
			view, err = f32.View(tc.shape)
			if view != nil || !errors.Is(err, ErrInvalid) {
				t.Fatalf("View(%v) = (%v, %v), want (nil, ErrInvalid)", tc.shape, view, err)
			}
		})
	}

	// Quantized tensors must reject View with ErrInvalid.
	for _, qdtype := range []DType{Q8_0, Q4_0} {
		t.Run("quantized_"+qdtype.String(), func(t *testing.T) {
			var qt = requireTensorWithDType(t, d, qdtype, 2, 32)
			defer func() { requireError(t, qt.Close(), nil) }()
			var view *Tensor
			view, err = qt.View([]int64{2, 32})
			if view != nil || !errors.Is(err, ErrInvalid) {
				t.Fatalf("%s View = (%v, %v), want (nil, ErrInvalid)", qdtype, view, err)
			}
		})
	}

	// Closed tensor must reject View with ErrClosed even if shape is invalid.
	requireError(t, f32.Close(), nil)
	var closedView *Tensor
	closedView, err = f32.View([]int64{12})
	if closedView != nil || !errors.Is(err, ErrClosed) {
		t.Fatalf("closed View = (%v, %v), want (nil, ErrClosed)", closedView, err)
	}
	closedView, err = f32.View([]int64{})
	if closedView != nil || !errors.Is(err, ErrClosed) {
		t.Fatalf("closed View invalid shape = (%v, %v), want (nil, ErrClosed)", closedView, err)
	}
}

func Test_TensorViewAliasRejection(t *testing.T) {
	var (
		b      fakeBackend
		d      = newDevice(&b, 4096)
		base   = requireTensor(t, d, 2, 2)
		weight = requireTensor(t, d, 2)
		other  = requireTensor(t, d, 2, 2)
		cos    = requireTensor(t, d, 2, 1)
		sin    = requireTensor(t, d, 2, 1)
	)
	requireError(t, base.Upload([]float32{1, 2, 3, 4}), nil)
	requireError(t, weight.Upload([]float32{1, 1}), nil)
	requireError(t, other.Upload([]float32{5, 6, 7, 8}), nil)
	requireError(t, cos.Upload([]float32{1, 1}), nil)
	requireError(t, sin.Upload([]float32{0, 0}), nil)

	var (
		v1, err1 = base.View([]int64{2, 2})
		v2, err2 = base.View([]int64{2, 2})
	)
	requireError(t, err1, nil)
	requireError(t, err2, nil)
	defer func() {
		for _, handle := range []*Tensor{v1, v2, base, weight, other, cos, sin} {
			requireError(t, handle.Close(), nil)
		}
		requireError(t, d.Close(), nil)
		if b.liveBuffers.Load() != 0 || b.liveScopes.Load() != 0 {
			t.Fatalf("leaked resources: buffers=%d scopes=%d", b.liveBuffers.Load(), b.liveScopes.Load())
		}
	}()

	type testcase struct {
		name   string
		encode func(s *Scope) error
	}
	var cases = []testcase{
		{name: "Copy", encode: func(s *Scope) error { return s.Copy(v1, v2) }},
		{name: "Add left alias", encode: func(s *Scope) error { return s.Add(v1, other, v2) }},
		{name: "Add right alias", encode: func(s *Scope) error { return s.Add(other, v1, v2) }},
		{name: "Mul alias", encode: func(s *Scope) error { return s.Mul(v1, other, v2) }},
		{name: "SiLU alias", encode: func(s *Scope) error { return s.SiLU(v1, v2) }},
		{name: "SwiGLU alias", encode: func(s *Scope) error { return s.SwiGLU(v1, other, v2) }},
		{name: "MatMul left alias", encode: func(s *Scope) error { return s.MatMul(v1, other, v2) }},
		{name: "MatMul right alias", encode: func(s *Scope) error { return s.MatMul(other, v1, v2) }},
		{name: "MatMulRightTranspose alias", encode: func(s *Scope) error { return s.MatMulRightTranspose(v1, other, v2) }},
		{name: "RMSNorm alias", encode: func(s *Scope) error { return s.RMSNorm(v1, weight, v2, 1e-5) }},
		{name: "GatherRows alias", encode: func(s *Scope) error { return s.GatherRows(v1, []int64{0, 1}, v2) }},
		{name: "RotatePairs alias", encode: func(s *Scope) error { return s.RotatePairs(v1, cos, sin, v2, 2, 0, 0) }},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var scope = requireScope(t, d)
			defer func() { requireError(t, scope.Close(), nil) }()
			requireError(t, tc.encode(scope), ErrInvalid)
		})
	}

	// Reading sibling views as both inputs with a distinct destination is valid.
	var scope = requireScope(t, d)
	requireError(t, scope.Add(v1, v2, other), nil)
	if got := base.state.storage.readers; got != 1 {
		t.Fatalf("storage readers for same-scope sibling reads = %d, want 1", got)
	}
	requireError(t, scope.Submit(context.Background()), nil)
	requireError(t, scope.Wait(context.Background()), nil)
	requireError(t, scope.Close(), nil)
	if got := base.state.storage.readers; got != 0 {
		t.Fatalf("storage readers after close = %d, want 0", got)
	}
}

func Test_TensorViewLeaseLifecycleAndDeviceClose(t *testing.T) {
	var (
		b      fakeBackend
		d      = newDevice(&b, 4096)
		ctx    = context.Background()
		parent = requireTensor(t, d, 4)
		dstA   = requireTensor(t, d, 2, 2)
		dstB   = requireTensor(t, d, 2, 2)
	)
	var v1, err = parent.View([]int64{2, 2})
	requireError(t, err, nil)
	var v2 *Tensor
	v2, err = parent.View([]int64{2, 2})
	requireError(t, err, nil)

	// Uninitialized storage is unreadable through any view until written.
	requireError(t, v1.DownloadInto(make([]float32, 4)), ErrUninitialized)
	requireError(t, parent.Upload([]float32{1, 2, 3, 4}), nil)
	var got [4]float32
	requireError(t, v1.DownloadInto(got[:]), nil)
	if got != [4]float32{1, 2, 3, 4} {
		t.Fatalf("view download = %v, want [1 2 3 4]", got)
	}

	// Concurrent read-sharing via views; competing write and host transfer rejected.
	var (
		scopeA = requireScope(t, d)
		scopeB = requireScope(t, d)
		scopeW = requireScope(t, d)
	)
	requireError(t, scopeA.Copy(v1, dstA), nil)
	requireError(t, scopeB.Copy(v2, dstB), nil)
	if readers := parent.state.storage.readers; readers != 2 {
		t.Fatalf("storage readers = %d, want 2", readers)
	}
	requireError(t, scopeW.Fill(parent, 9), ErrBusy)
	requireError(t, scopeW.Fill(v1, 9), ErrBusy)
	requireError(t, parent.Upload([]float32{0, 0, 0, 0}), ErrBusy)
	requireError(t, v2.DownloadInto(got[:]), ErrBusy)
	requireError(t, scopeW.Close(), nil)

	// Close parent while views remain open and leased.
	requireError(t, parent.Close(), nil)
	requireError(t, parent.Upload([]float32{0, 0, 0, 0}), ErrClosed)
	requireError(t, d.Close(), ErrBusy)

	requireError(t, scopeA.Submit(ctx), nil)
	requireError(t, scopeB.Submit(ctx), nil)
	requireError(t, scopeA.Wait(ctx), nil)
	requireError(t, scopeB.Wait(ctx), nil)
	requireError(t, scopeA.Close(), nil)
	requireError(t, scopeB.Close(), nil)

	// Parent is closed, but sibling view v1 remains usable in a new Scope.
	var (
		scopeC = requireScope(t, d)
		out    = requireTensor(t, d, 2, 2)
	)
	requireError(t, scopeC.Fill(v1, 7), nil)
	// Creating a view during an active recording scope is allowed and shares write visibility.
	var v3 *Tensor
	v3, err = v1.View([]int64{2, 2})
	requireError(t, err, nil)
	requireError(t, scopeC.Copy(v3, out), nil)
	requireError(t, scopeC.Submit(ctx), nil)
	requireError(t, scopeC.Wait(ctx), nil)
	requireError(t, scopeC.Close(), nil)

	requireError(t, out.DownloadInto(got[:]), nil)
	if got != [4]float32{7, 7, 7, 7} {
		t.Fatalf("scope copy after view write = %v, want [7 7 7 7]", got)
	}

	// Device.Close returns ErrBusy until every view is closed.
	requireError(t, dstA.Close(), nil)
	requireError(t, dstB.Close(), nil)
	requireError(t, out.Close(), nil)
	requireError(t, v1.Close(), nil)
	requireError(t, v2.Close(), nil)
	requireError(t, d.Close(), ErrBusy)
	if b.liveBuffers.Load() != 1 {
		t.Fatalf("live buffers with one view open = %d, want 1", b.liveBuffers.Load())
	}

	requireError(t, v3.Close(), nil)
	if b.liveBuffers.Load() != 0 {
		t.Fatalf("live buffers after last view closed = %d, want 0", b.liveBuffers.Load())
	}
	requireError(t, d.Close(), nil)
}

func Test_TensorViewCloseDuringHostTransfer(t *testing.T) {
	var (
		b      fakeBackend
		d      = newDevice(&b, 128)
		parent = requireTensor(t, d, 2)
		view   *Tensor
		err    error
		gate   = make(chan struct{})
		result = make(chan error, 1)
	)
	view, err = parent.View([]int64{1, 2})
	requireError(t, err, nil)
	var buffer = parent.state.storage.buffer.(*fakeAllocation)
	buffer.uploadStarted = make(chan struct{})
	buffer.uploadGate = gate
	var unblock = unblockAtCleanup(t, gate)

	go func() { result <- parent.Upload([]float32{3, 4}) }()
	waitSignal(t, buffer.uploadStarted)

	requireError(t, view.DownloadInto(make([]float32, 2)), ErrBusy)
	requireError(t, view.Upload([]float32{1, 2}), ErrBusy)
	requireError(t, parent.Close(), nil)
	requireError(t, view.Close(), nil)
	// Buffer and device child remain live until the active host transfer finishes.
	requireError(t, d.Close(), ErrBusy)
	if got := b.liveBuffers.Load(); got != 1 {
		t.Fatalf("live buffers during blocked transfer = %d, want 1", got)
	}

	unblock()
	requireError(t, <-result, nil)
	if got := b.liveBuffers.Load(); got != 0 {
		t.Fatalf("live buffers after blocked transfer = %d, want 0", got)
	}
	requireError(t, d.Close(), nil)
}
