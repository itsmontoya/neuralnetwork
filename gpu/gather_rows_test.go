package gpu

import (
	"context"
	"errors"
	"math"
	"testing"
)

func Test_GatherRowsValidation(t *testing.T) {
	var (
		backend       fakeBackend
		device        = newDevice(&backend, math.MaxUint64)
		other         = newDevice(&backend, math.MaxUint64)
		source        = requireTensor(t, device, 3, 2)
		target        = requireTensor(t, device, 2, 2)
		uninitialized = requireTensor(t, device, 3, 2)
		rankOne       = requireTensor(t, device, 4)
		wrongWidth    = requireTensor(t, device, 2, 3)
		foreign       = requireTensor(t, other, 2, 2)
		closed        = requireTensor(t, device, 2, 2)
		alias         = *source
	)
	requireError(t, source.Upload([]float32{1, 2, 3, 4, 5, 6}), nil)
	requireError(t, closed.Close(), nil)

	var cases = []struct {
		name   string
		source *Tensor
		rows   []int64
		target *Tensor
		want   error
	}{
		{name: "nil source", rows: []int64{0, 1}, target: target, want: ErrInvalid},
		{name: "nil destination", source: source, rows: []int64{0, 1}, want: ErrInvalid},
		{name: "rank-one source", source: rankOne, rows: []int64{0, 1}, target: target, want: ErrInvalid},
		{name: "rank-one destination", source: source, rows: []int64{0, 1}, target: rankOne, want: ErrInvalid},
		{name: "destination width", source: source, rows: []int64{0, 1}, target: wrongWidth, want: ErrInvalid},
		{name: "destination rows", source: source, rows: []int64{0}, target: target, want: ErrInvalid},
		{name: "empty indices", source: source, target: target, want: ErrInvalid},
		{name: "negative index", source: source, rows: []int64{0, -1}, target: target, want: ErrInvalid},
		{name: "one past end", source: source, rows: []int64{0, 3}, target: target, want: ErrInvalid},
		{name: "index would overflow a byte offset", source: source, rows: []int64{0, math.MaxInt64}, target: target, want: ErrInvalid},
		{name: "copied handle aliases source", source: source, rows: []int64{2, 0, 1}, target: &alias, want: ErrInvalid},
		{name: "foreign source", source: foreign, rows: []int64{0, 1}, target: target, want: ErrInvalid},
		{name: "foreign destination", source: source, rows: []int64{0, 1}, target: foreign, want: ErrInvalid},
		{name: "closed destination", source: source, rows: []int64{0, 1}, target: closed, want: ErrClosed},
		{name: "uninitialized source", source: uninitialized, rows: []int64{0, 1}, target: target, want: ErrUninitialized},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var scope = requireScope(t, device)
			requireError(t, scope.GatherRows(tc.source, tc.rows, tc.target), tc.want)
			requireError(t, scope.GatherRows(source, []int64{0, 2}, target), nil)
			requireError(t, scope.Close(), nil)
		})
	}
	if backend.uploads.Load() != 1 || backend.downloads.Load() != 0 || backend.submissions.Load() != 0 {
		t.Fatal("recording and discarding gathers performed a host transfer or submission")
	}
}

func Test_GatherRowsDiscardAndRetry(t *testing.T) {
	var (
		backend         fakeBackend
		device          = newDevice(&backend, 1024)
		source          = requireTensor(t, device, 3, 2)
		target          = requireTensor(t, device, 2, 2)
		ctx             = context.Background()
		stopped, cancel = context.WithCancel(ctx)
		fault           = errors.New("injected row gather encode failure")
	)
	cancel()
	requireError(t, source.Upload([]float32{1, 2, 3, 4, 5, 6}), nil)
	var cases = []struct {
		name        string
		context     context.Context
		enableFault bool
		wantEncode  error
		wantSubmit  error
	}{
		{name: "validation", context: ctx, wantEncode: ErrInvalid, wantSubmit: ErrInvalid},
		{name: "backend encode", context: ctx, enableFault: true, wantEncode: fault, wantSubmit: ErrInvalid},
		{name: "cancel before submit", context: stopped, wantSubmit: context.Canceled},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var (
				scope   = requireScope(t, device)
				output  [4]float32
				indices = []int64{2, 1}
			)
			requireError(t, target.Upload([]float32{9, 9, 9, 9}), nil)
			requireError(t, scope.GatherRows(source, indices, target), nil)
			if tc.wantEncode == ErrInvalid {
				indices[1] = 3
			}
			if tc.enableFault {
				scope.state.commands.(*fakeCommandBuffer).encodeErr = fault
			}
			requireError(t, scope.GatherRows(source, indices, target), tc.wantEncode)
			requireError(t, scope.Submit(tc.context), tc.wantSubmit)
			requireError(t, scope.Close(), nil)
			requireError(t, target.DownloadInto(output[:]), nil)
			if output != [4]float32{9, 9, 9, 9} {
				t.Fatalf("discard changed previous destination to %v", output)
			}

			var retry = requireScope(t, device)
			indices[0], indices[1] = 2, 0
			requireError(t, retry.GatherRows(source, indices, target), nil)
			indices[0], indices[1] = 1, 1
			requireError(t, retry.Submit(ctx), nil)
			requireError(t, retry.Wait(ctx), nil)
			requireError(t, retry.Close(), nil)
			requireError(t, target.DownloadInto(output[:]), nil)
			if output != [4]float32{5, 6, 1, 2} {
				t.Fatalf("retry copied %v", output)
			}
		})
	}
	if backend.submissions.Load() != int32(len(cases)) || backend.liveScopes.Load() != 0 {
		t.Fatalf("submissions = %d, live scopes = %d", backend.submissions.Load(), backend.liveScopes.Load())
	}
}

func Test_GatherRowsReducedPrecisionFake(t *testing.T) {
	var (
		backend fakeBackend
		device  = newDevice(&backend, 4096)
		ctx     = context.Background()
		f16Src  = requireTensorWithDType(t, device, Float16, 3, 2)
		f16Dst  = requireTensorWithDType(t, device, Float16, 2, 2)
		bfSrc   = requireTensorWithDType(t, device, BFloat16, 3, 2)
		bfDst   = requireTensorWithDType(t, device, BFloat16, 2, 2)
		f32Dst  = requireTensorWithDType(t, device, Float32, 2, 2)
		f32Dst2 = requireTensorWithDType(t, device, Float32, 2, 2)
	)
	requireError(t, f16Src.UploadFloat32As([]float32{1, 2, 3, 4, 5, 6}), nil)
	requireError(t, bfSrc.UploadFloat32As([]float32{-1, -2, -3, -4, -5, -6}), nil)

	var scope = requireScope(t, device)
	requireError(t, scope.GatherRows(f16Src, []int64{2, 0}, f16Dst), nil)
	requireError(t, scope.GatherRows(bfSrc, []int64{1, 2}, bfDst), nil)
	requireError(t, scope.GatherRows(f16Src, []int64{2, 0}, f32Dst), nil)
	requireError(t, scope.GatherRows(bfSrc, []int64{1, 2}, f32Dst2), nil)
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)

	var gotF32 = make([]float32, 4)
	requireError(t, f16Dst.DownloadFloat32Into(gotF32), nil)
	if gotF32[0] != 5 || gotF32[1] != 6 || gotF32[2] != 1 || gotF32[3] != 2 {
		t.Fatalf("f16->f16 gather = %v", gotF32)
	}
	requireError(t, bfDst.DownloadFloat32Into(gotF32), nil)
	if gotF32[0] != -3 || gotF32[1] != -4 || gotF32[2] != -5 || gotF32[3] != -6 {
		t.Fatalf("bf16->bf16 gather = %v", gotF32)
	}
	requireError(t, f32Dst.DownloadInto(gotF32), nil)
	if gotF32[0] != 5 || gotF32[1] != 6 || gotF32[2] != 1 || gotF32[3] != 2 {
		t.Fatalf("f16->f32 gather = %v", gotF32)
	}
	requireError(t, f32Dst2.DownloadInto(gotF32), nil)
	if gotF32[0] != -3 || gotF32[1] != -4 || gotF32[2] != -5 || gotF32[3] != -6 {
		t.Fatalf("bf16->f32 gather = %v", gotF32)
	}
}
