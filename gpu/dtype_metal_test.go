//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"math"
	"slices"
	"testing"
)

func Test_MetalReducedPrecisionRoundTrip(t *testing.T) {
	var (
		d      = requireMetalDevice(t)
		native = d.state.backend.(metalBackend).runtime
		dtypes = []DType{Float16, BFloat16}
		counts = []int64{1, 3, 257, 1025, 4099}
	)

	for _, dtype := range dtypes {
		for _, count := range counts {
			t.Run(fmt.Sprintf("%s/%d", dtype, count), func(t *testing.T) {
				var baseline = native.ResourceSnapshot()
				runMetalReducedPrecisionRoundTrip(t, d, dtype, count)
				var after = native.ResourceSnapshot()
				if after.LiveBuffers != baseline.LiveBuffers || after.LiveBufferBytes != baseline.LiveBufferBytes ||
					after.LiveScopes != baseline.LiveScopes {
					t.Fatalf("native live resource counts changed from %+v to %+v", baseline, after)
				}
				if after.SubmittedCommands != baseline.SubmittedCommands+3 ||
					after.CompletedCommands != baseline.CompletedCommands+3 {
					t.Fatalf("want three completed submissions, before %+v, after %+v", baseline, after)
				}
			})
		}
	}
}

func runMetalReducedPrecisionRoundTrip(t *testing.T, d *Device, dtype DType, count int64) {
	t.Helper()

	var (
		ctx       = context.Background()
		source    = requireTensorWithDType(t, d, dtype, count)
		copied    = requireTensorWithDType(t, d, dtype, count)
		converted = requireTensorWithDType(t, d, dtype, count)
		scope     = requireScope(t, d)
		wantBytes = uint64(count) * 2
	)

	if source.DType() != dtype || source.Bytes() != wantBytes {
		t.Fatalf("source metadata = (%v, %d), want (%v, %d)", source.DType(), source.Bytes(), dtype, wantBytes)
	}

	// Build raw 16-bit patterns covering normals, subnormals, signed zero, max finite, ±Inf, and NaN.
	var (
		rawInput = make([]uint16, count)
		specials = []uint16{0x0000, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e01, 0xfe55}
	)
	if dtype == BFloat16 {
		specials = []uint16{0x0000, 0x8000, 0x0001, 0x8001, 0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffd5}
	}
	for i := range rawInput {
		if i < len(specials) {
			rawInput[i] = specials[i]
		} else {
			rawInput[i] = uint16((i*257 + 0x1357) & 0xffff)
		}
	}

	var expectedBits = slices.Clone(rawInput)
	requireError(t, source.UploadBits(rawInput), nil)
	clear(rawInput)

	requireError(t, scope.Copy(source, copied), nil)
	requireError(t, copied.DownloadBitsInto(make([]uint16, count)), ErrBusy)
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)

	var gotBits = make([]uint16, count)
	requireError(t, copied.DownloadBitsInto(gotBits), nil)
	if !slices.Equal(gotBits, expectedBits) {
		t.Fatalf("%s bit-for-bit round-trip mismatch for count=%d", dtype, count)
	}

	// Verify UploadFloat32As -> DownloadFloat32Into matches deterministic host conversion.
	var f32Input = make([]float32, count)
	for i := range f32Input {
		f32Input[i] = float32(i%61-30) * 0.125
	}
	requireError(t, converted.UploadFloat32As(f32Input), nil)
	var gotF32 = make([]float32, count)
	requireError(t, converted.DownloadFloat32Into(gotF32), nil)
	for i, input := range f32Input {
		var want float32
		if dtype == Float16 {
			want = Float16ToFloat32(Float32ToFloat16(input))
		} else {
			want = BFloat16ToFloat32(Float32ToBFloat16(input))
		}
		if math.Float32bits(gotF32[i]) != math.Float32bits(want) {
			t.Fatalf("%s float32 conversion mismatch at %d: got %g, want %g", dtype, i, gotF32[i], want)
		}
	}

	// Concurrent read-only sharing and deferred Close during submitted Metal copies.
	var (
		a           = requireScope(t, d)
		b           = requireScope(t, d)
		first       = requireTensorWithDType(t, d, dtype, count)
		second      = requireTensorWithDType(t, d, dtype, count)
		waitResults = make(chan error, 2)
	)
	requireError(t, a.Copy(copied, first), nil)
	requireError(t, b.Copy(copied, second), nil)
	requireError(t, copied.Close(), nil)
	requireError(t, a.Submit(ctx), nil)
	requireError(t, b.Submit(ctx), nil)
	go func() { waitResults <- a.Wait(ctx) }()
	go func() { waitResults <- b.Wait(ctx) }()
	requireError(t, <-waitResults, nil)
	requireError(t, <-waitResults, nil)

	var firstBits = make([]uint16, count)
	var secondBits = make([]uint16, count)
	requireError(t, first.DownloadBitsInto(firstBits), nil)
	requireError(t, second.DownloadBitsInto(secondBits), nil)
	if !slices.Equal(firstBits, expectedBits) || !slices.Equal(secondBits, expectedBits) {
		t.Fatalf("%s concurrent copy mismatch for count=%d", dtype, count)
	}

	requireError(t, a.Close(), nil)
	requireError(t, b.Close(), nil)
	for _, tensor := range []*Tensor{source, converted, first, second} {
		requireError(t, tensor.Close(), nil)
	}
}
