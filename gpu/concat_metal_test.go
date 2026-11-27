//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"math"
	"sync"
	"testing"
)

func concatReferenceU32(inputs [][]uint32, shapes [][]int64, axis int) (out []uint32) {
	var (
		rank       = len(shapes[0])
		outer      = 1
		inner      = 1
		dstAxisLen = 0
	)
	for d := 0; d < axis; d++ {
		outer *= int(shapes[0][d])
	}
	for d := axis + 1; d < rank; d++ {
		inner *= int(shapes[0][d])
	}
	for _, s := range shapes {
		dstAxisLen += int(s[axis])
	}
	out = make([]uint32, outer*dstAxisLen*inner)
	var axisOffset = 0
	for i, src := range inputs {
		var srcAxisLen = int(shapes[i][axis])
		for o := range outer {
			for a := range srcAxisLen {
				var (
					srcStart = (o*srcAxisLen + a) * inner
					dstStart = (o*dstAxisLen + axisOffset + a) * inner
				)
				copy(out[dstStart:dstStart+inner], src[srcStart:srcStart+inner])
			}
		}
		axisOffset += srcAxisLen
	}
	return out
}

func concatReferenceU16(inputs [][]uint16, shapes [][]int64, axis int) (out []uint16) {
	var (
		rank       = len(shapes[0])
		outer      = 1
		inner      = 1
		dstAxisLen = 0
	)
	for d := 0; d < axis; d++ {
		outer *= int(shapes[0][d])
	}
	for d := axis + 1; d < rank; d++ {
		inner *= int(shapes[0][d])
	}
	for _, s := range shapes {
		dstAxisLen += int(s[axis])
	}
	out = make([]uint16, outer*dstAxisLen*inner)
	var axisOffset = 0
	for i, src := range inputs {
		var srcAxisLen = int(shapes[i][axis])
		for o := range outer {
			for a := range srcAxisLen {
				var (
					srcStart = (o*srcAxisLen + a) * inner
					dstStart = (o*dstAxisLen + axisOffset + a) * inner
				)
				copy(out[dstStart:dstStart+inner], src[srcStart:srcStart+inner])
			}
		}
		axisOffset += srcAxisLen
	}
	return out
}

func shapeProduct(shape []int64) (n int) {
	n = 1
	for _, d := range shape {
		n *= int(d)
	}
	return n
}

func concatDestShape(shapes [][]int64, axis int) (out []int64) {
	out = make([]int64, len(shapes[0]))
	copy(out, shapes[0])
	out[axis] = 0
	for _, s := range shapes {
		out[axis] += s[axis]
	}
	return out
}

func makePatternU32(count int, seed uint32) (bits []uint32, floats []float32) {
	var specials = []uint32{
		0x00000000, // +0.0
		0x80000000, // -0.0
		0x7f800000, // +Inf
		0xff800000, // -Inf
		0x7fc00001, // qNaN payload 1
		0x7fc01234, // qNaN payload 0x1234
		0xffc0beef, // negative qNaN payload
		0x00000001, // smallest positive subnormal
		0x80000001, // smallest negative subnormal
		0x3f800000, // 1.0
		0xbf800000, // -1.0
	}
	bits = make([]uint32, count)
	floats = make([]float32, count)
	for i := range count {
		var b uint32
		if i < len(specials) {
			b = specials[(i+int(seed))%len(specials)]
		} else {
			// Mix deterministic finite + bit-diverse values.
			var v = float32(int(seed)*97+i*13-count/2) * 0.03125
			b = math.Float32bits(v)
		}
		bits[i] = b
		floats[i] = math.Float32frombits(b)
	}
	return bits, floats
}

func makePatternU16(count int, seed uint16) (bits []uint16) {
	var specials = []uint16{
		0x0000, // +0.0
		0x8000, // -0.0
		0x7c00, // +Inf (FP16) / finite (BF16)
		0xfc00, // -Inf (FP16)
		0x7f80, // +Inf (BF16) / NaN (FP16)
		0xff80, // -Inf (BF16) / NaN (FP16)
		0x7e01, // FP16 qNaN payload
		0x7fc1, // BF16 qNaN payload
		0xffd5, // negative qNaN payload
		0x0001, // subnormal
		0x8001, // negative subnormal
		0x3c00, // 1.0 (FP16)
		0x3f80, // 1.0 (BF16)
	}
	bits = make([]uint16, count)
	for i := range count {
		if i < len(specials) {
			bits[i] = specials[(i+int(seed))%len(specials)]
		} else {
			bits[i] = uint16((uint32(seed)*131 + uint32(i)*17 + 0x1234) & 0xffff)
		}
	}
	return bits
}

func Test_MetalTensorConcatBitExact(t *testing.T) {
	type testcase struct {
		name   string
		shapes [][]int64
		axis   int
	}
	var tests = []testcase{
		// 1D (outer == 1 blit fast path)
		{name: "1d_single_input", shapes: [][]int64{{17}}, axis: 0},
		{name: "1d_two_odd", shapes: [][]int64{{7}, {13}}, axis: 0},
		{name: "1d_three_prime", shapes: [][]int64{{1}, {31}, {19}}, axis: 0},
		{name: "1d_eight_inputs", shapes: [][]int64{{3}, {5}, {1}, {7}, {2}, {11}, {4}, {9}}, axis: 0},
		// 2D (axis 0 = blit, axis 1 = strided compute)
		{name: "2d_axis0_blit", shapes: [][]int64{{3, 17}, {5, 17}, {2, 17}}, axis: 0},
		{name: "2d_axis1_outer1_blit", shapes: [][]int64{{1, 7}, {1, 13}, {1, 5}}, axis: 1},
		{name: "2d_axis1_strided", shapes: [][]int64{{7, 3}, {7, 11}, {7, 5}}, axis: 1},
		{name: "2d_axis1_eight_inputs", shapes: [][]int64{{5, 2}, {5, 3}, {5, 1}, {5, 4}, {5, 7}, {5, 2}, {5, 5}, {5, 1}}, axis: 1},
		// 3D
		{name: "3d_axis0_blit", shapes: [][]int64{{2, 3, 5}, {4, 3, 5}}, axis: 0},
		{name: "3d_axis1_strided", shapes: [][]int64{{3, 2, 7}, {3, 5, 7}, {3, 1, 7}}, axis: 1},
		{name: "3d_axis2_strided", shapes: [][]int64{{3, 4, 5}, {3, 4, 9}}, axis: 2},
		// 4D NCHW (batch, channel, height, width)
		{name: "4d_batch_axis0_blit", shapes: [][]int64{{2, 8, 7, 9}, {3, 8, 7, 9}}, axis: 0},
		{name: "4d_channel_axis1_outer1_blit", shapes: [][]int64{{1, 5, 7, 9}, {1, 11, 7, 9}, {1, 3, 7, 9}}, axis: 1},
		{name: "4d_channel_axis1_strided", shapes: [][]int64{{4, 5, 7, 9}, {4, 11, 7, 9}, {4, 3, 7, 9}}, axis: 1},
		{name: "4d_spatial_axis2_strided", shapes: [][]int64{{2, 3, 5, 7}, {2, 3, 8, 7}}, axis: 2},
		{name: "4d_spatial_axis3_strided", shapes: [][]int64{{2, 3, 5, 7}, {2, 3, 5, 11}, {2, 3, 5, 3}}, axis: 3},
		// 6D
		{name: "6d_axis0_blit", shapes: [][]int64{{1, 2, 2, 2, 3, 3}, {2, 2, 2, 2, 3, 3}}, axis: 0},
		{name: "6d_axis3_strided", shapes: [][]int64{{2, 2, 3, 1, 2, 5}, {2, 2, 3, 4, 2, 5}, {2, 2, 3, 2, 2, 5}}, axis: 3},
		{name: "6d_axis5_strided", shapes: [][]int64{{2, 2, 2, 2, 3, 1}, {2, 2, 2, 2, 3, 4}}, axis: 5},
	}

	var (
		dev    = requireMetalDevice(t)
		ctx    = context.Background()
		dtypes = []DType{Float32, Float16, BFloat16}
	)

	for _, dt := range dtypes {
		for _, tt := range tests {
			t.Run(fmt.Sprintf("%s/%s", dt, tt.name), func(t *testing.T) {
				var (
					dstShape = concatDestShape(tt.shapes, tt.axis)
					inputs   = make([]*Tensor, len(tt.shapes))
					dst      = requireTensorWithDType(t, dev, dt, dstShape...)
				)
				if dt == Float32 {
					var rawInputs = make([][]uint32, len(tt.shapes))
					for i, s := range tt.shapes {
						var (
							bits, floats = makePatternU32(shapeProduct(s), uint32(i+1))
							tensor       = requireTensorWithDType(t, dev, dt, s...)
						)
						requireError(t, tensor.Upload(floats), nil)
						inputs[i] = tensor
						rawInputs[i] = bits
					}
					var scope, err = dev.NewScope()
					requireError(t, err, nil)
					requireError(t, scope.Concat(inputs, dst, tt.axis), nil)
					requireError(t, scope.Submit(ctx), nil)
					requireError(t, scope.Wait(ctx), nil)
					requireError(t, scope.Close(), nil)

					var gotFloats = make([]float32, shapeProduct(dstShape))
					requireError(t, dst.DownloadInto(gotFloats), nil)
					var wantBits = concatReferenceU32(rawInputs, tt.shapes, tt.axis)
					for idx, want := range wantBits {
						var got = math.Float32bits(gotFloats[idx])
						if got != want {
							t.Fatalf("index %d: got bits 0x%08x, want 0x%08x", idx, got, want)
						}
					}
				} else {
					var rawInputs = make([][]uint16, len(tt.shapes))
					for i, s := range tt.shapes {
						var (
							bits   = makePatternU16(shapeProduct(s), uint16(i+1))
							tensor = requireTensorWithDType(t, dev, dt, s...)
						)
						requireError(t, tensor.UploadBits(bits), nil)
						inputs[i] = tensor
						rawInputs[i] = bits
					}
					var scope, err = dev.NewScope()
					requireError(t, err, nil)
					requireError(t, scope.Concat(inputs, dst, tt.axis), nil)
					requireError(t, scope.Submit(ctx), nil)
					requireError(t, scope.Wait(ctx), nil)
					requireError(t, scope.Close(), nil)

					var gotBits = make([]uint16, shapeProduct(dstShape))
					requireError(t, dst.DownloadBitsInto(gotBits), nil)
					var wantBits = concatReferenceU16(rawInputs, tt.shapes, tt.axis)
					for idx, want := range wantBits {
						if gotBits[idx] != want {
							t.Fatalf("index %d: got bits 0x%04x, want 0x%04x", idx, gotBits[idx], want)
						}
					}
				}
			})
		}
	}
}

func Test_MetalTensorConcatRepeatedAndViews(t *testing.T) {
	var (
		dev = requireMetalDevice(t)
		ctx = context.Background()
	)
	for _, dt := range []DType{Float32, Float16, BFloat16} {
		t.Run(dt.String(), func(t *testing.T) {
			var (
				base = requireTensorWithDType(t, dev, dt, 12)
				v1   *Tensor
				v2   *Tensor
				dst0 = requireTensorWithDType(t, dev, dt, 9, 4) // axis=0 blit with [v1, v1, v2]
				dst1 = requireTensorWithDType(t, dev, dt, 3, 12)
				err  error
			)
			if v1, err = base.View([]int64{3, 4}); err != nil {
				t.Fatalf("View v1 failed: %v", err)
			}
			t.Cleanup(func() { requireError(t, v1.Close(), nil) })
			if v2, err = base.View([]int64{3, 4}); err != nil {
				t.Fatalf("View v2 failed: %v", err)
			}
			t.Cleanup(func() { requireError(t, v2.Close(), nil) })

			if dt == Float32 {
				var bits, floats = makePatternU32(12, 7)
				requireError(t, base.Upload(floats), nil)

				var scope, scopeErr = dev.NewScope()
				requireError(t, scopeErr, nil)
				requireError(t, scope.Concat([]*Tensor{v1, v1, v2}, dst0, 0), nil)
				requireError(t, scope.Concat([]*Tensor{v1, v2, v1}, dst1, 1), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)

				var (
					got0  = make([]float32, 36)
					got1  = make([]float32, 36)
					want0 = concatReferenceU32([][]uint32{bits, bits, bits}, [][]int64{{3, 4}, {3, 4}, {3, 4}}, 0)
					want1 = concatReferenceU32([][]uint32{bits, bits, bits}, [][]int64{{3, 4}, {3, 4}, {3, 4}}, 1)
				)
				requireError(t, dst0.DownloadInto(got0), nil)
				requireError(t, dst1.DownloadInto(got1), nil)
				for i := range want0 {
					if math.Float32bits(got0[i]) != want0[i] {
						t.Fatalf("axis=0 index %d: got 0x%08x want 0x%08x", i, math.Float32bits(got0[i]), want0[i])
					}
					if math.Float32bits(got1[i]) != want1[i] {
						t.Fatalf("axis=1 index %d: got 0x%08x want 0x%08x", i, math.Float32bits(got1[i]), want1[i])
					}
				}
			} else {
				var bits = makePatternU16(12, 9)
				requireError(t, base.UploadBits(bits), nil)

				var scope, scopeErr = dev.NewScope()
				requireError(t, scopeErr, nil)
				requireError(t, scope.Concat([]*Tensor{v1, v1, v2}, dst0, 0), nil)
				requireError(t, scope.Concat([]*Tensor{v1, v2, v1}, dst1, 1), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)

				var (
					got0  = make([]uint16, 36)
					got1  = make([]uint16, 36)
					want0 = concatReferenceU16([][]uint16{bits, bits, bits}, [][]int64{{3, 4}, {3, 4}, {3, 4}}, 0)
					want1 = concatReferenceU16([][]uint16{bits, bits, bits}, [][]int64{{3, 4}, {3, 4}, {3, 4}}, 1)
				)
				requireError(t, dst0.DownloadBitsInto(got0), nil)
				requireError(t, dst1.DownloadBitsInto(got1), nil)
				for i := range want0 {
					if got0[i] != want0[i] {
						t.Fatalf("axis=0 index %d: got 0x%04x want 0x%04x", i, got0[i], want0[i])
					}
					if got1[i] != want1[i] {
						t.Fatalf("axis=1 index %d: got 0x%04x want 0x%04x", i, got1[i], want1[i])
					}
				}
			}
		})
	}
}

// Test_MetalTensorConcatEncoderCoalescingAndDiscard verifies that Concat's
// outer==1 blit path cleanly ends any active coalesced compute encoder, its
// outer>1 strided compute path stays inside the coalesced compute encoder, and
// discarding an uncommitted scope after either path leaks zero resources.
func Test_MetalTensorConcatEncoderCoalescingAndDiscard(t *testing.T) {
	var (
		dev  = requireMetalDevice(t)
		ctx  = context.Background()
		a    = requireTensor(t, dev, 2, 3)
		b    = requireTensor(t, dev, 2, 5)
		aS   = requireTensor(t, dev, 2, 3)
		bS   = requireTensor(t, dev, 2, 5)
		cat1 = requireTensor(t, dev, 2, 8) // outer=2 strided compute concat
		cat0 = requireTensor(t, dev, 4, 8) // outer=1 blit concat [cat1, cat1]
		out  = requireTensor(t, dev, 4, 8)
	)
	requireError(t, a.Upload([]float32{1, 2, 3, 4, 5, 6}), nil)
	requireError(t, b.Upload([]float32{10, 11, 12, 13, 14, 20, 21, 22, 23, 24}), nil)

	// Interleaved compute -> strided concat (compute) -> blit concat (outer==1) -> compute in 1 Scope.
	var scope, err = dev.NewScope()
	requireError(t, err, nil)
	requireError(t, scope.Scale(a, aS, 2), nil)
	requireError(t, scope.Scale(b, bS, 2), nil)
	requireError(t, scope.Concat([]*Tensor{aS, bS}, cat1, 1), nil)
	requireError(t, scope.Concat([]*Tensor{cat1, cat1}, cat0, 0), nil)
	requireError(t, scope.Scale(cat0, out, 0.5), nil)
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)

	var got = make([]float32, 32)
	requireError(t, out.DownloadInto(got), nil)
	var want = []float32{
		1, 2, 3, 10, 11, 12, 13, 14,
		4, 5, 6, 20, 21, 22, 23, 24,
		1, 2, 3, 10, 11, 12, 13, 14,
		4, 5, 6, 20, 21, 22, 23, 24,
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("interleaved chain index %d: got %v want %v", i, got[i], want[i])
		}
	}

	// Discarding uncommitted scopes after outer>1 compute concat and outer==1 blit concat must not leak.
	var (
		native = dev.state.backend.(metalBackend).runtime
		before = native.ResourceSnapshot()
	)
	var discard1, dErr1 = dev.NewScope()
	requireError(t, dErr1, nil)
	requireError(t, discard1.Concat([]*Tensor{a, b}, cat1, 1), nil)
	requireError(t, discard1.Close(), nil)

	var discard2, dErr2 = dev.NewScope()
	requireError(t, dErr2, nil)
	requireError(t, discard2.Concat([]*Tensor{cat1, cat1}, cat0, 0), nil)
	requireError(t, discard2.Close(), nil)

	var after = native.ResourceSnapshot()
	if after.LiveScopes != before.LiveScopes || after.LiveBuffers != before.LiveBuffers {
		t.Fatalf("resource leak after discarded concat scopes: before=%+v after=%+v", before, after)
	}
}

func Test_MetalTensorConcatConcurrentStress(t *testing.T) {
	var (
		dev       = requireMetalDevice(t)
		ctx       = context.Background()
		a         = requireTensor(t, dev, 4, 7, 13)
		b         = requireTensor(t, dev, 4, 11, 13)
		c         = requireTensor(t, dev, 4, 5, 13)
		bitsA, fA = makePatternU32(4*7*13, 11)
		bitsB, fB = makePatternU32(4*11*13, 23)
		bitsC, fC = makePatternU32(4*5*13, 37)
		want1     = concatReferenceU32([][]uint32{bitsA, bitsB, bitsC}, [][]int64{{4, 7, 13}, {4, 11, 13}, {4, 5, 13}}, 1)
		want0     = concatReferenceU32([][]uint32{bitsA, bitsA}, [][]int64{{4, 7, 13}, {4, 7, 13}}, 0)
	)
	requireError(t, a.Upload(fA), nil)
	requireError(t, b.Upload(fB), nil)
	requireError(t, c.Upload(fC), nil)

	const (
		workers    = 8
		iterations = 12
	)
	var (
		wg   sync.WaitGroup
		errs = make(chan error, workers)
	)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var (
				dst1, err1 = dev.NewFloat32([]int64{4, 23, 13})
				dst0, err0 = dev.NewFloat32([]int64{8, 7, 13})
			)
			if err1 != nil {
				errs <- err1
				return
			}
			defer dst1.Close()
			if err0 != nil {
				errs <- err0
				return
			}
			defer dst0.Close()

			var (
				got1 = make([]float32, len(want1))
				got0 = make([]float32, len(want0))
			)
			for range iterations {
				var scope, sErr = dev.NewScope()
				if sErr != nil {
					errs <- sErr
					return
				}
				if sErr = scope.Concat([]*Tensor{a, b, c}, dst1, 1); sErr != nil {
					_ = scope.Close()
					errs <- sErr
					return
				}
				if sErr = scope.Concat([]*Tensor{a, a}, dst0, 0); sErr != nil {
					_ = scope.Close()
					errs <- sErr
					return
				}
				if sErr = scope.Submit(ctx); sErr != nil {
					_ = scope.Close()
					errs <- sErr
					return
				}
				if sErr = scope.Wait(ctx); sErr != nil {
					_ = scope.Close()
					errs <- sErr
					return
				}
				if sErr = scope.Close(); sErr != nil {
					errs <- sErr
					return
				}
				if sErr = dst1.DownloadInto(got1); sErr != nil {
					errs <- sErr
					return
				}
				if sErr = dst0.DownloadInto(got0); sErr != nil {
					errs <- sErr
					return
				}
				for i := range want1 {
					if math.Float32bits(got1[i]) != want1[i] {
						errs <- fmt.Errorf("strided concat mismatch at %d: got 0x%08x want 0x%08x", i, math.Float32bits(got1[i]), want1[i])
						return
					}
				}
				for i := range want0 {
					if math.Float32bits(got0[i]) != want0[i] {
						errs <- fmt.Errorf("blit concat mismatch at %d: got 0x%08x want 0x%08x", i, math.Float32bits(got0[i]), want0[i])
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}
