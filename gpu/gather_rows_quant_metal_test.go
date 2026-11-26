//go:build darwin && cgo && metal && !purego

package gpu

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"testing"
)

func Test_MetalGatherRowsQuantizedBitExact(t *testing.T) {
	var dev = requireMetalDevice(t)

	type shapeCase struct {
		name    string
		srcRows int64
		cols    int64
		rows    []int64
	}

	var buildIndices = func(count int, srcRows int64) (ids []int64) {
		ids = make([]int64, count)
		for i := range count {
			switch i {
			case 0:
				ids[i] = 0
			case count - 1:
				ids[i] = srcRows - 1
			default:
				ids[i] = int64((i*7 + 3) % int(srcRows))
			}
		}
		return ids
	}

	var shapes = []shapeCase{
		{name: "M1_C32", srcRows: 8, cols: 32, rows: []int64{7}},
		{name: "M1_C96", srcRows: 8, cols: 96, rows: []int64{0}},
		{name: "M6_C64", srcRows: 8, cols: 64, rows: []int64{7, 0, 7, 3, 1, 0}},
		{name: "M27_C96", srcRows: 12, cols: 96, rows: buildIndices(27, 12)},
		{name: "M28_C128", srcRows: 12, cols: 128, rows: buildIndices(28, 12)},
		{name: "M29_C2048", srcRows: 16, cols: 2048, rows: buildIndices(29, 16)},
		{name: "M600_C64_multichunk", srcRows: 16, cols: 64, rows: buildIndices(600, 16)},
	}

	for _, dtype := range []DType{Q8_0, Q4_0, Q4_1} {
		for _, sc := range shapes {
			t.Run(fmt.Sprintf("%s/%s", dtype, sc.name), func(t *testing.T) {
				var (
					srcData  = attentionValues(sc.srcRows*sc.cols, 913, 3)
					srcBytes []byte
					deqRef   []float32
					err      error
				)
				switch dtype {
				case Q8_0:
					srcBytes, err = QuantizeQ8_0(srcData)
					requireError(t, err, nil)
					deqRef, err = DequantizeQ8_0(srcBytes)
					requireError(t, err, nil)
				case Q4_0:
					srcBytes, err = QuantizeQ4_0(srcData)
					requireError(t, err, nil)
					deqRef, err = DequantizeQ4_0(srcBytes)
					requireError(t, err, nil)
				case Q4_1:
					srcBytes, err = QuantizeQ4_1(srcData)
					requireError(t, err, nil)
					deqRef, err = DequantizeQ4_1(srcBytes)
					requireError(t, err, nil)
				}

				var (
					src      = requireTensorWithDType(t, dev, dtype, sc.srcRows, sc.cols)
					sameDst  = requireTensorWithDType(t, dev, dtype, int64(len(sc.rows)), sc.cols)
					mixedDst = requireTensor(t, dev, int64(len(sc.rows)), sc.cols)
					scope    = requireScope(t, dev)
				)
				requireError(t, src.UploadBytes(srcBytes), nil)
				requireError(t, scope.GatherRows(src, sc.rows, sameDst), nil)
				requireError(t, scope.GatherRows(src, sc.rows, mixedDst), nil)
				completeAttentionScope(t, scope)

				var gotRaw = make([]byte, sameDst.Bytes())
				requireError(t, sameDst.DownloadBytesInto(gotRaw), nil)
				var rowBytes = int((uint64(sc.cols) / QuantBlockSize) * dtype.BlockBytes())
				for outRow, srcRow := range sc.rows {
					var (
						wantRow = srcBytes[int(srcRow)*rowBytes : (int(srcRow)+1)*rowBytes]
						gotRow  = gotRaw[outRow*rowBytes : (outRow+1)*rowBytes]
					)
					if !bytes.Equal(gotRow, wantRow) {
						t.Fatalf("%s %s same-dtype row %d (src %d) mismatch", dtype, sc.name, outRow, srcRow)
					}
				}

				var gotF32 = make([]float32, int64(len(sc.rows))*sc.cols)
				requireError(t, mixedDst.DownloadInto(gotF32), nil)
				for outRow, srcRow := range sc.rows {
					for col := range sc.cols {
						var (
							got  = gotF32[int64(outRow)*sc.cols+col]
							want = deqRef[srcRow*sc.cols+col]
						)
						if math.Float32bits(got) != math.Float32bits(want) {
							t.Fatalf("%s %s mixed [%d,%d]: got %v (0x%08x), want %v (0x%08x)",
								dtype, sc.name, outRow, col, got, math.Float32bits(got), want, math.Float32bits(want))
						}
					}
				}
			})
		}
	}
}

func Test_MetalGatherRowsQuantizedSpecialScales(t *testing.T) {
	var (
		dev       = requireMetalDevice(t)
		scaleBits = []uint16{
			0x0000,                 // +0.0
			0x8000,                 // -0.0
			Float32ToFloat16(-1.5), // negative scale
			0x7bff,                 // max finite FP16 (65504)
			0x0001,                 // smallest positive FP16 subnormal
			0x03ff,                 // largest FP16 subnormal
			0x7e00,                 // quiet NaN
			0x7c00,                 // +Inf
			0xfc00,                 // -Inf
		}
		srcRows = int64(len(scaleBits))
		cols    = int64(QuantBlockSize * 2) // 2 blocks per row (tests odd & even block offsets)
		ids     = []int64{srcRows - 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0}
	)

	for _, dtype := range []DType{Q8_0, Q4_0, Q4_1} {
		t.Run(dtype.String(), func(t *testing.T) {
			var (
				blockBytes = int(dtype.BlockBytes())
				blocksRow  = int(cols / QuantBlockSize)
				raw        = make([]byte, int(srcRows)*blocksRow*blockBytes)
			)
			for row := range int(srcRows) {
				for b := range blocksRow {
					var dst = raw[(row*blocksRow+b)*blockBytes : (row*blocksRow+b+1)*blockBytes]
					binary.LittleEndian.PutUint16(dst[0:2], scaleBits[row])
					switch dtype {
					case Q8_0:
						for i := range QuantBlockSize {
							// Cover -128, 0, +127, and varied signed values.
							var q int8
							switch i {
							case 0:
								q = 0
							case 1:
								q = -128
							case 2:
								q = 127
							default:
								q = int8((i*17 + row*5 + b*11) - 64)
							}
							dst[2+i] = byte(q)
						}
					case Q4_0:
						for j := range 16 {
							// Cover low/high nibbles 0 (-8), 8 (0), 15 (+7).
							var (
								lo = uint8((j + row + b) & 0x0f)
								hi = uint8((15 - j + row + b) & 0x0f)
							)
							if j == 0 {
								lo = 8 // 0
								hi = 0 // -8
							} else if j == 1 {
								lo = 15 // +7
								hi = 8  // 0
							}
							dst[2+j] = lo | (hi << 4)
						}
					case Q4_1:
						// Pair special scales with varied min values (including negative and non-zero with d=0).
						var minBits = Float32ToFloat16(float32(b)*1.25 - 2.5)
						if row == 0 {
							minBits = Float32ToFloat16(3.5)
						}
						binary.LittleEndian.PutUint16(dst[2:4], minBits)
						for j := range 16 {
							var (
								lo = uint8((j + row + b) & 0x0f)
								hi = uint8((15 - j + row + b) & 0x0f)
							)
							if j == 0 {
								lo = 0
								hi = 15
							}
							dst[4+j] = lo | (hi << 4)
						}
					}
				}
			}

			var (
				deqRef []float32
				err    error
			)
			switch dtype {
			case Q8_0:
				deqRef, err = DequantizeQ8_0(raw)
			case Q4_0:
				deqRef, err = DequantizeQ4_0(raw)
			case Q4_1:
				deqRef, err = DequantizeQ4_1(raw)
			}
			requireError(t, err, nil)

			var (
				src      = requireTensorWithDType(t, dev, dtype, srcRows, cols)
				sameDst  = requireTensorWithDType(t, dev, dtype, int64(len(ids)), cols)
				mixedDst = requireTensor(t, dev, int64(len(ids)), cols)
				scope    = requireScope(t, dev)
			)
			requireError(t, src.UploadBytes(raw), nil)
			requireError(t, scope.GatherRows(src, ids, sameDst), nil)
			requireError(t, scope.GatherRows(src, ids, mixedDst), nil)
			completeAttentionScope(t, scope)

			var gotRaw = make([]byte, sameDst.Bytes())
			requireError(t, sameDst.DownloadBytesInto(gotRaw), nil)
			var rowBytes = blocksRow * blockBytes
			for outRow, srcRow := range ids {
				if !bytes.Equal(gotRaw[outRow*rowBytes:(outRow+1)*rowBytes], raw[int(srcRow)*rowBytes:(int(srcRow)+1)*rowBytes]) {
					t.Fatalf("%s special scales same-dtype row %d mismatch", dtype, outRow)
				}
			}

			var gotF32 = make([]float32, int64(len(ids))*cols)
			requireError(t, mixedDst.DownloadInto(gotF32), nil)
			for outRow, srcRow := range ids {
				for col := range cols {
					var (
						got  = gotF32[int64(outRow)*cols+col]
						want = deqRef[srcRow*cols+col]
					)
					if math.IsNaN(float64(want)) {
						if !math.IsNaN(float64(got)) {
							t.Fatalf("%s special scale 0x%04x [%d,%d]: got %v, want NaN", dtype, scaleBits[srcRow], outRow, col, got)
						}
						continue
					}
					if math.Float32bits(got) != math.Float32bits(want) {
						t.Fatalf("%s special scale 0x%04x [%d,%d]: got %v (0x%08x), want %v (0x%08x)",
							dtype, scaleBits[srcRow], outRow, col, got, math.Float32bits(got), want, math.Float32bits(want))
					}
				}
			}
		})
	}
}

func Test_MetalGatherRowsQuantizedTiedLlama128256x2048(t *testing.T) {
	const (
		vocab int64 = 128256
		width int64 = 2048
	)
	var dev = requireMetalDevice(t)

	for _, dtype := range []DType{Q8_0, Q4_0, Q4_1} {
		t.Run(dtype.String(), func(t *testing.T) {
			var (
				native    = dev.state.backend.(metalBackend).runtime
				before    = native.ResourceSnapshot()
				wantBytes = uint64(vocab) * (uint64(width) / QuantBlockSize) * dtype.BlockBytes()
			)
			t.Run("exec", func(t *testing.T) {
				var (
					weight = requireTensorWithDType(t, dev, dtype, vocab, width)
					embed  = requireTensor(t, dev, 1, width)
					logits = requireTensor(t, dev, 1, vocab)
					scope  = requireScope(t, dev)
				)
				if weight.Bytes() != wantBytes {
					t.Fatalf("%s weight bytes = %d, want %d", dtype, weight.Bytes(), wantBytes)
				}

				var (
					rowBytes = int((uint64(width) / QuantBlockSize) * dtype.BlockBytes())
					raw      = make([]byte, wantBytes)
					onesRow  = make([]float32, width)
					twosRow  = make([]float32, width)
				)
				for col := range width {
					onesRow[col] = 1.0
					twosRow[col] = 2.0
				}
				var (
					encOne []byte
					encTwo []byte
					err    error
				)
				switch dtype {
				case Q8_0:
					encOne, err = QuantizeQ8_0(onesRow)
					requireError(t, err, nil)
					encTwo, err = QuantizeQ8_0(twosRow)
					requireError(t, err, nil)
				case Q4_0:
					encOne, err = QuantizeQ4_0(onesRow)
					requireError(t, err, nil)
					encTwo, err = QuantizeQ4_0(twosRow)
					requireError(t, err, nil)
				case Q4_1:
					encOne, err = QuantizeQ4_1(onesRow)
					requireError(t, err, nil)
					encTwo, err = QuantizeQ4_1(twosRow)
					requireError(t, err, nil)
				}
				copy(raw[42*rowBytes:43*rowBytes], encOne)
				copy(raw[128000*rowBytes:128001*rowBytes], encTwo)

				requireError(t, weight.UploadBytes(raw), nil)
				requireError(t, scope.GatherRows(weight, []int64{42}, embed), nil)
				requireError(t, scope.MatMulRightTranspose(embed, weight, logits), nil)
				completeAttentionScope(t, scope)

				var (
					wantEmbed []float32
					wantTwo   []float32
				)
				switch dtype {
				case Q8_0:
					wantEmbed, err = DequantizeQ8_0(encOne)
					requireError(t, err, nil)
					wantTwo, err = DequantizeQ8_0(encTwo)
					requireError(t, err, nil)
				case Q4_0:
					wantEmbed, err = DequantizeQ4_0(encOne)
					requireError(t, err, nil)
					wantTwo, err = DequantizeQ4_0(encTwo)
					requireError(t, err, nil)
				case Q4_1:
					wantEmbed, err = DequantizeQ4_1(encOne)
					requireError(t, err, nil)
					wantTwo, err = DequantizeQ4_1(encTwo)
					requireError(t, err, nil)
				}

				var gotEmbed = make([]float32, width)
				requireError(t, embed.DownloadInto(gotEmbed), nil)
				for col := range width {
					if math.Float32bits(gotEmbed[col]) != math.Float32bits(wantEmbed[col]) {
						t.Fatalf("%s embed[%d] = %v, want %v", dtype, col, gotEmbed[col], wantEmbed[col])
					}
				}

				var (
					wantSelf float64
					wantPeer float64
				)
				for col := range width {
					wantSelf += float64(wantEmbed[col]) * float64(wantEmbed[col])
					wantPeer += float64(wantEmbed[col]) * float64(wantTwo[col])
				}

				var gotLogits = make([]float32, vocab)
				requireError(t, logits.DownloadInto(gotLogits), nil)
				if gotLogits[0] != 0 {
					t.Fatalf("%s logits[0] = %v, want 0", dtype, gotLogits[0])
				}
				if math.Abs(float64(gotLogits[42])-wantSelf) > 1e-2 {
					t.Fatalf("%s logits[42] = %v, want %v", dtype, gotLogits[42], wantSelf)
				}
				if math.Abs(float64(gotLogits[128000])-wantPeer) > 1e-2 {
					t.Fatalf("%s logits[128000] = %v, want %v", dtype, gotLogits[128000], wantPeer)
				}
			})

			var after = native.ResourceSnapshot()
			if after.LiveBuffers != before.LiveBuffers || after.LiveBufferBytes != before.LiveBufferBytes ||
				after.CreatedBuffers-before.CreatedBuffers != 3 || after.SubmittedCommands-before.SubmittedCommands != 1 {
				t.Fatalf("%s unexpected native resources: before %+v after %+v", dtype, before, after)
			}
		})
	}
}

func Test_MetalGatherRowsQuantizedViewAliasAndConcurrency(t *testing.T) {
	var (
		dev = requireMetalDevice(t)
		ctx = context.Background()
	)
	const (
		srcRows int64 = 16
		cols    int64 = 128
	)
	var (
		srcData   = attentionValues(srcRows*cols, 777, 2)
		q8Raw, _  = QuantizeQ8_0(srcData)
		q4Raw, _  = QuantizeQ4_0(srcData)
		q41Raw, _ = QuantizeQ4_1(srcData)
		q8Ref, _  = DequantizeQ8_0(q8Raw)
		q4Ref, _  = DequantizeQ4_0(q4Raw)
		q41Ref, _ = DequantizeQ4_1(q41Raw)
		q8Src     = requireTensorWithDType(t, dev, Q8_0, srcRows, cols)
		q4Src     = requireTensorWithDType(t, dev, Q4_0, srcRows, cols)
		q41Src    = requireTensorWithDType(t, dev, Q4_1, srcRows, cols)
	)
	requireError(t, q8Src.UploadBytes(q8Raw), nil)
	requireError(t, q4Src.UploadBytes(q4Raw), nil)
	requireError(t, q41Src.UploadBytes(q41Raw), nil)

	// Alias rejection: same quantized storage as source and destination,
	// and competing writes to distinct views of the same Float32 destination.
	var (
		q8Alias     = *q8Src
		baseDst     = requireTensor(t, dev, 2, cols)
		viewA, errA = baseDst.View([]int64{2, cols})
		viewB, errB = baseDst.View([]int64{2, cols})
	)
	requireError(t, errA, nil)
	requireError(t, errB, nil)
	t.Cleanup(func() {
		requireError(t, viewA.Close(), nil)
		requireError(t, viewB.Close(), nil)
	})

	var scopeAlias = requireScope(t, dev)
	requireError(t, scopeAlias.GatherRows(q8Src, []int64{0, 1}, &q8Alias), ErrInvalid)
	requireError(t, scopeAlias.Close(), nil)

	var (
		first     = requireScope(t, dev)
		competing = requireScope(t, dev)
	)
	requireError(t, first.GatherRows(q8Src, []int64{0, 1}, viewA), nil)
	requireError(t, competing.GatherRows(q4Src, []int64{2, 3}, viewB), ErrBusy)
	requireError(t, competing.Close(), nil)
	completeAttentionScope(t, first)

	// Concurrent read-sharing across 8 workers with -race -tags=metal.
	const workers = 8
	var (
		wg   sync.WaitGroup
		errs = make(chan error, workers)
	)
	for w := range workers {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			var (
				dtype = Q8_0
				src   = q8Src
				raw   = q8Raw
				ref   = q8Ref
			)
			switch worker % 3 {
			case 1:
				dtype = Q4_0
				src = q4Src
				raw = q4Raw
				ref = q4Ref
			case 2:
				dtype = Q4_1
				src = q41Src
				raw = q41Raw
				ref = q41Ref
			}
			var (
				ids       = []int64{int64(worker % int(srcRows)), int64((worker*3 + 5) % int(srcRows)), srcRows - 1}
				f32Dst, e = dev.NewFloat32([]int64{int64(len(ids)), cols})
			)
			if e != nil {
				errs <- e
				return
			}
			defer f32Dst.Close()
			var sameDst, e2 = dev.NewTensorWithDType([]int64{int64(len(ids)), cols}, dtype)
			if e2 != nil {
				errs <- e2
				return
			}
			defer sameDst.Close()

			var sc, e3 = dev.NewScope()
			if e3 != nil {
				errs <- e3
				return
			}
			defer sc.Close()
			if e = sc.GatherRows(src, ids, f32Dst); e != nil {
				errs <- e
				return
			}
			if e = sc.GatherRows(src, ids, sameDst); e != nil {
				errs <- e
				return
			}
			if e = sc.Submit(ctx); e != nil {
				errs <- e
				return
			}
			if e = sc.Wait(ctx); e != nil {
				errs <- e
				return
			}

			var gotF32 = make([]float32, int64(len(ids))*cols)
			if e = f32Dst.DownloadInto(gotF32); e != nil {
				errs <- e
				return
			}
			for outRow, srcRow := range ids {
				for col := range cols {
					if math.Float32bits(gotF32[int64(outRow)*cols+col]) != math.Float32bits(ref[srcRow*cols+col]) {
						errs <- fmt.Errorf("worker %d %s f32 mismatch at [%d,%d]", worker, dtype, outRow, col)
						return
					}
				}
			}

			var gotRaw = make([]byte, sameDst.Bytes())
			if e = sameDst.DownloadBytesInto(gotRaw); e != nil {
				errs <- e
				return
			}
			var rowBytes = int((uint64(cols) / QuantBlockSize) * dtype.BlockBytes())
			for outRow, srcRow := range ids {
				if !bytes.Equal(gotRaw[outRow*rowBytes:(outRow+1)*rowBytes], raw[int(srcRow)*rowBytes:(int(srcRow)+1)*rowBytes]) {
					errs <- fmt.Errorf("worker %d %s raw mismatch at row %d", worker, dtype, outRow)
					return
				}
			}
		}(w)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}
