package gpu

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"
)

func Test_QuantizedDTypeMetadata(t *testing.T) {
	type testcase struct {
		dtype       DType
		valid       bool
		quantized   bool
		bytes       uint64
		blockSize   uint64
		blockBytes  uint64
		stringValue string
	}

	var tests = []testcase{
		{dtype: 0, valid: false, quantized: false, bytes: 0, blockSize: 0, blockBytes: 0, stringValue: "invalid"},
		{dtype: Float32, valid: true, quantized: false, bytes: 4, blockSize: 1, blockBytes: 4, stringValue: "float32"},
		{dtype: Float16, valid: true, quantized: false, bytes: 2, blockSize: 1, blockBytes: 2, stringValue: "float16"},
		{dtype: BFloat16, valid: true, quantized: false, bytes: 2, blockSize: 1, blockBytes: 2, stringValue: "bfloat16"},
		{dtype: Q8_0, valid: true, quantized: true, bytes: 0, blockSize: 32, blockBytes: 34, stringValue: "q8_0"},
		{dtype: Q4_0, valid: true, quantized: true, bytes: 0, blockSize: 32, blockBytes: 18, stringValue: "q4_0"},
		{dtype: Q4_1, valid: true, quantized: true, bytes: 0, blockSize: 32, blockBytes: 20, stringValue: "q4_1"},
		{dtype: DType(99), valid: false, quantized: false, bytes: 0, blockSize: 0, blockBytes: 0, stringValue: "invalid"},
	}

	for _, tt := range tests {
		if tt.dtype.Valid() != tt.valid ||
			tt.dtype.IsQuantized() != tt.quantized ||
			tt.dtype.Bytes() != tt.bytes ||
			tt.dtype.BlockSize() != tt.blockSize ||
			tt.dtype.BlockBytes() != tt.blockBytes ||
			tt.dtype.String() != tt.stringValue {
			t.Fatalf("DType(%d) = (valid=%t, quant=%t, bytes=%d, blockSize=%d, blockBytes=%d, str=%q), want (%t, %t, %d, %d, %d, %q)",
				tt.dtype, tt.dtype.Valid(), tt.dtype.IsQuantized(), tt.dtype.Bytes(), tt.dtype.BlockSize(), tt.dtype.BlockBytes(), tt.dtype.String(),
				tt.valid, tt.quantized, tt.bytes, tt.blockSize, tt.blockBytes, tt.stringValue)
		}
	}
}

func Test_QuantizeDequantizeQ8_0(t *testing.T) {
	// Invalid lengths return ErrInvalid.
	for _, length := range []int{0, 1, 16, 31, 33, 63} {
		var _, err = QuantizeQ8_0(make([]float32, length))
		requireError(t, err, ErrInvalid)
	}
	for _, length := range []int{0, 1, 33, 35, 67} {
		var _, err = DequantizeQ8_0(make([]byte, length))
		requireError(t, err, ErrInvalid)
	}

	// Zero block produces zero scale and zero payload.
	var (
		zeros           = make([]float32, 32)
		zeroBytes, errZ = QuantizeQ8_0(zeros)
	)
	requireError(t, errZ, nil)
	if len(zeroBytes) != QuantBlockBytesQ8_0 {
		t.Fatalf("Q8_0 zero block bytes = %d, want %d", len(zeroBytes), QuantBlockBytesQ8_0)
	}
	for i, b := range zeroBytes {
		if b != 0 {
			t.Fatalf("zero block byte[%d] = %d, want 0", i, b)
		}
	}
	var deqZero, errDZ = DequantizeQ8_0(zeroBytes)
	requireError(t, errDZ, nil)
	if !slices.Equal(deqZero, zeros) {
		t.Fatalf("dequantized zeros = %v, want all zeros", deqZero)
	}

	// Exact known scale and signed int8 layout verification.
	var input = make([]float32, 64)
	for i := range 32 {
		input[i] = float32(i-16) * 0.5 // maxAbs = 8.0
	}
	for i := range 32 {
		input[32+i] = float32(16-i) * 0.25 // maxAbs = 4.0
	}
	var encoded, errQ = QuantizeQ8_0(input)
	requireError(t, errQ, nil)
	if len(encoded) != 2*QuantBlockBytesQ8_0 {
		t.Fatalf("encoded length = %d, want %d", len(encoded), 2*QuantBlockBytesQ8_0)
	}
	var decoded, errD = DequantizeQ8_0(encoded)
	requireError(t, errD, nil)
	for i, want := range input {
		var diff = math.Abs(float64(decoded[i] - want))
		if diff > 0.07 {
			t.Fatalf("Q8_0 index %d: got %g, want %g (diff %g)", i, decoded[i], want, diff)
		}
	}

	// Verify exact byte layout with manual scale and int8(-128).
	var (
		raw    = make([]byte, QuantBlockBytesQ8_0)
		neg128 = int8(-128)
	)
	binary.LittleEndian.PutUint16(raw[0:2], Float32ToFloat16(0.25))
	raw[2] = byte(neg128)
	raw[3] = byte(int8(127))
	var manual, errM = DequantizeQ8_0(raw)
	requireError(t, errM, nil)
	if manual[0] != -32.0 || manual[1] != 31.75 || manual[2] != 0.0 {
		t.Fatalf("manual Q8_0 dequant = [%g, %g, %g], want [-32, 31.75, 0]", manual[0], manual[1], manual[2])
	}

	// NaN propagation in block.
	var nanBlock = make([]float32, 32)
	nanBlock[5] = float32(math.NaN())
	var nanEnc, errN = QuantizeQ8_0(nanBlock)
	requireError(t, errN, nil)
	var nanDec, errND = DequantizeQ8_0(nanEnc)
	requireError(t, errND, nil)
	if !math.IsNaN(float64(nanDec[0])) {
		t.Fatalf("NaN block did not propagate NaN: got %g", nanDec[0])
	}
}

func Test_QuantizeDequantizeQ4_0(t *testing.T) {
	// Invalid lengths return ErrInvalid.
	for _, length := range []int{0, 1, 16, 31, 33, 63} {
		var _, err = QuantizeQ4_0(make([]float32, length))
		requireError(t, err, ErrInvalid)
	}
	for _, length := range []int{0, 1, 17, 19, 35} {
		var _, err = DequantizeQ4_0(make([]byte, length))
		requireError(t, err, ErrInvalid)
	}

	// Zero block produces zero scale and 0x88 (offset 8 -> 0) nibbles.
	var (
		zeros           = make([]float32, 32)
		zeroBytes, errZ = QuantizeQ4_0(zeros)
	)
	requireError(t, errZ, nil)
	if len(zeroBytes) != QuantBlockBytesQ4_0 {
		t.Fatalf("Q4_0 zero block bytes = %d, want %d", len(zeroBytes), QuantBlockBytesQ4_0)
	}
	if binary.LittleEndian.Uint16(zeroBytes[0:2]) != 0 {
		t.Fatalf("zero block scale = %x, want 0", binary.LittleEndian.Uint16(zeroBytes[0:2]))
	}
	for j := range 16 {
		if zeroBytes[2+j] != 0x88 {
			t.Fatalf("zero block qs[%d] = 0x%02x, want 0x88", j, zeroBytes[2+j])
		}
	}
	var deqZero, errDZ = DequantizeQ4_0(zeroBytes)
	requireError(t, errDZ, nil)
	if !slices.Equal(deqZero, zeros) {
		t.Fatalf("dequantized Q4_0 zeros = %v, want all zeros", deqZero)
	}

	// Verify exact GGUF split-half nibble ordering:
	// low nibble of qs[j] -> element j (0..15), high nibble of qs[j] -> element j+16 (16..31).
	var raw = make([]byte, QuantBlockBytesQ4_0)
	binary.LittleEndian.PutUint16(raw[0:2], Float32ToFloat16(0.5))
	for j := range 16 {
		var (
			lo = uint8(j)      // q[j] = j - 8
			hi = uint8(15 - j) // q[j+16] = (15 - j) - 8 = 7 - j
		)
		raw[2+j] = lo | (hi << 4)
	}
	var manual, errM = DequantizeQ4_0(raw)
	requireError(t, errM, nil)
	for j := range 16 {
		var (
			wantLo = 0.5 * float32(j-8)
			wantHi = 0.5 * float32(7-j)
		)
		if manual[j] != wantLo {
			t.Fatalf("Q4_0 low nibble element[%d] = %g, want %g", j, manual[j], wantLo)
		}
		if manual[j+16] != wantHi {
			t.Fatalf("Q4_0 high nibble element[%d] = %g, want %g", j+16, manual[j+16], wantHi)
		}
	}

	// Round-trip accuracy check on two blocks (positive and negative maxVal).
	var input = make([]float32, 64)
	for i := range 32 {
		input[i] = float32(i-12) * 0.25 // maxVal = +4.75
	}
	for i := range 32 {
		input[32+i] = float32(10-i) * 0.25 // maxVal = -5.25
	}
	var encoded, errQ = QuantizeQ4_0(input)
	requireError(t, errQ, nil)
	var decoded, errD = DequantizeQ4_0(encoded)
	requireError(t, errD, nil)
	for i, want := range input {
		var maxErr = 0.5 // within half a quantization step (~5.25/8 * 0.5)
		if math.Abs(float64(decoded[i]-want)) > maxErr {
			t.Fatalf("Q4_0 index %d: got %g, want %g", i, decoded[i], want)
		}
	}

	// NaN propagation in block.
	var nanBlock = make([]float32, 32)
	nanBlock[19] = float32(math.NaN())
	var nanEnc, errN = QuantizeQ4_0(nanBlock)
	requireError(t, errN, nil)
	var nanDec, errND = DequantizeQ4_0(nanEnc)
	requireError(t, errND, nil)
	if !math.IsNaN(float64(nanDec[19])) {
		t.Fatalf("NaN block did not propagate NaN: got %g", nanDec[19])
	}
}

func Test_QuantizeDequantizeQ4_1(t *testing.T) {
	// Invalid lengths return ErrInvalid.
	for _, length := range []int{0, 1, 16, 31, 33, 63} {
		var _, err = QuantizeQ4_1(make([]float32, length))
		requireError(t, err, ErrInvalid)
	}
	for _, length := range []int{0, 1, 19, 21, 39} {
		var _, err = DequantizeQ4_1(make([]byte, length))
		requireError(t, err, ErrInvalid)
	}

	// Zero block produces zero scale, zero min, and zero nibbles.
	var (
		zeros           = make([]float32, 32)
		zeroBytes, errZ = QuantizeQ4_1(zeros)
	)
	requireError(t, errZ, nil)
	if len(zeroBytes) != QuantBlockBytesQ4_1 {
		t.Fatalf("Q4_1 zero block bytes = %d, want %d", len(zeroBytes), QuantBlockBytesQ4_1)
	}
	for i, b := range zeroBytes {
		if b != 0 {
			t.Fatalf("zero block byte[%d] = 0x%02x, want 0", i, b)
		}
	}
	var deqZero, errDZ = DequantizeQ4_1(zeroBytes)
	requireError(t, errDZ, nil)
	if !slices.Equal(deqZero, zeros) {
		t.Fatalf("dequantized Q4_1 zeros = %v, want all zeros", deqZero)
	}

	// Constant non-zero block encodes d=0, m=Float32ToFloat16(min), qs=0.
	var constant = make([]float32, 32)
	for i := range constant {
		constant[i] = -3.25
	}
	var constBytes, errC = QuantizeQ4_1(constant)
	requireError(t, errC, nil)
	if binary.LittleEndian.Uint16(constBytes[0:2]) != 0 {
		t.Fatalf("constant block scale = 0x%04x, want 0", binary.LittleEndian.Uint16(constBytes[0:2]))
	}
	if binary.LittleEndian.Uint16(constBytes[2:4]) != Float32ToFloat16(-3.25) {
		t.Fatalf("constant block min = 0x%04x, want 0x%04x", binary.LittleEndian.Uint16(constBytes[2:4]), Float32ToFloat16(-3.25))
	}
	for j := range 16 {
		if constBytes[4+j] != 0 {
			t.Fatalf("constant block qs[%d] = 0x%02x, want 0", j, constBytes[4+j])
		}
	}
	var deqConst, errDC = DequantizeQ4_1(constBytes)
	requireError(t, errDC, nil)
	if !slices.Equal(deqConst, constant) {
		t.Fatalf("dequantized Q4_1 constant = %v, want %v", deqConst, constant)
	}

	// Verify exact GGUF split-half unsigned nibble ordering:
	// low nibble of qs[j] -> element j (0..15), high nibble of qs[j] -> element j+16 (16..31).
	var raw = make([]byte, QuantBlockBytesQ4_1)
	binary.LittleEndian.PutUint16(raw[0:2], Float32ToFloat16(0.5))
	binary.LittleEndian.PutUint16(raw[2:4], Float32ToFloat16(-2.25))
	for j := range 16 {
		var (
			lo = uint8(j)
			hi = uint8(15 - j)
		)
		raw[4+j] = lo | (hi << 4)
	}
	var manual, errM = DequantizeQ4_1(raw)
	requireError(t, errM, nil)
	for j := range 16 {
		var (
			wantLo = float32(0.5*float32(j)) - 2.25
			wantHi = float32(0.5*float32(15-j)) - 2.25
		)
		if manual[j] != wantLo {
			t.Fatalf("Q4_1 low nibble element[%d] = %g, want %g", j, manual[j], wantLo)
		}
		if manual[j+16] != wantHi {
			t.Fatalf("Q4_1 high nibble element[%d] = %g, want %g", j+16, manual[j+16], wantHi)
		}
	}

	// Round-trip accuracy check on two asymmetric blocks (positive-shifted and negative-shifted).
	var input = make([]float32, 64)
	for i := range 32 {
		input[i] = 10.0 + float32(i)*0.125 // [10.0, 13.875] -> span 3.875, step ~0.258
	}
	for i := range 32 {
		input[32+i] = -6.0 + float32(i)*0.25 // [-6.0, 1.75] -> span 7.75, step ~0.517
	}
	var encoded, errQ = QuantizeQ4_1(input)
	requireError(t, errQ, nil)
	var decoded, errD = DequantizeQ4_1(encoded)
	requireError(t, errD, nil)
	for i, want := range input {
		var maxErr = 0.35
		if math.Abs(float64(decoded[i]-want)) > maxErr {
			t.Fatalf("Q4_1 index %d: got %g, want %g", i, decoded[i], want)
		}
	}

	// Overflow scale/min produces non-finite header without panicking or corrupting qs.
	var overflowBlock = make([]float32, 32)
	overflowBlock[0] = -1e6
	overflowBlock[1] = 1e6
	var ovEnc, errOv = QuantizeQ4_1(overflowBlock)
	requireError(t, errOv, nil)
	var ovDec, errOvD = DequantizeQ4_1(ovEnc)
	requireError(t, errOvD, nil)
	if !math.IsNaN(float64(ovDec[0])) && !math.IsInf(float64(ovDec[0]), 0) {
		t.Fatalf("overflow block dequant[0] = %g, want Inf or NaN", ovDec[0])
	}

	// NaN propagation in block.
	var nanBlock = make([]float32, 32)
	nanBlock[19] = float32(math.NaN())
	var nanEnc, errN = QuantizeQ4_1(nanBlock)
	requireError(t, errN, nil)
	var nanDec, errND = DequantizeQ4_1(nanEnc)
	requireError(t, errND, nil)
	if !math.IsNaN(float64(nanDec[19])) {
		t.Fatalf("NaN block did not propagate NaN: got %g", nanDec[19])
	}
}

func Test_QuantizedTensorValidationAndTransfers(t *testing.T) {
	type invalidCase struct {
		name  string
		shape []int64
		dtype DType
	}
	var (
		b            fakeBackend
		d            = newDevice(&b, 200)
		invalidCases = []invalidCase{
			{name: "q8 rank 1", shape: []int64{32}, dtype: Q8_0},
			{name: "q4 rank 1", shape: []int64{32}, dtype: Q4_0},
			{name: "q8 rank 3", shape: []int64{1, 1, 32}, dtype: Q8_0},
			{name: "q4 rank 3", shape: []int64{1, 1, 32}, dtype: Q4_0},
			{name: "q8 non-positive row", shape: []int64{0, 32}, dtype: Q8_0},
			{name: "q4 negative col", shape: []int64{2, -32}, dtype: Q4_0},
			{name: "q8 k not multiple of 32", shape: []int64{2, 31}, dtype: Q8_0},
			{name: "q8 k 33", shape: []int64{2, 33}, dtype: Q8_0},
			{name: "q4 k 16", shape: []int64{4, 16}, dtype: Q4_0},
			{name: "q4 k 48", shape: []int64{1, 48}, dtype: Q4_0},
			{name: "q8 over device limit", shape: []int64{6, 32}, dtype: Q8_0},  // 6*34 = 204 > 200
			{name: "q4 over device limit", shape: []int64{12, 32}, dtype: Q4_0}, // 12*18 = 216 > 200
			{name: "q8 overflow", shape: []int64{math.MaxInt64, 32}, dtype: Q8_0},
			{name: "q4 overflow", shape: []int64{math.MaxInt64, 32}, dtype: Q4_0},
			{name: "q4_1 rank 1", shape: []int64{32}, dtype: Q4_1},
			{name: "q4_1 rank 3", shape: []int64{1, 1, 32}, dtype: Q4_1},
			{name: "q4_1 negative col", shape: []int64{2, -32}, dtype: Q4_1},
			{name: "q4_1 k 48", shape: []int64{1, 48}, dtype: Q4_1},
			{name: "q4_1 over device limit", shape: []int64{11, 32}, dtype: Q4_1}, // 11*20 = 220 > 200
			{name: "q4_1 overflow", shape: []int64{math.MaxInt64, 32}, dtype: Q4_1},
		}
	)
	for _, tt := range invalidCases {
		t.Run(tt.name, func(t *testing.T) {
			var out, err = d.NewTensorWithDType(tt.shape, tt.dtype)
			if out != nil || !errors.Is(err, ErrInvalid) {
				t.Fatalf("NewTensorWithDType(%v, %v) = %v, %v, want nil and ErrInvalid", tt.shape, tt.dtype, out, err)
			}
		})
	}

	// Exact valid sizes within 200-byte limit.
	var (
		q8  = requireTensorWithDType(t, d, Q8_0, 2, 32)
		q4  = requireTensorWithDType(t, d, Q4_0, 2, 32)
		q41 = requireTensorWithDType(t, d, Q4_1, 2, 32)
	)
	if q8.Bytes() != 2*QuantBlockBytesQ8_0 || q4.Bytes() != 2*QuantBlockBytesQ4_0 || q41.Bytes() != 2*QuantBlockBytesQ4_1 {
		t.Fatalf("quantized bytes = (%d, %d, %d), want (%d, %d, %d)",
			q8.Bytes(), q4.Bytes(), q41.Bytes(), 2*QuantBlockBytesQ8_0, 2*QuantBlockBytesQ4_0, 2*QuantBlockBytesQ4_1)
	}

	// Upload/DownloadInto and UploadBits/DownloadBitsInto reject Q8_0/Q4_0/Q4_1.
	requireError(t, q8.Upload(make([]float32, 64)), ErrInvalid)
	requireError(t, q4.Upload(make([]float32, 64)), ErrInvalid)
	requireError(t, q41.Upload(make([]float32, 64)), ErrInvalid)
	requireError(t, q8.DownloadInto(make([]float32, 64)), ErrInvalid)
	requireError(t, q4.DownloadInto(make([]float32, 64)), ErrInvalid)
	requireError(t, q41.DownloadInto(make([]float32, 64)), ErrInvalid)
	requireError(t, q8.UploadBits(make([]uint16, 64)), ErrInvalid)
	requireError(t, q4.UploadBits(make([]uint16, 64)), ErrInvalid)
	requireError(t, q41.UploadBits(make([]uint16, 64)), ErrInvalid)
	requireError(t, q8.DownloadBitsInto(make([]uint16, 64)), ErrInvalid)
	requireError(t, q4.DownloadBitsInto(make([]uint16, 64)), ErrInvalid)
	requireError(t, q41.DownloadBitsInto(make([]uint16, 64)), ErrInvalid)

	// Uninitialized and byte-length mismatch checks.
	requireError(t, q8.DownloadBytesInto(make([]byte, q8.Bytes())), ErrUninitialized)
	requireError(t, q4.DownloadBytesInto(make([]byte, q4.Bytes())), ErrUninitialized)
	requireError(t, q41.DownloadBytesInto(make([]byte, q41.Bytes())), ErrUninitialized)
	requireError(t, q8.UploadBytes(make([]byte, q8.Bytes()-1)), ErrInvalid)
	requireError(t, q4.UploadBytes(make([]byte, q4.Bytes()+1)), ErrInvalid)
	requireError(t, q41.UploadBytes(make([]byte, q41.Bytes()+1)), ErrInvalid)

	// UploadFloat32As -> DownloadBytesInto -> DownloadFloat32Into round-trip.
	var values = make([]float32, 64)
	for i := range values {
		values[i] = float32(i-32) * 0.125
	}
	var (
		wantQ8Bytes, _  = QuantizeQ8_0(values)
		wantQ8F32, _    = DequantizeQ8_0(wantQ8Bytes)
		wantQ4Bytes, _  = QuantizeQ4_0(values)
		wantQ4F32, _    = DequantizeQ4_0(wantQ4Bytes)
		wantQ41Bytes, _ = QuantizeQ4_1(values)
		wantQ41F32, _   = DequantizeQ4_1(wantQ41Bytes)
	)
	requireError(t, q8.UploadFloat32As(values), nil)
	requireError(t, q4.UploadFloat32As(values), nil)
	requireError(t, q41.UploadFloat32As(values), nil)

	var (
		gotQ8Bytes  = make([]byte, q8.Bytes())
		gotQ4Bytes  = make([]byte, q4.Bytes())
		gotQ41Bytes = make([]byte, q41.Bytes())
		gotQ8F32    = make([]float32, 64)
		gotQ4F32    = make([]float32, 64)
		gotQ41F32   = make([]float32, 64)
	)
	requireError(t, q8.DownloadBytesInto(gotQ8Bytes), nil)
	requireError(t, q4.DownloadBytesInto(gotQ4Bytes), nil)
	requireError(t, q41.DownloadBytesInto(gotQ41Bytes), nil)
	if !slices.Equal(gotQ8Bytes, wantQ8Bytes) {
		t.Fatalf("Q8_0 bytes mismatch")
	}
	if !slices.Equal(gotQ4Bytes, wantQ4Bytes) {
		t.Fatalf("Q4_0 bytes mismatch")
	}
	if !slices.Equal(gotQ41Bytes, wantQ41Bytes) {
		t.Fatalf("Q4_1 bytes mismatch")
	}
	requireError(t, q8.DownloadFloat32Into(gotQ8F32), nil)
	requireError(t, q4.DownloadFloat32Into(gotQ4F32), nil)
	requireError(t, q41.DownloadFloat32Into(gotQ41F32), nil)
	if !slices.Equal(gotQ8F32, wantQ8F32) {
		t.Fatalf("Q8_0 dequantized float32 mismatch")
	}
	if !slices.Equal(gotQ4F32, wantQ4F32) {
		t.Fatalf("Q4_0 dequantized float32 mismatch")
	}
	if !slices.Equal(gotQ41F32, wantQ41F32) {
		t.Fatalf("Q4_1 dequantized float32 mismatch")
	}
}

func Test_ScopeQuantizedOpValidation(t *testing.T) {
	var (
		ctx   = context.Background()
		b     fakeBackend
		d     = newDevice(&b, 65536)
		f32L  = requireTensorWithDType(t, d, Float32, 1, 32)
		f32R  = requireTensorWithDType(t, d, Float32, 2, 32)
		f32D  = requireTensorWithDType(t, d, Float32, 1, 2)
		f32W  = requireTensorWithDType(t, d, Float32, 32)
		f32C  = requireTensorWithDType(t, d, Float32, 2, 16)
		q8A   = requireTensorWithDType(t, d, Q8_0, 2, 32)
		q8B   = requireTensorWithDType(t, d, Q8_0, 2, 32)
		q4A   = requireTensorWithDType(t, d, Q4_0, 2, 32)
		q4B   = requireTensorWithDType(t, d, Q4_0, 2, 32)
		q41A  = requireTensorWithDType(t, d, Q4_1, 2, 32)
		q41B  = requireTensorWithDType(t, d, Q4_1, 2, 32)
		q8Sq  = requireTensorWithDType(t, d, Q8_0, 32, 32)
		q4Sq  = requireTensorWithDType(t, d, Q4_0, 32, 32)
		q41Sq = requireTensorWithDType(t, d, Q4_1, 32, 32)
	)
	requireError(t, f32L.Upload(make([]float32, 32)), nil)
	requireError(t, f32R.Upload(make([]float32, 64)), nil)
	requireError(t, f32W.Upload(make([]float32, 32)), nil)
	requireError(t, f32C.Upload(make([]float32, 32)), nil)
	requireError(t, q8A.UploadFloat32As(make([]float32, 64)), nil)
	requireError(t, q4A.UploadFloat32As(make([]float32, 64)), nil)
	requireError(t, q41A.UploadFloat32As(make([]float32, 64)), nil)
	requireError(t, q8Sq.UploadFloat32As(make([]float32, 32*32)), nil)
	requireError(t, q4Sq.UploadFloat32As(make([]float32, 32*32)), nil)
	requireError(t, q41Sq.UploadFloat32As(make([]float32, 32*32)), nil)

	// Same-dtype Copy, GatherRows (same-dtype and dequantizing into Float32),
	// and MatMulRightTranspose succeed for Q8_0, Q4_0, and Q4_1.
	var (
		f16B   = requireTensorWithDType(t, d, Float16, 2, 32)
		bf16B  = requireTensorWithDType(t, d, BFloat16, 2, 32)
		f32G8  = requireTensor(t, d, 2, 32)
		f32G4  = requireTensor(t, d, 2, 32)
		f32G41 = requireTensor(t, d, 2, 32)
		q8Dst  = requireTensorWithDType(t, d, Q8_0, 2, 32)
		q4Dst  = requireTensorWithDType(t, d, Q4_0, 2, 32)
		q41Dst = requireTensorWithDType(t, d, Q4_1, 2, 32)
		scope  = requireScope(t, d)
	)
	requireError(t, f16B.UploadFloat32As(make([]float32, 64)), nil)
	requireError(t, bf16B.UploadFloat32As(make([]float32, 64)), nil)
	requireError(t, scope.Copy(q8A, q8B), nil)
	requireError(t, scope.Copy(q4A, q4B), nil)
	requireError(t, scope.Copy(q41A, q41B), nil)
	requireError(t, scope.GatherRows(q8A, []int64{1, 0}, q8Dst), nil)
	requireError(t, scope.GatherRows(q4A, []int64{1, 0}, q4Dst), nil)
	requireError(t, scope.GatherRows(q41A, []int64{1, 0}, q41Dst), nil)
	requireError(t, scope.GatherRows(q8A, []int64{1, 0}, f32G8), nil)
	requireError(t, scope.GatherRows(q4A, []int64{1, 0}, f32G4), nil)
	requireError(t, scope.GatherRows(q41A, []int64{1, 0}, f32G41), nil)
	requireError(t, scope.MatMulRightTranspose(f32L, q8A, f32D), nil)
	requireError(t, scope.MatMulRightTranspose(f32L, q4A, f32D), nil)
	requireError(t, scope.MatMulRightTranspose(f32L, q41A, f32D), nil)
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)

	// NewKVCacheWithDType rejects Q8_0, Q4_0, and Q4_1.
	for _, dtype := range []DType{Q8_0, Q4_0, Q4_1} {
		var cache, err = NewKVCacheWithDType(d, 4, 1, 32, dtype)
		if cache != nil || !errors.Is(err, ErrInvalid) {
			t.Fatalf("NewKVCacheWithDType(%v) = %v, %v, want nil and ErrInvalid", dtype, cache, err)
		}
	}

	var cache, cacheErr = NewKVCache(d, 4, 1, 32)
	requireError(t, cacheErr, nil)
	t.Cleanup(func() { requireError(t, cache.Close(), nil) })

	// All unsupported ops reject Q8_0 / Q4_0 / Q4_1 with ErrInvalid.
	type opCase struct {
		name string
		call func(s *Scope) error
	}
	var cases = []opCase{
		{name: "Copy q8->q4", call: func(s *Scope) error { return s.Copy(q8A, q4B) }},
		{name: "Copy q4_1->q4", call: func(s *Scope) error { return s.Copy(q41A, q4B) }},
		{name: "Copy q8->f32", call: func(s *Scope) error { return s.Copy(q8A, f32R) }},
		{name: "Fill q8", call: func(s *Scope) error { return s.Fill(q8B, 1) }},
		{name: "Fill q4", call: func(s *Scope) error { return s.Fill(q4B, 1) }},
		{name: "Fill q4_1", call: func(s *Scope) error { return s.Fill(q41B, 1) }},
		{name: "MatMul non-transposed q8", call: func(s *Scope) error { return s.MatMul(f32L, q8Sq, f32L) }},
		{name: "MatMul non-transposed q4", call: func(s *Scope) error { return s.MatMul(f32L, q4Sq, f32L) }},
		{name: "MatMul non-transposed q4_1", call: func(s *Scope) error { return s.MatMul(f32L, q41Sq, f32L) }},
		{name: "MatMulRightTranspose q8 left", call: func(s *Scope) error { return s.MatMulRightTranspose(q8A, f32R, f32D) }},
		{name: "MatMulRightTranspose q4 dst", call: func(s *Scope) error { return s.MatMulRightTranspose(f32R, q4Sq, q4B) }},
		{name: "MatMulRightTranspose q4_1 left", call: func(s *Scope) error { return s.MatMulRightTranspose(q41A, f32R, f32D) }},
		{name: "MatMulRightTranspose q4_1 dst", call: func(s *Scope) error { return s.MatMulRightTranspose(f32R, q41Sq, q41B) }},
		{name: "GatherRows q8->q4", call: func(s *Scope) error { return s.GatherRows(q8A, []int64{0, 1}, q4B) }},
		{name: "GatherRows q4->q8", call: func(s *Scope) error { return s.GatherRows(q4A, []int64{0, 1}, q8B) }},
		{name: "GatherRows q4_1->q4", call: func(s *Scope) error { return s.GatherRows(q41A, []int64{0, 1}, q4B) }},
		{name: "GatherRows q8->f16", call: func(s *Scope) error { return s.GatherRows(q8A, []int64{0, 1}, f16B) }},
		{name: "GatherRows q8->bf16", call: func(s *Scope) error { return s.GatherRows(q8A, []int64{0, 1}, bf16B) }},
		{name: "GatherRows q4->f16", call: func(s *Scope) error { return s.GatherRows(q4A, []int64{0, 1}, f16B) }},
		{name: "GatherRows q4->bf16", call: func(s *Scope) error { return s.GatherRows(q4A, []int64{0, 1}, bf16B) }},
		{name: "GatherRows q4_1->f16", call: func(s *Scope) error { return s.GatherRows(q41A, []int64{0, 1}, f16B) }},
		{name: "GatherRows q4_1->bf16", call: func(s *Scope) error { return s.GatherRows(q41A, []int64{0, 1}, bf16B) }},
		{name: "GatherRows f32->q8", call: func(s *Scope) error { return s.GatherRows(f32R, []int64{0, 1}, q8B) }},
		{name: "GatherRows f32->q4", call: func(s *Scope) error { return s.GatherRows(f32R, []int64{0, 1}, q4B) }},
		{name: "GatherRows f32->q4_1", call: func(s *Scope) error { return s.GatherRows(f32R, []int64{0, 1}, q41B) }},
		{name: "GatherRows f16->q8", call: func(s *Scope) error { return s.GatherRows(f16B, []int64{0, 1}, q8B) }},
		{name: "GatherRows bf16->q4", call: func(s *Scope) error { return s.GatherRows(bf16B, []int64{0, 1}, q4B) }},
		{name: "Add q8", call: func(s *Scope) error { return s.Add(q8A, q8A, q8B) }},
		{name: "Mul q4", call: func(s *Scope) error { return s.Mul(q4A, q4A, q4B) }},
		{name: "SiLU q8", call: func(s *Scope) error { return s.SiLU(q8A, q8B) }},
		{name: "SwiGLU q4", call: func(s *Scope) error { return s.SwiGLU(q4A, q4A, q4B) }},
		{name: "RMSNorm q8 input", call: func(s *Scope) error { return s.RMSNorm(q8A, f32W, f32R, 1e-5) }},
		{name: "RotatePairs q4 input", call: func(s *Scope) error { return s.RotatePairs(q4A, f32C, f32C, f32R, 32, 0, 0) }},
		{name: "AppendKV q8 keys", call: func(s *Scope) error { _, err := s.AppendKV(cache, q8A, f32R, 0); return err }},
		{name: "CausalAttention q4 queries", call: func(s *Scope) error { return s.CausalAttention(q4A, cache, f32R, 0) }},
		{name: "Add q4_1", call: func(s *Scope) error { return s.Add(q41A, q41A, q41B) }},
		{name: "RMSNorm q4_1 input", call: func(s *Scope) error { return s.RMSNorm(q41A, f32W, f32R, 1e-5) }},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			var s = requireScope(t, d)
			requireError(t, tt.call(s), ErrInvalid)
			requireError(t, s.Close(), nil)
		})
	}
}
