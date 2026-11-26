package gpu

import (
	"encoding/binary"
	"fmt"
	"math"
)

const (
	// QuantBlockSize is the number of logical elements encoded in each Q8_0,
	// Q4_0, or Q4_1 quantization block along the inner dimension.
	QuantBlockSize = 32
	// QuantBlockBytesQ8_0 is the encoded byte size of one Q8_0 block: 2-byte
	// little-endian IEEE-754 binary16 scale plus 32 signed int8 values.
	QuantBlockBytesQ8_0 = 34
	// QuantBlockBytesQ4_0 is the encoded byte size of one Q4_0 block: 2-byte
	// little-endian IEEE-754 binary16 scale plus 16 packed uint8 nibble pairs.
	QuantBlockBytesQ4_0 = 18
	// QuantBlockBytesQ4_1 is the encoded byte size of one Q4_1 block: 2-byte
	// little-endian IEEE-754 binary16 scale, 2-byte little-endian IEEE-754
	// binary16 minimum, plus 16 packed uint8 unsigned nibble pairs.
	QuantBlockBytesQ4_1 = 20
)

// QuantizeQ8_0 encodes contiguous FP32 values into GGUF-compatible Q8_0 blocks.
// The input length must be a positive multiple of QuantBlockSize (32).
func QuantizeQ8_0(values []float32) (encoded []byte, err error) {
	if len(values) == 0 || len(values)%QuantBlockSize != 0 {
		return nil, fmt.Errorf("%w: Q8_0 quantization length %d must be a positive multiple of %d", ErrInvalid, len(values), QuantBlockSize)
	}

	var blockCount = len(values) / QuantBlockSize
	encoded = make([]byte, blockCount*QuantBlockBytesQ8_0)
	for block := range blockCount {
		var (
			src    = values[block*QuantBlockSize : (block+1)*QuantBlockSize]
			dst    = encoded[block*QuantBlockBytesQ8_0 : (block+1)*QuantBlockBytesQ8_0]
			maxAbs float32
			hasNaN bool
		)
		for _, v := range src {
			if math.IsNaN(float64(v)) {
				hasNaN = true
				break
			}
			var abs = float32(math.Abs(float64(v)))
			if abs > maxAbs {
				maxAbs = abs
			}
		}
		if hasNaN {
			binary.LittleEndian.PutUint16(dst[0:2], Float32ToFloat16(float32(math.NaN())))
			continue
		}
		if maxAbs == 0 {
			continue
		}

		var (
			scaleBits = Float32ToFloat16(maxAbs / 127.0)
			scale     = Float16ToFloat32(scaleBits)
		)
		binary.LittleEndian.PutUint16(dst[0:2], scaleBits)
		if scale == 0 || math.IsInf(float64(scale), 0) || math.IsNaN(float64(scale)) {
			continue
		}

		var invScale = 1.0 / scale
		for i, v := range src {
			var scaled = math.RoundToEven(float64(v * invScale))
			if scaled > 127 {
				scaled = 127
			} else if scaled < -127 {
				scaled = -127
			}
			dst[2+i] = byte(int8(scaled))
		}
	}
	return encoded, nil
}

// DequantizeQ8_0 decodes GGUF-compatible Q8_0 blocks into FP32 values.
// The input byte length must be a positive multiple of QuantBlockBytesQ8_0 (34).
func DequantizeQ8_0(encoded []byte) (values []float32, err error) {
	if len(encoded) == 0 || len(encoded)%QuantBlockBytesQ8_0 != 0 {
		return nil, fmt.Errorf("%w: Q8_0 byte length %d must be a positive multiple of %d", ErrInvalid, len(encoded), QuantBlockBytesQ8_0)
	}
	values = make([]float32, (len(encoded)/QuantBlockBytesQ8_0)*QuantBlockSize)
	dequantizeQ8_0Into(encoded, values)
	return values, nil
}

func dequantizeQ8_0Into(encoded []byte, destination []float32) {
	var blockCount = len(encoded) / QuantBlockBytesQ8_0
	for block := range blockCount {
		var (
			src   = encoded[block*QuantBlockBytesQ8_0 : (block+1)*QuantBlockBytesQ8_0]
			dst   = destination[block*QuantBlockSize : (block+1)*QuantBlockSize]
			scale = Float16ToFloat32(binary.LittleEndian.Uint16(src[0:2]))
		)
		for i := range QuantBlockSize {
			dst[i] = scale * float32(int8(src[2+i]))
		}
	}
}

// QuantizeQ4_0 encodes contiguous FP32 values into GGUF-compatible Q4_0 blocks.
// Each block stores a 2-byte little-endian IEEE-754 binary16 scale followed by
// 16 packed uint8 bytes where byte j holds element j in its low nibble and
// element j+16 in its high nibble, each offset by -8. The input length must be
// a positive multiple of QuantBlockSize (32).
func QuantizeQ4_0(values []float32) (encoded []byte, err error) {
	if len(values) == 0 || len(values)%QuantBlockSize != 0 {
		return nil, fmt.Errorf("%w: Q4_0 quantization length %d must be a positive multiple of %d", ErrInvalid, len(values), QuantBlockSize)
	}

	var blockCount = len(values) / QuantBlockSize
	encoded = make([]byte, blockCount*QuantBlockBytesQ4_0)
	for block := range blockCount {
		var (
			src    = values[block*QuantBlockSize : (block+1)*QuantBlockSize]
			dst    = encoded[block*QuantBlockBytesQ4_0 : (block+1)*QuantBlockBytesQ4_0]
			maxAbs float32
			maxVal float32
			hasNaN bool
		)
		for _, v := range src {
			if math.IsNaN(float64(v)) {
				hasNaN = true
				break
			}
			var abs = float32(math.Abs(float64(v)))
			if abs > maxAbs {
				maxAbs = abs
				maxVal = v
			}
		}
		if hasNaN {
			binary.LittleEndian.PutUint16(dst[0:2], Float32ToFloat16(float32(math.NaN())))
			for j := range 16 {
				dst[2+j] = 0x88
			}
			continue
		}
		if maxAbs == 0 {
			for j := range 16 {
				dst[2+j] = 0x88
			}
			continue
		}

		var (
			scaleBits = Float32ToFloat16(maxVal / -8.0)
			scale     = Float16ToFloat32(scaleBits)
		)
		binary.LittleEndian.PutUint16(dst[0:2], scaleBits)
		if scale == 0 || math.IsInf(float64(scale), 0) || math.IsNaN(float64(scale)) {
			for j := range 16 {
				dst[2+j] = 0x88
			}
			continue
		}

		var invScale = 1.0 / scale
		for j := range 16 {
			var (
				lo = clampQ4Nibble(math.RoundToEven(float64(src[j]*invScale)) + 8)
				hi = clampQ4Nibble(math.RoundToEven(float64(src[j+16]*invScale)) + 8)
			)
			dst[2+j] = lo | (hi << 4)
		}
	}
	return encoded, nil
}

// DequantizeQ4_0 decodes GGUF-compatible Q4_0 blocks into FP32 values.
// The input byte length must be a positive multiple of QuantBlockBytesQ4_0 (18).
func DequantizeQ4_0(encoded []byte) (values []float32, err error) {
	if len(encoded) == 0 || len(encoded)%QuantBlockBytesQ4_0 != 0 {
		return nil, fmt.Errorf("%w: Q4_0 byte length %d must be a positive multiple of %d", ErrInvalid, len(encoded), QuantBlockBytesQ4_0)
	}
	values = make([]float32, (len(encoded)/QuantBlockBytesQ4_0)*QuantBlockSize)
	dequantizeQ4_0Into(encoded, values)
	return values, nil
}

func dequantizeQ4_0Into(encoded []byte, destination []float32) {
	var blockCount = len(encoded) / QuantBlockBytesQ4_0
	for block := range blockCount {
		var (
			src   = encoded[block*QuantBlockBytesQ4_0 : (block+1)*QuantBlockBytesQ4_0]
			dst   = destination[block*QuantBlockSize : (block+1)*QuantBlockSize]
			scale = Float16ToFloat32(binary.LittleEndian.Uint16(src[0:2]))
		)
		for j := range 16 {
			var packed = src[2+j]
			dst[j] = scale * float32(int(packed&0x0f)-8)
			dst[j+16] = scale * float32(int(packed>>4)-8)
		}
	}
}

// QuantizeQ4_1 encodes contiguous FP32 values into GGUF-compatible asymmetric
// Q4_1 blocks. Each 20-byte block stores a 2-byte little-endian IEEE-754
// binary16 scale d, a 2-byte little-endian IEEE-754 binary16 minimum m, and 16
// packed uint8 bytes where byte j holds unsigned element j in its low nibble
// and unsigned element j+16 in its high nibble (range 0..15). The input length
// must be a positive multiple of QuantBlockSize (32).
func QuantizeQ4_1(values []float32) (encoded []byte, err error) {
	if len(values) == 0 || len(values)%QuantBlockSize != 0 {
		return nil, fmt.Errorf("%w: Q4_1 quantization length %d must be a positive multiple of %d", ErrInvalid, len(values), QuantBlockSize)
	}

	var blockCount = len(values) / QuantBlockSize
	encoded = make([]byte, blockCount*QuantBlockBytesQ4_1)
	for block := range blockCount {
		var (
			src    = values[block*QuantBlockSize : (block+1)*QuantBlockSize]
			dst    = encoded[block*QuantBlockBytesQ4_1 : (block+1)*QuantBlockBytesQ4_1]
			minVal = src[0]
			maxVal = src[0]
			hasNaN bool
		)
		for _, v := range src {
			if math.IsNaN(float64(v)) {
				hasNaN = true
				break
			}
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
		if hasNaN {
			var nanBits = Float32ToFloat16(float32(math.NaN()))
			binary.LittleEndian.PutUint16(dst[0:2], nanBits)
			binary.LittleEndian.PutUint16(dst[2:4], nanBits)
			continue
		}

		var (
			minBits = Float32ToFloat16(minVal)
			minF32  = Float16ToFloat32(minBits)
		)
		binary.LittleEndian.PutUint16(dst[2:4], minBits)
		if maxVal == minVal {
			continue
		}

		var (
			span      = float64(maxVal) - float64(minVal)
			scaleBits = Float32ToFloat16(float32(span / 15.0))
			scale     = Float16ToFloat32(scaleBits)
		)
		binary.LittleEndian.PutUint16(dst[0:2], scaleBits)
		if scale == 0 || math.IsInf(float64(scale), 0) || math.IsNaN(float64(scale)) ||
			math.IsInf(float64(minF32), 0) || math.IsNaN(float64(minF32)) {
			continue
		}

		var invScale = 1.0 / scale
		for j := range 16 {
			var (
				lo = clampQ4Nibble(math.RoundToEven(float64((src[j] - minF32) * invScale)))
				hi = clampQ4Nibble(math.RoundToEven(float64((src[j+16] - minF32) * invScale)))
			)
			dst[4+j] = lo | (hi << 4)
		}
	}
	return encoded, nil
}

// DequantizeQ4_1 decodes GGUF-compatible Q4_1 blocks into FP32 values.
// The input byte length must be a positive multiple of QuantBlockBytesQ4_1 (20).
func DequantizeQ4_1(encoded []byte) (values []float32, err error) {
	if len(encoded) == 0 || len(encoded)%QuantBlockBytesQ4_1 != 0 {
		return nil, fmt.Errorf("%w: Q4_1 byte length %d must be a positive multiple of %d", ErrInvalid, len(encoded), QuantBlockBytesQ4_1)
	}
	values = make([]float32, (len(encoded)/QuantBlockBytesQ4_1)*QuantBlockSize)
	dequantizeQ4_1Into(encoded, values)
	return values, nil
}

func dequantizeQ4_1Into(encoded []byte, destination []float32) {
	var blockCount = len(encoded) / QuantBlockBytesQ4_1
	for block := range blockCount {
		var (
			src    = encoded[block*QuantBlockBytesQ4_1 : (block+1)*QuantBlockBytesQ4_1]
			dst    = destination[block*QuantBlockSize : (block+1)*QuantBlockSize]
			scale  = Float16ToFloat32(binary.LittleEndian.Uint16(src[0:2]))
			minVal = Float16ToFloat32(binary.LittleEndian.Uint16(src[2:4]))
		)
		for j := range 16 {
			var packed = src[4+j]
			// Explicit float32 conversion prevents FMA contraction so host
			// dequantization matches Metal fastMathEnabled=NO bit-for-bit.
			dst[j] = float32(scale*float32(packed&0x0f)) + minVal
			dst[j+16] = float32(scale*float32(packed>>4)) + minVal
		}
	}
}

func clampQ4Nibble(value float64) (nibble uint8) {
	if value < 0 {
		return 0
	}
	if value > 15 {
		return 15
	}
	return uint8(value)
}
