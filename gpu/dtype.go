package gpu

import "math"

// DType identifies the element storage representation of a Tensor.
// The zero value is invalid and must not be used.
type DType uint8

const (
	// Float32 is contiguous IEEE-754 binary32 storage (4 bytes per element).
	Float32 DType = iota + 1
	// Float16 is contiguous IEEE-754 binary16 storage (2 bytes per element).
	Float16
	// BFloat16 is contiguous 16-bit brain floating-point storage (top 16 bits of
	// IEEE-754 binary32, 2 bytes per element).
	BFloat16
	// Q8_0 is block-quantized 8-bit storage (34 bytes per 32 elements: 2-byte
	// IEEE-754 binary16 scale followed by 32 signed int8 values).
	Q8_0
	// Q4_0 is block-quantized 4-bit storage (18 bytes per 32 elements: 2-byte
	// IEEE-754 binary16 scale followed by 16 packed uint8 nibble pairs in GGUF
	// split-half order, offset by -8).
	Q4_0
)

// Valid reports whether d is a supported tensor element type.
func (d DType) Valid() (ok bool) {
	return d == Float32 || d == Float16 || d == BFloat16 || d == Q8_0 || d == Q4_0
}

// IsQuantized reports whether d is a block-quantized storage format.
func (d DType) IsQuantized() (ok bool) {
	return d == Q8_0 || d == Q4_0
}

// Bytes returns the storage width in bytes of one element of d, or zero if d is
// invalid or block-quantized (use BlockSize and BlockBytes for quantized types).
func (d DType) Bytes() (size uint64) {
	switch d {
	case Float32:
		return 4
	case Float16, BFloat16:
		return 2
	default:
		return 0
	}
}

// BlockSize returns the number of logical elements per storage block (1 for
// scalar floating-point dtypes, 32 for Q8_0 and Q4_0), or zero if d is invalid.
func (d DType) BlockSize() (size uint64) {
	switch d {
	case Float32, Float16, BFloat16:
		return 1
	case Q8_0, Q4_0:
		return QuantBlockSize
	default:
		return 0
	}
}

// BlockBytes returns the byte size of one storage block of d, or zero if d is invalid.
func (d DType) BlockBytes() (size uint64) {
	switch d {
	case Float32:
		return 4
	case Float16, BFloat16:
		return 2
	case Q8_0:
		return QuantBlockBytesQ8_0
	case Q4_0:
		return QuantBlockBytesQ4_0
	default:
		return 0
	}
}

// String returns the canonical lowercase name of d, or "invalid" for unsupported values.
func (d DType) String() (name string) {
	switch d {
	case Float32:
		return "float32"
	case Float16:
		return "float16"
	case BFloat16:
		return "bfloat16"
	case Q8_0:
		return "q8_0"
	case Q4_0:
		return "q4_0"
	default:
		return "invalid"
	}
}

func (d DType) is16Bit() (ok bool) {
	return d == Float16 || d == BFloat16
}

// Float32ToFloat16 converts an IEEE-754 binary32 value to IEEE-754 binary16
// bits using round-to-nearest-even. Overflow rounds to signed infinity,
// underflow rounds to signed zero, and NaNs preserve sign while setting the
// quiet NaN bit so no NaN collapses to infinity.
func Float32ToFloat16(value float32) (bits uint16) {
	var (
		raw  = math.Float32bits(value)
		sign = uint16((raw >> 16) & 0x8000)
		exp  = int((raw >> 23) & 0xff)
		mant = raw & 0x007fffff
	)

	if exp == 0xff {
		if mant == 0 {
			return sign | 0x7c00
		}
		return sign | 0x7e00 | uint16(mant>>13)
	}
	if exp == 0 {
		return sign
	}

	var halfExp = exp - 112
	if halfExp >= 31 {
		return sign | 0x7c00
	}
	if halfExp >= 1 {
		var (
			roundBit = (mant >> 13) & 1
			rounded  = mant + 0x0fff + roundBit
		)
		if rounded&0x00800000 != 0 {
			rounded = 0
			halfExp++
			if halfExp >= 31 {
				return sign | 0x7c00
			}
		}
		return sign | uint16(halfExp<<10) | uint16(rounded>>13)
	}
	if halfExp < -10 {
		return sign
	}

	var (
		sig   = uint32(0x00800000) | mant
		shift = uint(14 - halfExp)
		half  = uint32(1) << (shift - 1)
		mask  = (uint32(1) << shift) - 1
		rem   = sig & mask
		quot  = sig >> shift
	)
	if rem > half || (rem == half && (quot&1) != 0) {
		quot++
	}
	return sign | uint16(quot)
}

// Float16ToFloat32 converts IEEE-754 binary16 bits to an exact binary32 value.
// NaNs are promoted to quiet binary32 NaNs preserving sign and payload bits.
func Float16ToFloat32(bits uint16) (value float32) {
	var (
		sign = uint32(bits&0x8000) << 16
		exp  = uint32((bits >> 10) & 0x1f)
		mant = uint32(bits & 0x03ff)
	)

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		var floatExp uint32 = 113
		for mant&0x0400 == 0 {
			mant <<= 1
			floatExp--
		}
		mant &= 0x03ff
		return math.Float32frombits(sign | (floatExp << 23) | (mant << 13))
	}
	if exp == 31 {
		if mant == 0 {
			return math.Float32frombits(sign | 0x7f800000)
		}
		return math.Float32frombits(sign | 0x7fc00000 | (mant << 13))
	}

	var floatExp = exp + 112
	return math.Float32frombits(sign | (floatExp << 23) | (mant << 13))
}

// Float32ToBFloat16 converts an IEEE-754 binary32 value to BFloat16 bits using
// round-to-nearest-even. Overflow rounds to signed infinity, signed zero and
// subnormals follow IEEE-754 RNE, and NaNs preserve sign while setting the
// quiet NaN bit so no NaN collapses to infinity.
func Float32ToBFloat16(value float32) (bits uint16) {
	var raw = math.Float32bits(value)
	if raw&0x7fffffff > 0x7f800000 {
		return uint16(raw>>16) | 0x0040
	}
	var bias = uint32(0x7fff) + ((raw >> 16) & 1)
	return uint16((raw + bias) >> 16)
}

// BFloat16ToFloat32 converts BFloat16 bits to an exact IEEE-754 binary32 value.
func BFloat16ToFloat32(bits uint16) (value float32) {
	return math.Float32frombits(uint32(bits) << 16)
}
