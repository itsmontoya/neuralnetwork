package gpu

import (
	"math"
	"testing"
)

func Test_DTypeMetadata(t *testing.T) {
	type testcase struct {
		name  string
		dtype DType
		valid bool
		bytes uint64
		str   string
		is16  bool
	}

	var tests = []testcase{
		{name: "zero invalid", dtype: 0, valid: false, bytes: 0, str: "invalid", is16: false},
		{name: "float32", dtype: Float32, valid: true, bytes: 4, str: "float32", is16: false},
		{name: "float16", dtype: Float16, valid: true, bytes: 2, str: "float16", is16: true},
		{name: "bfloat16", dtype: BFloat16, valid: true, bytes: 2, str: "bfloat16", is16: true},
		{name: "out of range", dtype: DType(99), valid: false, bytes: 0, str: "invalid", is16: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.dtype.Valid(); got != tt.valid {
				t.Fatalf("Valid() = %t, want %t", got, tt.valid)
			}
			if got := tt.dtype.Bytes(); got != tt.bytes {
				t.Fatalf("Bytes() = %d, want %d", got, tt.bytes)
			}
			if got := tt.dtype.String(); got != tt.str {
				t.Fatalf("String() = %q, want %q", got, tt.str)
			}
			if got := tt.dtype.is16Bit(); got != tt.is16 {
				t.Fatalf("is16Bit() = %t, want %t", got, tt.is16)
			}
		})
	}
}

func Test_Float16ExhaustiveRoundTrip(t *testing.T) {
	for raw := 0; raw <= 0xffff; raw++ {
		var (
			bits  = uint16(raw)
			value = Float16ToFloat32(bits)
			back  = Float32ToFloat16(value)
		)

		var isNaN = (bits&0x7c00) == 0x7c00 && (bits&0x03ff) != 0
		if isNaN {
			if !math.IsNaN(float64(value)) {
				t.Fatalf("Float16ToFloat32(0x%04x) = %v, want NaN", bits, value)
			}
			if (back&0x7c00) != 0x7c00 || (back&0x03ff) == 0 {
				t.Fatalf("Float32ToFloat16(Float16ToFloat32(0x%04x)) = 0x%04x, want NaN", bits, back)
			}
			if (back & 0x8000) != (bits & 0x8000) {
				t.Fatalf("NaN sign lost for 0x%04x -> 0x%04x", bits, back)
			}
			continue
		}

		if back != bits {
			t.Fatalf("round-trip mismatch: bits=0x%04x -> f32=%g (0x%08x) -> bits=0x%04x",
				bits, value, math.Float32bits(value), back)
		}
	}
}

func Test_Float32ToFloat16EdgeCases(t *testing.T) {
	type testcase struct {
		name string
		in   float32
		want uint16
	}

	var tests = []testcase{
		{name: "pos zero", in: 0, want: 0x0000},
		{name: "neg zero", in: float32(math.Copysign(0, -1)), want: 0x8000},
		{name: "one", in: 1.0, want: 0x3c00},
		{name: "neg two", in: -2.0, want: 0xc000},
		{name: "max finite", in: 65504.0, want: 0x7bff},
		{name: "neg max finite", in: -65504.0, want: 0xfbff},
		{name: "below overflow tie", in: math.Float32frombits(math.Float32bits(65520.0) - 1), want: 0x7bff},
		{name: "overflow tie rounds to +inf", in: 65520.0, want: 0x7c00},
		{name: "neg overflow tie rounds to -inf", in: -65520.0, want: 0xfc00},
		{name: "large overflow +inf", in: 1e20, want: 0x7c00},
		{name: "large overflow -inf", in: -1e20, want: 0xfc00},
		{name: "pos inf", in: float32(math.Inf(1)), want: 0x7c00},
		{name: "neg inf", in: float32(math.Inf(-1)), want: 0xfc00},
		{name: "min positive normal", in: float32(math.Ldexp(1, -14)), want: 0x0400},
		{name: "max positive subnormal", in: float32(1023) * float32(math.Ldexp(1, -24)), want: 0x03ff},
		{name: "subnormal tie promotes to min normal", in: float32(1023.5) * float32(math.Ldexp(1, -24)), want: 0x0400},
		{name: "min positive subnormal", in: float32(math.Ldexp(1, -24)), want: 0x0001},
		{name: "half min subnormal ties to zero", in: float32(math.Ldexp(1, -25)), want: 0x0000},
		{name: "neg half min subnormal ties to neg zero", in: -float32(math.Ldexp(1, -25)), want: 0x8000},
		{name: "just above half min subnormal rounds up", in: math.Float32frombits(math.Float32bits(float32(math.Ldexp(1, -25))) + 1), want: 0x0001},
		{name: "float32 subnormal underflows to pos zero", in: math.Float32frombits(0x00000001), want: 0x0000},
		{name: "neg float32 subnormal underflows to neg zero", in: math.Float32frombits(0x80000001), want: 0x8000},
		{name: "normal RNE tie to even (1.0 + 0.5 ulp)", in: math.Float32frombits(0x3f801000), want: 0x3c00},
		{name: "normal RNE tie to odd (1.0 + 1.5 ulp)", in: math.Float32frombits(0x3f803000), want: 0x3c02},
		{name: "subnormal RNE tie to even (2 * 2^-24 + 0.5 ulp)", in: float32(2.5) * float32(math.Ldexp(1, -24)), want: 0x0002},
		{name: "subnormal RNE tie to odd (3 * 2^-24 + 0.5 ulp)", in: float32(3.5) * float32(math.Ldexp(1, -24)), want: 0x0004},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Float32ToFloat16(tt.in); got != tt.want {
				t.Fatalf("Float32ToFloat16(%g / 0x%08x) = 0x%04x, want 0x%04x",
					tt.in, math.Float32bits(tt.in), got, tt.want)
			}
		})
	}

	// Signaling NaN with payload only in low 13 bits must remain NaN, not collapse to Inf.
	var lowPayloadNaN = math.Float32frombits(0x7f800001)
	var gotNaN = Float32ToFloat16(lowPayloadNaN)
	if (gotNaN&0x7c00) != 0x7c00 || (gotNaN&0x03ff) == 0 {
		t.Fatalf("low-payload sNaN collapsed to 0x%04x", gotNaN)
	}
	var negLowPayloadNaN = math.Float32frombits(0xff800001)
	var gotNegNaN = Float32ToFloat16(negLowPayloadNaN)
	if (gotNegNaN&0xfc00) != 0xfc00 || (gotNegNaN&0x03ff) == 0 {
		t.Fatalf("negative low-payload sNaN collapsed to 0x%04x", gotNegNaN)
	}
}

func Test_BFloat16ExhaustiveRoundTrip(t *testing.T) {
	for raw := 0; raw <= 0xffff; raw++ {
		var (
			bits  = uint16(raw)
			value = BFloat16ToFloat32(bits)
			back  = Float32ToBFloat16(value)
		)

		var isNaN = (bits&0x7f80) == 0x7f80 && (bits&0x007f) != 0
		if isNaN {
			if !math.IsNaN(float64(value)) {
				t.Fatalf("BFloat16ToFloat32(0x%04x) = %v, want NaN", bits, value)
			}
			if (back&0x7f80) != 0x7f80 || (back&0x007f) == 0 {
				t.Fatalf("Float32ToBFloat16(BFloat16ToFloat32(0x%04x)) = 0x%04x, want NaN", bits, back)
			}
			if (back & 0x8000) != (bits & 0x8000) {
				t.Fatalf("NaN sign lost for 0x%04x -> 0x%04x", bits, back)
			}
			continue
		}

		if back != bits {
			t.Fatalf("round-trip mismatch: bits=0x%04x -> f32=%g (0x%08x) -> bits=0x%04x",
				bits, value, math.Float32bits(value), back)
		}
	}
}

func Test_Float32ToBFloat16EdgeCases(t *testing.T) {
	type testcase struct {
		name string
		in   uint32
		want uint16
	}

	var tests = []testcase{
		{name: "pos zero", in: 0x00000000, want: 0x0000},
		{name: "neg zero", in: 0x80000000, want: 0x8000},
		{name: "one", in: 0x3f800000, want: 0x3f80},
		{name: "neg two", in: 0xc0000000, want: 0xc000},
		{name: "pos inf", in: 0x7f800000, want: 0x7f80},
		{name: "neg inf", in: 0xff800000, want: 0xff80},
		{name: "RNE tie to even", in: 0x3f808000, want: 0x3f80},
		{name: "RNE tie to odd rounds up", in: 0x3f818000, want: 0x3f82},
		{name: "above tie rounds up", in: 0x3f808001, want: 0x3f81},
		{name: "below tie rounds down", in: 0x3f817fff, want: 0x3f81},
		{name: "max finite stays finite below tie", in: 0x7f7f7fff, want: 0x7f7f},
		{name: "max finite tie overflows to +inf", in: 0x7f7f8000, want: 0x7f80},
		{name: "max float32 overflows to +inf", in: 0x7f7fffff, want: 0x7f80},
		{name: "neg max float32 overflows to -inf", in: 0xff7fffff, want: 0xff80},
		{name: "subnormal tie to zero", in: 0x00008000, want: 0x0000},
		{name: "subnormal above tie rounds up", in: 0x00008001, want: 0x0001},
		{name: "subnormal tie promotes to min normal", in: 0x007f8000, want: 0x0080},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got = Float32ToBFloat16(math.Float32frombits(tt.in))
			if got != tt.want {
				t.Fatalf("Float32ToBFloat16(0x%08x) = 0x%04x, want 0x%04x", tt.in, got, tt.want)
			}
		})
	}

	// Signaling NaN with payload only in low 16 bits must remain NaN, not collapse to Inf.
	var gotNaN = Float32ToBFloat16(math.Float32frombits(0x7f800001))
	if (gotNaN&0x7f80) != 0x7f80 || (gotNaN&0x007f) == 0 {
		t.Fatalf("low-payload sNaN collapsed to 0x%04x", gotNaN)
	}
	var gotNegNaN = Float32ToBFloat16(math.Float32frombits(0xff800001))
	if (gotNegNaN&0xff80) != 0xff80 || (gotNegNaN&0x007f) == 0 {
		t.Fatalf("negative low-payload sNaN collapsed to 0x%04x", gotNegNaN)
	}
}
