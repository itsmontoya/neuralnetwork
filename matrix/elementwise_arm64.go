//go:build arm64 && !purego

package matrix

import simd32 "github.com/tphakala/simd/f32"

const addScalarSIMDMinLength = 16

func addScaledInPlace(left, right []float32, scale float32) {
	addScaledInPlacePure(left, right, scale)
}

func addScalarInto(source []float32, value float32, result []float32) {
	if len(source) < addScalarSIMDMinLength {
		addScalarIntoPure(source, value, result)
		return
	}

	simd32.AddScalar(result, source, value)
}

func multiplyScalarInto(source []float32, value float32, result []float32) {
	simd32.Scale(result, source, value)
}

func multiplyScalarInPlace(source []float32, value float32) {
	simd32.Scale(source, source, value)
}
