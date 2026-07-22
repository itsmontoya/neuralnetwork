//go:build amd64 && !purego

package matrix

import simd32 "github.com/tphakala/simd/f32"

const elementwiseSIMDMinLength = 16

func addInto(left, right, result []float32) {
	if len(left) < elementwiseSIMDMinLength {
		addIntoPure(left, right, result)
		return
	}

	simd32.Add(result, left, right)
}

func addScaledInPlace(left, right []float32, scale float32) {
	addScaledInPlacePure(left, right, scale)
}

func subtractInto(left, right, result []float32) {
	if len(left) < elementwiseSIMDMinLength {
		subtractIntoPure(left, right, result)
		return
	}

	simd32.Sub(result, left, right)
}

func multiplyElementsInto(left, right, result []float32) {
	simd32.Mul(result, left, right)
}

func addScalarInto(source []float32, value float32, result []float32) {
	if len(source) < elementwiseSIMDMinLength {
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
