//go:build arm64 && !purego

package matrix

import simd64 "github.com/tphakala/simd/f64"

const elementwiseSIMDMinLength = 16

//go:noescape
func addIntoArm64(left, right, result []float64)

//go:noescape
func addScaledInPlaceArm64(left, right []float64, scale float64)

//go:noescape
func subtractIntoArm64(left, right, result []float64)

//go:noescape
func addScalarIntoArm64(source []float64, value float64, result []float64)

func addInto(left, right, result []float64) {
	if len(left) < elementwiseSIMDMinLength {
		addIntoPure(left, right, result)
		return
	}

	addIntoArm64(left, right, result)
}

func addScaledInPlace(left, right []float64, scale float64) {
	if len(left) < elementwiseSIMDMinLength {
		addScaledInPlacePure(left, right, scale)
		return
	}

	addScaledInPlaceArm64(left, right, scale)
}

func subtractInto(left, right, result []float64) {
	if len(left) < elementwiseSIMDMinLength {
		subtractIntoPure(left, right, result)
		return
	}

	subtractIntoArm64(left, right, result)
}

func multiplyElementsInto(left, right, result []float64) {
	simd64.Mul(result, left, right)
}

func addScalarInto(source []float64, value float64, result []float64) {
	if len(source) < elementwiseSIMDMinLength {
		addScalarIntoPure(source, value, result)
		return
	}

	addScalarIntoArm64(source, value, result)
}

func multiplyScalarInto(source []float64, value float64, result []float64) {
	simd64.Scale(result, source, value)
}

func multiplyScalarInPlace(source []float64, value float64) {
	simd64.Scale(source, source, value)
}
