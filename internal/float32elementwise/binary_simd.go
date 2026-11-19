//go:build (arm64 || amd64) && !purego

package float32elementwise

import simd32 "github.com/tphakala/simd/f32"

const addSubtractSIMDMinLength = 16

// Add writes the elementwise sum of equal-length slices.
func Add(left, right, result []float32) {
	if len(left) < addSubtractSIMDMinLength {
		AddPure(left, right, result)
		return
	}
	simd32.Add(result, left, right)
}

// Subtract writes the elementwise difference of equal-length slices.
func Subtract(left, right, result []float32) {
	if len(left) < addSubtractSIMDMinLength {
		SubtractPure(left, right, result)
		return
	}
	simd32.Sub(result, left, right)
}

// Multiply writes the elementwise product of equal-length slices.
func Multiply(left, right, result []float32) {
	simd32.Mul(result, left, right)
}
