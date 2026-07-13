// Package f32 provides float32 wrappers around standard-library math helpers.
package f32

import "math"

// Abs returns the absolute value of value.
func Abs(value float32) (result float32) {
	result = float32(math.Abs(float64(value)))
	return result
}

// Erf returns the error function of value.
func Erf(value float32) (result float32) {
	result = float32(math.Erf(float64(value)))
	return result
}

// Exp returns e**value.
func Exp(value float32) (result float32) {
	result = float32(math.Exp(float64(value)))
	return result
}

// Inf returns positive or negative infinity according to sign.
func Inf(sign int) (value float32) {
	value = float32(math.Inf(sign))
	return value
}

// IsInf reports whether value is an infinity according to sign.
func IsInf(value float32, sign int) (ok bool) {
	ok = math.IsInf(float64(value), sign)
	return ok
}

// IsNaN reports whether value is not a number.
func IsNaN(value float32) (ok bool) {
	ok = math.IsNaN(float64(value))
	return ok
}

// Log returns the natural logarithm of value.
func Log(value float32) (result float32) {
	result = float32(math.Log(float64(value)))
	return result
}

// NaN returns an IEEE 754 not-a-number value.
func NaN() (value float32) {
	value = float32(math.NaN())
	return value
}

// Pow returns x**y.
func Pow(x, y float32) (result float32) {
	result = float32(math.Pow(float64(x), float64(y)))
	return result
}

// Sqrt returns the square root of value.
func Sqrt(value float32) (result float32) {
	result = float32(math.Sqrt(float64(value)))
	return result
}

// Tanh returns the hyperbolic tangent of value.
func Tanh(value float32) (result float32) {
	result = float32(math.Tanh(float64(value)))
	return result
}
