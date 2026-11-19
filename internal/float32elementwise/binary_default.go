//go:build purego || (!arm64 && !amd64)

package float32elementwise

// Add writes the elementwise sum of equal-length slices.
func Add(left, right, result []float32) {
	AddPure(left, right, result)
}

// Subtract writes the elementwise difference of equal-length slices.
func Subtract(left, right, result []float32) {
	SubtractPure(left, right, result)
}

// Multiply writes the elementwise product of equal-length slices.
func Multiply(left, right, result []float32) {
	MultiplyPure(left, right, result)
}
