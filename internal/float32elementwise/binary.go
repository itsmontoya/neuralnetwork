// Package float32elementwise provides host kernels for validated float32 slices.
package float32elementwise

// AddPure writes the elementwise sum of equal-length slices using Go arithmetic.
func AddPure(left, right, result []float32) {
	var index int

	for index = range result {
		result[index] = left[index] + right[index]
	}
}

// SubtractPure writes the elementwise difference of equal-length slices using Go arithmetic.
func SubtractPure(left, right, result []float32) {
	var index int

	for index = range result {
		result[index] = left[index] - right[index]
	}
}

// MultiplyPure writes the elementwise product of equal-length slices using Go arithmetic.
func MultiplyPure(left, right, result []float32) {
	var index int

	for index = range result {
		result[index] = left[index] * right[index]
	}
}
