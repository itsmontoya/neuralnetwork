// Package float32product shares CPU matrix-product loops with the private
// tensor adapter. Callers validate dimensions, storage bounds, and disjointness.
// Inputs are read-only, output is overwritten, and no storage is retained.
package float32product

// Standard multiplies row-major [rows, inner] and [inner, cols] blocks.
func Standard(left, right, output []float32, rows, inner, cols int) {
	var row, column, contraction, leftOffset, rightOffset, outputOffset int
	clear(output)
	for row = 0; row < rows; row++ {
		outputOffset = row * cols
		leftOffset = row * inner
		for contraction = 0; contraction < inner; contraction++ {
			leftValue := left[leftOffset+contraction]
			rightOffset = contraction * cols
			for column = 0; column < cols; column++ {
				output[outputOffset+column] += leftValue * right[rightOffset+column]
			}
		}
	}
}

// LeftTranspose multiplies transposed [inner, rows] by [inner, cols].
func LeftTranspose(left, right, output []float32, rows, inner, cols int) {
	var row, column, contraction, leftOffset, rightOffset, outputOffset int
	clear(output)
	for contraction = 0; contraction < inner; contraction++ {
		leftOffset = contraction * rows
		rightOffset = contraction * cols
		for row = 0; row < rows; row++ {
			leftValue := left[leftOffset+row]
			outputOffset = row * cols
			for column = 0; column < cols; column++ {
				output[outputOffset+column] += leftValue * right[rightOffset+column]
			}
		}
	}
}

// RightTranspose multiplies [rows, inner] by transposed [cols, inner].
func RightTranspose(left, right, output []float32, rows, inner, cols int) {
	var row, column, contraction, leftOffset, rightOffset, outputOffset int
	clear(output)
	for row = 0; row < rows; row++ {
		outputOffset = row * cols
		leftOffset = row * inner
		for contraction = 0; contraction < inner; contraction++ {
			leftValue := left[leftOffset+contraction]
			for column = 0; column < cols; column++ {
				rightOffset = column * inner
				output[outputOffset+column] += leftValue * right[rightOffset+contraction]
			}
		}
	}
}

// BothTranspose multiplies transposed [inner, rows] and [cols, inner].
func BothTranspose(left, right, output []float32, rows, inner, cols int) {
	var row, column, contraction, leftOffset, outputOffset int
	clear(output)
	for contraction = 0; contraction < inner; contraction++ {
		leftOffset = contraction * rows
		for row = 0; row < rows; row++ {
			leftValue := left[leftOffset+row]
			outputOffset = row * cols
			for column = 0; column < cols; column++ {
				output[outputOffset+column] += leftValue * right[column*inner+contraction]
			}
		}
	}
}
