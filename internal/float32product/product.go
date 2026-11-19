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
	if len(output) == 0 {
		return
	}

	if inner == 0 {
		clear(output)
		return
	}

	var row int
	for ; row+3 <= rows; row += 3 {
		rightTransposeThreeRows(left[row*inner:], right, output[row*cols:], inner, cols)
	}

	for ; row < rows; row++ {
		leftRow := left[row*inner:][:inner]
		outputRow := output[row*cols:][:cols]
		for column := range outputRow {
			rightRow := right[column*inner:][:inner]
			var sum float32
			for contraction, rightValue := range rightRow {
				sum += leftRow[contraction] * rightValue
			}
			outputRow[column] = sum
		}
	}
}

// Reuse each weight load across three input rows while each dot product
// accumulates in ascending contraction order.
func rightTransposeThreeRows(left, right, output []float32, inner, cols int) {
	var (
		left0 = left[:inner]
		left1 = left[inner:][:inner]
		left2 = left[2*inner:][:inner]
		out0  = output[:cols]
		out1  = output[cols:][:cols]
		out2  = output[2*cols:][:cols]
	)

	for column := range out0 {
		rightRow := right[column*inner:][:inner]
		var sum0, sum1, sum2 float32
		for contraction, rightValue := range rightRow {
			sum0 += left0[contraction] * rightValue
			sum1 += left1[contraction] * rightValue
			sum2 += left2[contraction] * rightValue
		}

		out0[column] = sum0
		out1[column] = sum1
		out2[column] = sum2
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
