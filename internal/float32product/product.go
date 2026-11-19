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

	if rows == 1 && cols >= rightTransposeColumnTile {
		var workers = rightTransposeOneRowWorkerCount(inner, cols)
		if workers > 1 {
			rightTransposeOneRowParallel(left[:inner], right, output[:cols], workers)
		} else {
			rightTransposeOneRow(left[:inner], right, output[:cols])
		}
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
			var sum float32
			rightRow := right[column*inner:][:inner]
			for contraction, rightValue := range rightRow {
				sum += leftRow[contraction] * rightValue
			}

			outputRow[column] = sum
		}
	}
}

// Reuse each input load across adjacent weight rows while preserving
// the ascending contraction order of each output.
func rightTransposeOneRow(left, right, output []float32) {
	var (
		inner  = len(left)
		column int
	)
	for ; column+rightTransposeColumnTile <= len(output); column += rightTransposeColumnTile {
		var (
			offset                 = column * inner
			weight0                = right[offset:][:inner]
			weight1                = right[offset+inner:][:inner]
			weight2                = right[offset+2*inner:][:inner]
			weight3                = right[offset+3*inner:][:inner]
			sum0, sum1, sum2, sum3 float32
		)
		for contraction, input := range left {
			sum0 += input * weight0[contraction]
			sum1 += input * weight1[contraction]
			sum2 += input * weight2[contraction]
			sum3 += input * weight3[contraction]
		}

		output[column] = sum0
		output[column+1] = sum1
		output[column+2] = sum2
		output[column+3] = sum3
	}

	for ; column < len(output); column++ {
		var sum float32
		weight := right[column*inner:][:inner]
		for contraction, rightValue := range weight {
			sum += left[contraction] * rightValue
		}

		output[column] = sum
	}
}

// Reuse weight loads across input rows and input loads across output columns.
// Each dot product accumulates in ascending contraction order.
func rightTransposeThreeRows(left, right, output []float32, inner, cols int) {
	var (
		left0  = left[:inner]
		left1  = left[inner:][:inner]
		left2  = left[2*inner:][:inner]
		out0   = output[:cols]
		out1   = output[cols:][:cols]
		out2   = output[2*cols:][:cols]
		column int
	)

	for ; column+4 <= cols; column += 4 {
		var (
			offset                     = column * inner
			weights0                   = right[offset:][:inner]
			weights1                   = right[offset+inner:][:inner]
			weights2                   = right[offset+2*inner:][:inner]
			weights3                   = right[offset+3*inner:][:inner]
			sum00, sum01, sum02, sum03 float32
			sum10, sum11, sum12, sum13 float32
			sum20, sum21, sum22, sum23 float32
		)
		for contraction, input0 := range left0 {
			var (
				input1  = left1[contraction]
				input2  = left2[contraction]
				weight0 = weights0[contraction]
				weight1 = weights1[contraction]
				weight2 = weights2[contraction]
				weight3 = weights3[contraction]
			)
			sum00 += input0 * weight0
			sum01 += input0 * weight1
			sum02 += input0 * weight2
			sum03 += input0 * weight3
			sum10 += input1 * weight0
			sum11 += input1 * weight1
			sum12 += input1 * weight2
			sum13 += input1 * weight3
			sum20 += input2 * weight0
			sum21 += input2 * weight1
			sum22 += input2 * weight2
			sum23 += input2 * weight3
		}

		out0[column] = sum00
		out0[column+1] = sum01
		out0[column+2] = sum02
		out0[column+3] = sum03
		out1[column] = sum10
		out1[column+1] = sum11
		out1[column+2] = sum12
		out1[column+3] = sum13
		out2[column] = sum20
		out2[column+1] = sum21
		out2[column+2] = sum22
		out2[column+3] = sum23
	}

	for ; column < cols; column++ {
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
