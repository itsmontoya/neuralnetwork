package float32product

// Share each weight load across both rows while accumulating in ascending order.
func rightTransposeTwoRows(left, right, output []float32, inner, stride, cols int) {
	var (
		left0  = left[:inner]
		left1  = left[inner:][:inner]
		out0   = output[:cols]
		out1   = output[stride:][:cols]
		column int
	)

	for ; column+rightTransposeColumnTile <= cols; column += rightTransposeColumnTile {
		var (
			offset                     = column * inner
			weights0                   = right[offset:][:inner]
			weights1                   = right[offset+inner:][:inner]
			weights2                   = right[offset+2*inner:][:inner]
			weights3                   = right[offset+3*inner:][:inner]
			sum00, sum01, sum02, sum03 float32
			sum10, sum11, sum12, sum13 float32
		)

		for contraction, input0 := range left0 {
			var (
				input1  = left1[contraction]
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
		}

		out0[column] = sum00
		out0[column+1] = sum01
		out0[column+2] = sum02
		out0[column+3] = sum03
		out1[column] = sum10
		out1[column+1] = sum11
		out1[column+2] = sum12
		out1[column+3] = sum13
	}

	for ; column < cols; column++ {
		rightRow := right[column*inner:][:inner]
		var sum0, sum1 float32
		for contraction, rightValue := range rightRow {
			sum0 += left0[contraction] * rightValue
			sum1 += left1[contraction] * rightValue
		}

		out0[column] = sum0
		out1[column] = sum1
	}
}
