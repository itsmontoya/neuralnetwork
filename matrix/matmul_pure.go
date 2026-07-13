package matrix

func matMulIntoPure(left, right, result *Matrix) {
	var (
		index        int
		row          int
		col          int
		inner        int
		leftValue    float32
		leftOffset   int
		rightOffset  int
		resultOffset int
	)

	for index = range result.data {
		result.data[index] = 0
	}

	for row = 0; row < left.rows; row++ {
		resultOffset = row * result.cols
		leftOffset = row * left.cols
		for inner = 0; inner < left.cols; inner++ {
			leftValue = left.data[leftOffset+inner]
			rightOffset = inner * right.cols
			for col = 0; col < right.cols; col++ {
				result.data[resultOffset+col] += leftValue * right.data[rightOffset+col]
			}
		}
	}
}

func matMulLeftTransposeIntoPure(left, right, result *Matrix) {
	var (
		index        int
		row          int
		col          int
		inner        int
		leftValue    float32
		leftOffset   int
		rightOffset  int
		resultOffset int
	)

	for index = range result.data {
		result.data[index] = 0
	}

	for inner = 0; inner < left.rows; inner++ {
		leftOffset = inner * left.cols
		rightOffset = inner * right.cols
		for row = 0; row < left.cols; row++ {
			leftValue = left.data[leftOffset+row]
			resultOffset = row * result.cols
			for col = 0; col < right.cols; col++ {
				result.data[resultOffset+col] += leftValue * right.data[rightOffset+col]
			}
		}
	}
}

func matMulRightTransposeIntoPure(left, right, result *Matrix) {
	var (
		index        int
		row          int
		col          int
		inner        int
		leftValue    float32
		leftOffset   int
		rightOffset  int
		resultOffset int
	)

	for index = range result.data {
		result.data[index] = 0
	}

	for row = 0; row < left.rows; row++ {
		resultOffset = row * result.cols
		leftOffset = row * left.cols
		for inner = 0; inner < left.cols; inner++ {
			leftValue = left.data[leftOffset+inner]
			for col = 0; col < right.rows; col++ {
				rightOffset = col * right.cols
				result.data[resultOffset+col] += leftValue * right.data[rightOffset+inner]
			}
		}
	}
}
