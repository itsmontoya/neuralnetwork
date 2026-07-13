package matrix

func addIntoPure(left, right, result []float32) {
	var index int

	for index = range result {
		result[index] = left[index] + right[index]
	}
}

func addScaledInPlacePure(left, right []float32, scale float32) {
	var index int

	for index = range left {
		left[index] += scale * right[index]
	}
}

func subtractIntoPure(left, right, result []float32) {
	var index int

	for index = range result {
		result[index] = left[index] - right[index]
	}
}

func multiplyElementsIntoPure(left, right, result []float32) {
	var index int

	for index = range result {
		result[index] = left[index] * right[index]
	}
}

func addScalarIntoPure(source []float32, value float32, result []float32) {
	var index int

	for index = range result {
		result[index] = source[index] + value
	}
}

func multiplyScalarIntoPure(source []float32, value float32, result []float32) {
	var index int

	for index = range result {
		result[index] = source[index] * value
	}
}

func multiplyScalarInPlacePure(source []float32, value float32) {
	var index int

	for index = range source {
		source[index] *= value
	}
}
