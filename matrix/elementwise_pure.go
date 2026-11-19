package matrix

func addScaledInPlacePure(left, right []float32, scale float32) {
	var index int

	for index = range left {
		left[index] += scale * right[index]
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
