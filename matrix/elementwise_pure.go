package matrix

func addIntoPure(left, right, result []float64) {
	var index int

	for index = range result {
		result[index] = left[index] + right[index]
	}
}

func addScaledInPlacePure(left, right []float64, scale float64) {
	var index int

	for index = range left {
		left[index] += scale * right[index]
	}
}

func subtractIntoPure(left, right, result []float64) {
	var index int

	for index = range result {
		result[index] = left[index] - right[index]
	}
}

func multiplyElementsIntoPure(left, right, result []float64) {
	var index int

	for index = range result {
		result[index] = left[index] * right[index]
	}
}

func addScalarIntoPure(source []float64, value float64, result []float64) {
	var index int

	for index = range result {
		result[index] = source[index] + value
	}
}

func multiplyScalarIntoPure(source []float64, value float64, result []float64) {
	var index int

	for index = range result {
		result[index] = source[index] * value
	}
}

func multiplyScalarInPlacePure(source []float64, value float64) {
	var index int

	for index = range source {
		source[index] *= value
	}
}
