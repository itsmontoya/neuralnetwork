package matrix

func dotProductPure(left, right []float64) (sum float64) {
	var index int

	for index = range left {
		sum += left[index] * right[index]
	}

	return sum
}
