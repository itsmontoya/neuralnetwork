package matrix

func dotProductPure(left, right []float32) (sum float32) {
	var index int

	for index = range left {
		sum += left[index] * right[index]
	}

	return sum
}
