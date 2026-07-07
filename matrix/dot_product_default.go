//go:build (!arm64 && !amd64) || purego

package matrix

func dotProduct(left, right []float64) (sum float64) {
	sum = dotProductPure(left, right)
	return sum
}
