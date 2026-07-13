//go:build (!arm64 && !amd64) || purego

package matrix

func dotProduct(left, right []float32) (sum float32) {
	sum = dotProductPure(left, right)
	return sum
}
