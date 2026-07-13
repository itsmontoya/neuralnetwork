//go:build darwin && cgo && metal && !purego

package matrix

func dotProduct(left, right []float32) (sum float32) {
	sum = dotProductPure(left, right)
	return sum
}
