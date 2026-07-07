//go:build amd64 && !purego

package matrix

func dotProduct(left, right []float64) (sum float64) {
	// Amd64 assembly should replace this scalar fallback only after amd64
	// benchmark evidence proves a stable win.
	sum = dotProductPure(left, right)
	return sum
}
