//go:build arm64 && !purego

package matrix

func dotProduct(left, right []float64) (sum float64) {
	// Arm64 is the primary SIMD development path; keep this scalar fallback
	// until arm64 benchmark evidence proves an assembly kernel is stable.
	sum = dotProductPure(left, right)
	return sum
}
