//go:build (amd64 || arm64) && !purego

package matrix

import simd64 "github.com/tphakala/simd/f64"

func dotProduct(left, right []float64) (sum float64) {
	sum = simd64.DotProduct(left, right)
	return sum
}
