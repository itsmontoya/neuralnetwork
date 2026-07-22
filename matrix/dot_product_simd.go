//go:build (amd64 || arm64) && !purego

package matrix

import simd32 "github.com/tphakala/simd/f32"

func dotProduct(left, right []float32) (sum float32) {
	sum = simd32.DotProduct(left, right)
	return sum
}
