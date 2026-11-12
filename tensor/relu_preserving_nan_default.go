//go:build !arm64 || purego

package tensor

// reluPreservingNaNPrefix leaves the entire interval to the portable loop.
// The compiler inlines this no-op so the fallback retains its original traversal.
func reluPreservingNaNPrefix(input, destination *Tensor, offset, end int) (hasNaN bool, next int) {
	return false, offset
}
