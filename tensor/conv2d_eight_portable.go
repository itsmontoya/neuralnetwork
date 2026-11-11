//go:build !arm64 || purego || race

package tensor

// Race instrumentation changes Go's NaN operand selection in the fallback
// traversal. Keep its original control flow in that build as well as pure Go.
const conv2DHasEightFilterTile = false

func conv2DEightFilterRows(input, weights, bias, destination *Tensor, batch, channel int, config Conv2DConfig) (handled bool) {
	return false
}
