//go:build !arm64 || purego

package tensor

const conv2DHasInteriorTile = false

// conv2DFourByFour keeps portable builds on the existing spatial helpers,
// preserving their architecture-specific rounding and NaN operand priority.
func conv2DFourByFour(input, weights, bias, destination *Tensor, batch, channel, row, start, offset int, config Conv2DConfig) (handled bool) {
	return false
}
