//go:build !arm64 || purego

package tensor

func conv2DResidualTile(input, weights, bias, destination *Tensor, batch, channel, row, start, offset, width int, config Conv2DConfig) (handled bool) {
	return false
}
