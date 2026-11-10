//go:build !arm64 || purego

package tensor

func conv2DResidualChannels(input, weights, bias, destination *Tensor, batch, channel, row, start, offset int, config Conv2DConfig) (handled bool) {
	return false
}
