//go:build arm64 && !purego

package tensor

// conv2DFourByFour shares an interior input span across four filters. The caller
// checks four valid columns, unit width stride, unit dilation and group bounds.
func conv2DFourByFour(input, weights, bias, destination *Tensor, batch, channel, row, start, offset int, config Conv2DConfig) (handled bool) {
	var (
		sums         [16]float32
		channels     = int(weights.shape[1])
		kernelH      = int(weights.shape[2])
		kernelW      = int(weights.shape[3])
		inputH       = int(input.shape[2])
		inputW       = int(input.shape[3])
		inputPlane   = inputH * inputW
		kernelPlane  = kernelH * kernelW
		filterSize   = channels * kernelPlane
		rowStart     = row*int(config.Strides[0]) - int(config.Pads[0])
		kernelStart  = max(0, -rowStart)
		kernelEnd    = min(kernelH, inputH-rowStart)
		groupStart   = channel / int(weights.shape[0]/config.Groups) * channels
		plane        = int(destination.shape[2] * destination.shape[3])
		lane, column int
	)

	// Clip rows before reading either payload: padded products must be skipped,
	// including those with infinite or NaN weights. Slices bound every tile load.
	if kernelStart < kernelEnd {
		var (
			rows       = kernelEnd - kernelStart
			inputBase  = (batch*int(input.shape[1])+groupStart)*inputPlane + (rowStart+kernelStart)*inputW + start
			weightBase = channel*filterSize + kernelStart*kernelW
			inputSpan  = (channels-1)*inputPlane + (rows-1)*inputW + kernelW + 3
			weightSpan = 3*filterSize + (channels-1)*kernelPlane + rows*kernelW
		)

		conv2DInteriorTile(input.floats[inputBase:inputBase+inputSpan], weights.floats[weightBase:weightBase+weightSpan],
			&sums, channels, rows, kernelW, inputPlane, inputW, kernelPlane)
	}
	for lane = 0; lane < 4; lane++ {
		var values = destination.floats[offset+lane*plane : offset+lane*plane+4]

		for column = 0; column < 4; column++ {
			var value = sums[lane*4+column]

			if bias != nil {
				value += bias.floats[channel+lane]
			}
			values[column] = value
		}
	}
	return true
}
