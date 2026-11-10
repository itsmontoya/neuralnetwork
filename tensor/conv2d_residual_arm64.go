//go:build arm64 && !purego

package tensor

// conv2DResidualChannels clips a single column across four filters. The caller
// checks unit width stride, unit dilation and filter bounds within one group.
func conv2DResidualChannels(input, weights, bias, destination *Tensor, batch, channel, row, start, offset int, config Conv2DConfig) (handled bool) {
	var (
		sums        [4]float32
		channels    = int(weights.shape[1])
		kernelH     = int(weights.shape[2])
		kernelW     = int(weights.shape[3])
		inputH      = int(input.shape[2])
		inputW      = int(input.shape[3])
		inputPlane  = inputH * inputW
		kernelPlane = kernelH * kernelW
		filterSize  = channels * kernelPlane
		rowStart    = row*int(config.Strides[0]) - int(config.Pads[0])
		rowFirst    = max(0, -rowStart)
		rowEnd      = min(kernelH, inputH-rowStart)
		columnFirst = max(0, -start)
		columnEnd   = min(kernelW, inputW-start)
		groupStart  = channel / int(weights.shape[0]/config.Groups) * channels
		plane       = int(destination.shape[2] * destination.shape[3])
		lane        int
	)

	// Exclude padded products before forming spans, even for nonfinite weights.
	// The last scalar load in each channel and row lies inside these slices.
	if rowFirst < rowEnd && columnFirst < columnEnd {
		var (
			rows       = rowEnd - rowFirst
			columns    = columnEnd - columnFirst
			inputBase  = (batch*int(input.shape[1])+groupStart)*inputPlane + (rowStart+rowFirst)*inputW + start + columnFirst
			weightBase = channel*filterSize + rowFirst*kernelW + columnFirst
			inputSpan  = (channels-1)*inputPlane + (rows-1)*inputW + columns
			weightSpan = 3*filterSize + (channels-1)*kernelPlane + (rows-1)*kernelW + columns
		)

		conv2DResidualSums(input.floats[inputBase:inputBase+inputSpan], weights.floats[weightBase:weightBase+weightSpan],
			&sums, channels, rows, columns, inputPlane, inputW, kernelPlane, kernelW)
	}
	if bias != nil {
		conv2DResidualBias(&sums, bias.floats[channel:channel+4])
	}
	for lane = 0; lane < 4; lane++ {
		destination.floats[offset+lane*plane] = sums[lane]
	}
	return true
}

// conv2DResidualSums retains the scalar helper's fused reduction and input-first
// NaN operand priority. The caller bounds all spans and passes positive counts.
//
//go:noescape
func conv2DResidualSums(input, weights []float32, sums *[4]float32, channels, rows, columns, inputPlane, inputW, kernelPlane, kernelW int)

// conv2DResidualBias preserves the released four-filter caller's bias-first NaN
// operand priority. Go may commute addition even when its source is unchanged.
//
//go:noescape
func conv2DResidualBias(sums *[4]float32, bias []float32)
