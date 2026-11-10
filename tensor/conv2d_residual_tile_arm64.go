//go:build arm64 && !purego

package tensor

// conv2DResidualTile shares weights across two to four residual columns. The
// caller checks unit width stride/dilation and group bounds, retaining the
// original interior tile boundaries and their different NaN operand priority.
func conv2DResidualTile(input, weights, bias, destination *Tensor, batch, channel, row, start, offset, width int, config Conv2DConfig) (handled bool) {
	var (
		sums        [16]float32
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
		columnFirst = max(0, -start-width+1)
		columnEnd   = min(kernelW, inputW-start)
		groupStart  = channel / int(weights.shape[0]/config.Groups) * channels
		plane       = int(destination.shape[2] * destination.shape[3])
	)

	// Bound the union of valid samples; assembly commits only the lanes whose
	// products belong to each output, even for nonfinite weights.
	if rowFirst < rowEnd && columnFirst < columnEnd {
		var (
			rows        = rowEnd - rowFirst
			columns     = columnEnd - columnFirst
			first       = max(0, -start-columnFirst)
			end         = inputW - start - columnFirst
			inputColumn = max(0, start+columnFirst)
			inputEnd    = min(inputW, start+columnEnd+width-1)
			inputBase   = (batch*int(input.shape[1])+groupStart)*inputPlane + (rowStart+rowFirst)*inputW + inputColumn
			weightBase  = channel*filterSize + rowFirst*kernelW + columnFirst
			inputSpan   = (channels-1)*inputPlane + (rows-1)*inputW + inputEnd - inputColumn
			weightSpan  = 3*filterSize + (channels-1)*kernelPlane + (rows-1)*kernelW + columns
		)

		conv2DResidualTileSums(input.floats[inputBase:inputBase+inputSpan], weights.floats[weightBase:weightBase+weightSpan],
			&sums, channels, rows, columns, inputPlane, inputW, kernelPlane, kernelW, first, end, width)
	}
	if bias != nil {
		conv2DResidualTileBias(&sums, bias.floats[channel:channel+4])
	}
	for lane := 0; lane < 4; lane++ {
		var values = destination.floats[offset+lane*plane : offset+lane*plane+width]

		for column := range values {
			values[column] = sums[lane*4+column]
		}
	}
	return true
}

// conv2DResidualTileSums keeps each column's fused reduction order and the
// residual helper's input-first NaN priority. Only valid samples are loaded;
// padded lanes retain their accumulators even when weights are nonfinite.
//
//go:noescape
func conv2DResidualTileSums(input, weights []float32, sums *[16]float32, channels, rows, columns, inputPlane, inputW, kernelPlane, kernelW, first, end, width int)

// conv2DResidualTileBias retains the residual caller's bias-first NaN priority.
//
//go:noescape
func conv2DResidualTileBias(sums *[16]float32, bias []float32)
