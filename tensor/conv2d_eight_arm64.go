//go:build arm64 && !purego && !race

package tensor

const conv2DHasEightFilterTile = true

// conv2DEightFilterRows shares row geometry and input loads across eight filters.
// The caller checks unit width stride/dilation, nonempty input, and group bounds.
func conv2DEightFilterRows(input, weights, bias, destination *Tensor, batch, channel int, config Conv2DConfig) (handled bool) {
	var (
		channels    = int(weights.shape[1])
		kernelH     = int(weights.shape[2])
		kernelW     = int(weights.shape[3])
		inputH      = int(input.shape[2])
		inputW      = int(input.shape[3])
		inputPlane  = inputH * inputW
		kernelPlane = kernelH * kernelW
		filterSize  = channels * kernelPlane
		groupStart  = channel / int(weights.shape[0]/config.Groups) * channels
		inputBase   = (batch*int(input.shape[1]) + groupStart) * inputPlane
		height      = int(destination.shape[2])
		width       = int(destination.shape[3])
		plane       = height * width
		outputBase  = (batch*int(destination.shape[1]) + channel) * plane
		biasValues  []float32
	)

	if bias != nil {
		biasValues = bias.floats[channel : channel+8]
	}
	for row := 0; row < height; row++ {
		var (
			rowStart = row*int(config.Strides[0]) - int(config.Pads[0])
			rowFirst = max(0, -rowStart)
			rowEnd   = min(kernelH, inputH-rowStart)
			rows     = rowEnd - rowFirst
		)

		// Empty rows retain the existing zero and bias behavior.
		if rows <= 0 {
			for lane := 0; lane < 8; lane++ {
				var value float32
				if bias != nil {
					value += biasValues[lane]
				}
				for column := 0; column < width; column++ {
					destination.floats[outputBase+lane*plane+row*width+column] = value
				}
			}
			continue
		}
		for column := 0; column < width; {
			var (
				start   = column - int(config.Pads[1])
				offset  = outputBase + row*width + column
				columns = min(4, width-column)
			)

			if columns == 4 && start >= 0 && start+kernelW+3 <= inputW {
				var (
					first       = inputBase + (rowStart+rowFirst)*inputW + start
					weightFirst = channel*filterSize + rowFirst*kernelW
					inputSpan   = (channels-1)*inputPlane + (rows-1)*inputW + kernelW + 3
					weightSpan  = 7*filterSize + (channels-1)*kernelPlane + rows*kernelW
				)
				conv2DEightInterior(input.floats[first:first+inputSpan], weights.floats[weightFirst:weightFirst+weightSpan],
					destination.floats[offset:offset+7*plane+4], biasValues, channels, rows, kernelW, inputPlane, inputW, kernelPlane, plane)
				column += 4
				continue
			}
			// Preserve the released interior partition and its NaN operand priority.
			if start < 0 {
				columns = min(columns, -start)
			}
			if columns == 1 {
				conv2DResidualChannels(input, weights, bias, destination, batch, channel, row, start, offset, config)
				conv2DResidualChannels(input, weights, bias, destination, batch, channel+4, row, start, offset+4*plane, config)
				column++
				continue
			}
			var (
				columnFirst = max(0, -start-columns+1)
				columnEnd   = min(kernelW, inputW-start)
			)
			if columnFirst >= columnEnd {
				conv2DResidualTile(input, weights, bias, destination, batch, channel, row, start, offset, columns, config)
				conv2DResidualTile(input, weights, bias, destination, batch, channel+4, row, start, offset+4*plane, columns, config)
				column += columns
				continue
			}
			var (
				firstLane   = max(0, -start-columnFirst)
				end         = inputW - start - columnFirst
				inputColumn = max(0, start+columnFirst)
				inputEnd    = min(inputW, start+columnEnd+columns-1)
				first       = inputBase + (rowStart+rowFirst)*inputW + inputColumn
				weightFirst = channel*filterSize + rowFirst*kernelW + columnFirst
				inputSpan   = (channels-1)*inputPlane + (rows-1)*inputW + inputEnd - inputColumn
				weightSpan  = 7*filterSize + (channels-1)*kernelPlane + (rows-1)*kernelW + columnEnd - columnFirst
			)
			conv2DEightResidual(input.floats[first:first+inputSpan], weights.floats[weightFirst:weightFirst+weightSpan],
				destination.floats[offset:offset+7*plane+columns], biasValues, channels, rows, columnEnd-columnFirst,
				inputPlane, inputW, kernelPlane, kernelW, firstLane, end, columns, plane)
			column += columns
		}
	}
	return true
}

//go:noescape
func conv2DEightInterior(input, weights, output, bias []float32, channels, rows, kernelW, inputPlane, inputW, kernelPlane, plane int)

//go:noescape
func conv2DEightResidual(input, weights, output, bias []float32, channels, rows, columns, inputPlane, inputW, kernelPlane, kernelW, first, end, width, plane int)
