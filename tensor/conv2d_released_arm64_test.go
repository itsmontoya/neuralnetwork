//go:build arm64 && !purego

package tensor

// conv2DReleasedChannelBlocks pins the v0.10.3 traversal for bitwise regression
// checks. It deliberately calls the unchanged interior and single-column helpers.
func conv2DReleasedChannelBlocks(input, weights, bias, destination *Tensor, config Conv2DConfig) {
	var (
		batch, channel, row, col, lane int
		filters                        = int(weights.shape[0] / config.Groups)
		height                         = int(destination.shape[2])
		width                          = int(destination.shape[3])
		plane                          = height * width
		// Empty inputs and single-product reductions keep the cheaper scalar path.
		residual = conv2DHasInteriorTile && config.Strides[1] == 1 && config.Dilations == [2]int64{1, 1} &&
			input.shape[2] != 0 && input.shape[3] != 0 && weights.shape[1]*weights.shape[2]*weights.shape[3] > 1
	)

	for batch = 0; batch < int(destination.shape[0]); batch++ {
		for channel = 0; channel < int(destination.shape[1]); {
			var lanes = 1

			// Channel blocks must stay within a group so all four filters
			// read the same input channels.
			if channel%filters+4 <= filters {
				lanes = 4
			}

			for row = 0; row < height; row++ {
				for col = 0; col < width; col++ {
					var (
						start  = col*int(config.Strides[1]) - int(config.Pads[1])
						offset = (batch*int(destination.shape[1])+channel)*plane + row*width + col
					)

					if config.Dilations[1] == 1 && col+3 < width && start >= 0 &&
						start+3*int(config.Strides[1])+(int(weights.shape[3])-1) < int(input.shape[3]) {
						if lanes == 4 && config.Strides[1] == 1 && config.Dilations[0] == 1 &&
							conv2DFourByFour(input, weights, bias, destination, batch, channel, row, start, offset, config) {
							col += 3
							continue
						}
						for lane = 0; lane < lanes; lane++ {
							var (
								first, second, third, fourth = conv2DFourValues(input, weights, batch, channel+lane, row, start, config)
								values                       = destination.floats[offset+lane*plane : offset+lane*plane+4]
							)

							if bias != nil {
								first += bias.floats[channel+lane]
								second += bias.floats[channel+lane]
								third += bias.floats[channel+lane]
								fourth += bias.floats[channel+lane]
							}

							values[0], values[1], values[2], values[3] = first, second, third, fourth
						}
						col += 3
						continue
					}

					if lanes == 4 {
						if residual && conv2DResidualChannels(input, weights, bias, destination, batch, channel, row, start, offset, config) {
							continue
						}

						var first, second, third, fourth = conv2DFourChannels(input, weights, batch, channel, row, col, config)

						if bias != nil {
							first += bias.floats[channel]
							second += bias.floats[channel+1]
							third += bias.floats[channel+2]
							fourth += bias.floats[channel+3]
						}

						destination.floats[offset] = first
						destination.floats[offset+plane] = second
						destination.floats[offset+2*plane] = third
						destination.floats[offset+3*plane] = fourth
						continue
					}

					var value = conv2DValue(input, weights, batch, channel, row, col, config)

					if bias != nil {
						value += bias.floats[channel]
					}
					destination.floats[offset] = value
				}
			}
			channel += lanes
		}
	}
}
