package tensor

import "fmt"

// Conv2DShape returns the checked float32 NCHW cross-correlation output shape.
// Weights are [M, C/Groups, kH, kW]. Nonpositive spatial outputs become zero;
// channels, filters, and kernel extents must be positive. No payload is allocated.
func Conv2DShape(input, weights []int64, config Conv2DConfig) (shape []int64, err error) {
	var height, width int64

	if len(input) != 4 || len(weights) != 4 {
		err = fmt.Errorf("tensor: Conv2D requires rank-four input and weights")
		return nil, err
	}
	if _, _, err = ValidateShape(input, Float32); err != nil {
		return nil, fmt.Errorf("tensor: Conv2D input shape is invalid: %w", err)
	}
	if _, _, err = ValidateShape(weights, Float32); err != nil {
		return nil, fmt.Errorf("tensor: Conv2D weights shape is invalid: %w", err)
	}
	if config.Groups <= 0 || input[1] == 0 || weights[0] == 0 || weights[2] == 0 || weights[3] == 0 {
		err = fmt.Errorf("tensor: Conv2D groups, channels, filters, and kernel extents must be positive")
		return nil, err
	}
	if input[1]%config.Groups != 0 || weights[0]%config.Groups != 0 || weights[1] != input[1]/config.Groups {
		err = fmt.Errorf("tensor: Conv2D channel/group mismatch: input=%d weights=%v groups=%d", input[1], weights, config.Groups)
		return nil, err
	}
	if height, err = spatialOutputDimension(input[2], weights[2], config.Strides[0], config.Dilations[0], config.Pads[0], config.Pads[2]); err != nil {
		return nil, fmt.Errorf("tensor: Conv2D height geometry is invalid: %w", err)
	}
	if width, err = spatialOutputDimension(input[3], weights[3], config.Strides[1], config.Dilations[1], config.Pads[1], config.Pads[3]); err != nil {
		return nil, fmt.Errorf("tensor: Conv2D width geometry is invalid: %w", err)
	}
	shape = []int64{input[0], weights[0], height, width}
	if _, _, err = ValidateShape(shape, Float32); err != nil {
		return nil, fmt.Errorf("tensor: Conv2D output shape is invalid: %w", err)
	}
	return shape, nil
}

// Conv2DInto writes float32 NCHW cross-correlation into a disjoint destination.
// Weights use [M, C/Groups, kH, kW]; optional bias has shape [M]. Accumulation
// is float32. Inputs and parameters are unchanged; all errors preserve destination.
// No training state, packed weights, or backward caches are constructed.
func Conv2DInto(input, weights, bias, destination *Tensor, config Conv2DConfig) (err error) {
	var shape []int64

	if err = validateTensorType("Conv2D input", input, Float32); err != nil {
		return err
	}
	if err = validateTensorType("Conv2D weights", weights, Float32); err != nil {
		return err
	}
	if shape, err = Conv2DShape(input.shape, weights.shape, config); err != nil {
		return err
	}
	if bias != nil {
		if err = validateTensorType("Conv2D bias", bias, Float32); err != nil {
			return err
		}
		if !shapesEqual(bias.shape, []int64{weights.shape[0]}) {
			err = fmt.Errorf("tensor: Conv2D bias shape is %v, want [%d]", bias.shape, weights.shape[0])
			return err
		}
	}
	if err = validateDestination(destination, Float32, shape, input, weights, bias); err != nil {
		return err
	}
	if len(destination.floats) == 0 {
		return nil
	}
	conv2DValuesInto(input, weights, bias, destination, config)
	return nil
}

func conv2DValuesInto(input, weights, bias, destination *Tensor, config Conv2DConfig) {
	var (
		batch, channel, row, col, offset int
		value                            float32
	)

	for batch = 0; batch < int(destination.shape[0]); batch++ {
		for channel = 0; channel < int(destination.shape[1]); channel++ {
			for row = 0; row < int(destination.shape[2]); row++ {
				for col = 0; col < int(destination.shape[3]); col++ {
					value = conv2DValue(input, weights, batch, channel, row, col, config)
					if bias != nil {
						value += bias.floats[channel]
					}
					destination.floats[offset] = value
					offset++
				}
			}
		}
	}
}

func conv2DValue(input, weights *Tensor, batch, channel, row, col int, config Conv2DConfig) (value float32) {
	var (
		inputChannel, kernelRow, kernelCol int
		inputRow                           int
		inputOffset, weightOffset          int
		channels                           = int(weights.shape[1])
		kernelH                            = int(weights.shape[2])
		kernelW                            = int(weights.shape[3])
		first                              = channel / int(weights.shape[0]/config.Groups) * channels
		inputH                             = int(input.shape[2])
		inputW                             = int(input.shape[3])
		inputPlane                         = inputH * inputW
		kernelPlane                        = kernelH * kernelW
		inputBase                          = (batch*int(input.shape[1]) + first) * inputPlane
		weightBase                         = channel * channels * kernelPlane
		rowStart                           = row*int(config.Strides[0]) - int(config.Pads[0])
		colStart                           = col*int(config.Strides[1]) - int(config.Pads[1])
		dilationW                          = int(config.Dilations[1])
		kernelStart                        int
		kernelEnd                          = kernelW
	)

	if inputH == 0 || inputW == 0 || colStart >= inputW {
		return 0
	}

	if colStart < 0 {
		kernelStart = (-colStart-1)/dilationW + 1
	}

	if kernelStart >= kernelW {
		return 0
	}

	colStart += kernelStart * dilationW
	if colStart >= inputW {
		return 0
	}

	kernelEnd = min(kernelEnd, kernelStart+(inputW-1-colStart)/dilationW+1)
	for inputChannel = 0; inputChannel < channels; inputChannel++ {
		for kernelRow = 0; kernelRow < kernelH; kernelRow++ {
			// The planner checks maximum padded coordinates before subtraction;
			// valid spatial indices then bound every payload offset below.
			inputRow = rowStart + kernelRow*int(config.Dilations[0])
			if inputRow < 0 || inputRow >= inputH {
				continue
			}

			inputOffset = inputBase + inputChannel*inputPlane + inputRow*inputW + colStart
			weightOffset = weightBase + inputChannel*kernelPlane + kernelRow*kernelW + kernelStart
			for kernelCol = kernelStart; kernelCol < kernelEnd; kernelCol++ {
				value += input.floats[inputOffset] * weights.floats[weightOffset]
				inputOffset += dilationW
				weightOffset++
			}
		}
	}
	return value
}
