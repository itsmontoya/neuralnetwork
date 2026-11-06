package tensor

import "github.com/itsmontoya/neuralnetwork/internal/f32"

// MaxPool2DInto writes float32 NCHW maxima into a disjoint destination.
// Padding is excluded; windows without input samples produce negative infinity.
// Input NaNs propagate within their windows. Indices and nondefault storage
// order are unsupported; callers must reject those requests. All errors
// preserve destination. No argmax history or backward state is allocated.
func MaxPool2DInto(input, destination *Tensor, config Pool2DConfig) (err error) {
	var batch, channel, row, col, offset int

	if err = validatePoolDestination(input, destination, config); err != nil {
		return err
	}
	if len(destination.floats) == 0 {
		return nil
	}
	for batch = 0; batch < int(destination.shape[0]); batch++ {
		for channel = 0; channel < int(destination.shape[1]); channel++ {
			for row = 0; row < int(destination.shape[2]); row++ {
				for col = 0; col < int(destination.shape[3]); col++ {
					destination.floats[offset] = maxPool2DValue(input, batch, channel, row, col, config)
					offset++
				}
			}
		}
	}
	return nil
}

func maxPool2DValue(input *Tensor, batch, channel, row, col int, config Pool2DConfig) (value float32) {
	var (
		firstRow, endRow, firstCol, endCol     int
		kernelRow, kernelCol, offset           int
		startRow, startCol, inputRow, inputCol int64
		sample                                 float32
	)

	// The planner bounds all padded sampling coordinates and host conversions.
	startRow = int64(row)*config.Strides[0] - config.Pads[0]
	startCol = int64(col)*config.Strides[1] - config.Pads[1]
	firstRow, endRow = poolKernelRange(startRow, input.shape[2], config.Kernel[0], config.Dilations[0])
	firstCol, endCol = poolKernelRange(startCol, input.shape[3], config.Kernel[1], config.Dilations[1])
	value = f32.Inf(-1)
	for kernelRow = firstRow; kernelRow < endRow; kernelRow++ {
		inputRow = startRow + int64(kernelRow)*config.Dilations[0]
		for kernelCol = firstCol; kernelCol < endCol; kernelCol++ {
			inputCol = startCol + int64(kernelCol)*config.Dilations[1]
			// Valid coordinates and checked payload lengths bound this offset.
			offset = ((batch*int(input.shape[1])+channel)*int(input.shape[2])+int(inputRow))*int(input.shape[3]) + int(inputCol)
			sample = input.floats[offset]
			if sample > value || f32.IsNaN(sample) {
				value = sample
			}
		}
	}
	return value
}
