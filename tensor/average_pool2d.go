package tensor

import "fmt"

// AveragePool2DInto writes float32 NCHW averages into a disjoint destination.
// Dilation must be [1,1]. CountIncludePad uses the full kernel area, including
// positions beyond the input in a trailing ceil window. A zero divisor produces NaN;
// input NaNs propagate. All errors preserve destination. No training state or
// backward cache is constructed.
func AveragePool2DInto(input, destination *Tensor, config Pool2DConfig) (err error) {
	var batch, channel, row, col, offset int

	if err = validatePoolDestination(input, destination, config); err != nil {
		return err
	}
	if config.Dilations != [2]int64{1, 1} {
		err = fmt.Errorf("tensor: average pooling requires unit dilation")
		return err
	}
	if len(destination.floats) == 0 {
		return nil
	}
	if _, err = checkedProduct(config.Kernel[:]); err != nil {
		return fmt.Errorf("tensor: average pooling divisor size is invalid: %w", err)
	}
	for batch = 0; batch < int(destination.shape[0]); batch++ {
		for channel = 0; channel < int(destination.shape[1]); channel++ {
			for row = 0; row < int(destination.shape[2]); row++ {
				for col = 0; col < int(destination.shape[3]); col++ {
					destination.floats[offset] = averagePool2DValue(input, batch, channel, row, col, config)
					offset++
				}
			}
		}
	}
	return nil
}

func averagePool2DValue(input *Tensor, batch, channel, row, col int, config Pool2DConfig) (value float32) {
	var (
		firstRow, endRow, firstCol, endCol     int
		kernelRow, kernelCol, offset           int
		startRow, startCol, inputRow, inputCol int64
		count                                  int64
	)

	startRow = int64(row)*config.Strides[0] - config.Pads[0]
	startCol = int64(col)*config.Strides[1] - config.Pads[1]
	firstRow, endRow = poolKernelRange(startRow, input.shape[2], config.Kernel[0], 1)
	firstCol, endCol = poolKernelRange(startCol, input.shape[3], config.Kernel[1], 1)
	for kernelRow = firstRow; kernelRow < endRow; kernelRow++ {
		inputRow = startRow + int64(kernelRow)
		for kernelCol = firstCol; kernelCol < endCol; kernelCol++ {
			inputCol = startCol + int64(kernelCol)
			// Checked geometry and valid coordinates bound this payload offset.
			offset = ((batch*int(input.shape[1])+channel)*int(input.shape[2])+int(inputRow))*int(input.shape[3]) + int(inputCol)
			value += input.floats[offset]
		}
	}
	// Both counts are bounded by the prevalidated kernel area.
	count = int64(endRow-firstRow) * int64(endCol-firstCol)
	if config.CountIncludePad {
		count = config.Kernel[0] * config.Kernel[1]
	}
	return value / float32(count)
}
