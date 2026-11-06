package tensor

import "fmt"

// Pool2DShape returns the checked float32 NCHW pooling output shape.
// Floor or ceil geometry uses explicit padding and dilated kernel extents.
// Ceil windows starting at or beyond input plus begin padding are removed.
// Nonpositive spatial extents become zero; no numeric payload is allocated.
func Pool2DShape(input []int64, config Pool2DConfig) (shape []int64, err error) {
	var height, width int64

	if len(input) != 4 {
		err = fmt.Errorf("tensor: pooling requires rank-four NCHW input")
		return nil, err
	}
	if _, _, err = ValidateShape(input, Float32); err != nil {
		return nil, fmt.Errorf("tensor: pooling input shape is invalid: %w", err)
	}
	if height, err = poolOutputDimension(input[2], config.Kernel[0], config.Strides[0], config.Dilations[0], config.Pads[0], config.Pads[2], config.CeilMode); err != nil {
		return nil, fmt.Errorf("tensor: pooling height geometry is invalid: %w", err)
	}
	if width, err = poolOutputDimension(input[3], config.Kernel[1], config.Strides[1], config.Dilations[1], config.Pads[1], config.Pads[3], config.CeilMode); err != nil {
		return nil, fmt.Errorf("tensor: pooling width geometry is invalid: %w", err)
	}
	shape = []int64{input[0], input[1], height, width}
	if _, _, err = ValidateShape(shape, Float32); err != nil {
		return nil, fmt.Errorf("tensor: pooling output shape is invalid: %w", err)
	}
	return shape, nil
}

func poolOutputDimension(input, kernel, stride, dilation, padBegin, padEnd int64, ceilMode bool) (output int64, err error) {
	var effective, padded, boundary, allowed int64

	if output, err = spatialOutputDimension(input, kernel, stride, dilation, padBegin, padEnd); err != nil || !ceilMode {
		return output, err
	}
	if effective, padded, err = spatialExtents(input, kernel, stride, dilation, padBegin, padEnd); err != nil {
		return 0, err
	}
	if padded < effective {
		if effective-padded < stride {
			output = 1
		}
	} else if (padded-effective)%stride != 0 {
		if output, err = checkedAdd(output, 1); err != nil {
			return 0, err
		}
	}
	if boundary, err = checkedAdd(input, padBegin); err != nil {
		return 0, err
	}
	if boundary > 0 {
		allowed = (boundary-1)/stride + 1
	}
	output = min(output, allowed)
	if err = validateSpatialCoordinates(output, kernel, stride, dilation); err != nil {
		return 0, err
	}
	return output, nil
}

func validatePoolDestination(input, destination *Tensor, config Pool2DConfig) (err error) {
	var shape []int64

	if err = validateTensorType("pooling input", input, Float32); err != nil {
		return err
	}
	if shape, err = Pool2DShape(input.shape, config); err != nil {
		return err
	}
	err = validateDestination(destination, Float32, shape, input)
	return err
}

// poolKernelRange selects valid input samples from checked spatial geometry.
func poolKernelRange(start, input, kernel, dilation int64) (first, end int) {
	var begin, limit int64

	if input == 0 || start >= input {
		return 0, 0
	}
	if start < 0 {
		begin = -start / dilation
		if -start%dilation != 0 {
			begin++
		}
	}
	// Checked input+beginPad bounds this difference when start is negative.
	limit = min(kernel, (input-1-start)/dilation+1)
	if begin >= limit {
		return 0, 0
	}
	return int(begin), int(limit)
}
