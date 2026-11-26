package gpu

import (
	"fmt"
	"math"
)

// Conv2DConfig specifies explicit NCHW 2D cross-correlation geometry without
// schema defaults. Pads are ordered [top, left, bottom, right]. Strides,
// Dilations, and Groups must be positive; Pads must be nonnegative.
type Conv2DConfig struct {
	Strides   [2]int64
	Pads      [4]int64
	Dilations [2]int64
	Groups    int64
}

// Conv2DShape computes and validates the rank-four NCHW output shape for 2D
// cross-correlation with input [N, C, H, W] and weights [M, C/G, kH, kW].
// All input, weight, and output dimensions must be positive, element counts
// must fit within uint32, strides/dilations/groups must be positive, pads must
// be nonnegative, and channels/filters must divide evenly by Groups. Invalid
// inputs return ErrInvalid.
func Conv2DShape(input, weights []int64, config Conv2DConfig) (shape []int64, err error) {
	var (
		outH int64
		outW int64
	)

	if len(input) != 4 || len(weights) != 4 {
		return nil, fmt.Errorf("%w: Conv2D requires rank-four NCHW input and weights, got ranks %d and %d", ErrInvalid, len(input), len(weights))
	}
	if err = validateConv2DDims("input", input); err != nil {
		return nil, err
	}
	if err = validateConv2DDims("weights", weights); err != nil {
		return nil, err
	}
	if config.Groups <= 0 {
		return nil, fmt.Errorf("%w: Conv2D groups must be positive, got %d", ErrInvalid, config.Groups)
	}
	if input[1]%config.Groups != 0 || weights[0]%config.Groups != 0 || weights[1] != input[1]/config.Groups {
		return nil, fmt.Errorf("%w: Conv2D channel/group mismatch: input=%d weights=%v groups=%d", ErrInvalid, input[1], weights, config.Groups)
	}
	if outH, err = conv2DOutputDimension(input[2], weights[2], config.Strides[0], config.Dilations[0], config.Pads[0], config.Pads[2]); err != nil {
		return nil, fmt.Errorf("Conv2D height: %w", err)
	}
	if outW, err = conv2DOutputDimension(input[3], weights[3], config.Strides[1], config.Dilations[1], config.Pads[1], config.Pads[3]); err != nil {
		return nil, fmt.Errorf("Conv2D width: %w", err)
	}
	shape = []int64{input[0], weights[0], outH, outW}
	if err = validateConv2DDims("output", shape); err != nil {
		return nil, err
	}
	return shape, nil
}

func validateConv2DDims(label string, dims []int64) (err error) {
	var count uint64 = 1
	for i, dim := range dims {
		if dim <= 0 {
			return fmt.Errorf("%w: Conv2D %s dimension %d must be positive, got %d", ErrInvalid, label, i, dim)
		}
		if count > math.MaxUint32/uint64(dim) {
			return fmt.Errorf("%w: Conv2D %s element count exceeds uint32", ErrInvalid, label)
		}
		count *= uint64(dim)
	}
	return nil
}

func conv2DOutputDimension(input, kernel, stride, dilation, padBegin, padEnd int64) (output int64, err error) {
	var (
		effective  int64
		padded     int64
		lastStart  int64
		lastExtent int64
	)

	if kernel <= 0 || stride <= 0 || dilation <= 0 {
		return 0, fmt.Errorf("%w: kernel, stride, and dilation must be positive", ErrInvalid)
	}
	if padBegin < 0 || padEnd < 0 {
		return 0, fmt.Errorf("%w: pads must be nonnegative", ErrInvalid)
	}
	if effective, err = checkedMulInt64(kernel-1, dilation); err != nil {
		return 0, err
	}
	if effective, err = checkedAddInt64(effective, 1); err != nil {
		return 0, err
	}
	if padded, err = checkedAddInt64(input, padBegin); err != nil {
		return 0, err
	}
	if padded, err = checkedAddInt64(padded, padEnd); err != nil {
		return 0, err
	}
	if padded < effective {
		return 0, fmt.Errorf("%w: output spatial dimension must be positive", ErrInvalid)
	}
	if output, err = checkedAddInt64((padded-effective)/stride, 1); err != nil {
		return 0, err
	}
	if output <= 0 {
		return 0, fmt.Errorf("%w: output spatial dimension must be positive", ErrInvalid)
	}
	if lastStart, err = checkedMulInt64(output-1, stride); err != nil {
		return 0, err
	}
	if lastExtent, err = checkedMulInt64(kernel-1, dilation); err != nil {
		return 0, err
	}
	if _, err = checkedAddInt64(lastStart, lastExtent); err != nil {
		return 0, err
	}
	return output, nil
}
