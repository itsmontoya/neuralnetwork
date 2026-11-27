package gpu

import (
	"fmt"
	"math"
)

// Pool2DConfig specifies explicit NCHW 2D spatial pooling geometry without
// schema defaults. Pads are ordered [top, left, bottom, right]. Kernel,
// Strides, and Dilations must be positive; Pads must be nonnegative.
// CountIncludePad applies only to AvgPool2D and is ignored by MaxPool2D.
type Pool2DConfig struct {
	Kernel          [2]int64
	Strides         [2]int64
	Pads            [4]int64
	Dilations       [2]int64
	CeilMode        bool
	CountIncludePad bool
}

// Pool2DShape computes and validates the rank-four NCHW output shape for 2D
// spatial pooling. Input must have four positive dimensions; kernel, strides,
// and dilations must be positive; pads must be nonnegative. In CeilMode,
// trailing windows starting at or beyond input + padBegin are removed,
// matching tensor.Pool2DShape and ONNX MaxPool/AveragePool. Output spatial
// dimensions must be strictly positive; invalid inputs return ErrInvalid.
func Pool2DShape(input []int64, config Pool2DConfig) (shape []int64, err error) {
	var (
		outH int64
		outW int64
	)

	if len(input) != 4 {
		return nil, fmt.Errorf("%w: Pool2D requires rank-four NCHW input, got rank %d", ErrInvalid, len(input))
	}
	var inCount uint64 = 1
	for i, dim := range input {
		if dim <= 0 {
			return nil, fmt.Errorf("%w: Pool2D input dimension %d must be positive, got %d", ErrInvalid, i, dim)
		}
		if inCount > math.MaxUint64/uint64(dim) {
			return nil, fmt.Errorf("%w: Pool2D input element count overflows", ErrInvalid)
		}
		inCount *= uint64(dim)
	}
	if outH, err = pool2DOutputDimension(input[2], config.Kernel[0], config.Strides[0], config.Dilations[0], config.Pads[0], config.Pads[2], config.CeilMode); err != nil {
		return nil, fmt.Errorf("Pool2D height: %w", err)
	}
	if outW, err = pool2DOutputDimension(input[3], config.Kernel[1], config.Strides[1], config.Dilations[1], config.Pads[1], config.Pads[3], config.CeilMode); err != nil {
		return nil, fmt.Errorf("Pool2D width: %w", err)
	}
	shape = []int64{input[0], input[1], outH, outW}
	var outCount uint64 = 1
	for _, dim := range shape {
		if outCount > math.MaxUint64/uint64(dim) {
			return nil, fmt.Errorf("%w: Pool2D output element count overflows", ErrInvalid)
		}
		outCount *= uint64(dim)
	}
	return shape, nil
}

func pool2DOutputDimension(input, kernel, stride, dilation, padBegin, padEnd int64, ceilMode bool) (output int64, err error) {
	var (
		effective int64
		padded    int64
		boundary  int64
		allowed   int64
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
		if ceilMode && effective-padded < stride {
			output = 1
		} else {
			return 0, fmt.Errorf("%w: output spatial dimension must be positive", ErrInvalid)
		}
	} else {
		if output, err = checkedAddInt64((padded-effective)/stride, 1); err != nil {
			return 0, err
		}
		if ceilMode && (padded-effective)%stride != 0 {
			if output, err = checkedAddInt64(output, 1); err != nil {
				return 0, err
			}
		}
	}

	if ceilMode {
		if boundary, err = checkedAddInt64(input, padBegin); err != nil {
			return 0, err
		}
		allowed = (boundary-1)/stride + 1
		if output > allowed {
			output = allowed
		}
	}
	if output <= 0 {
		return 0, fmt.Errorf("%w: output spatial dimension must be positive", ErrInvalid)
	}

	var (
		lastStart  int64
		lastExtent int64
	)
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

func checkedAddInt64(left, right int64) (sum int64, err error) {
	if right > 0 && left > math.MaxInt64-right {
		return 0, fmt.Errorf("%w: spatial coordinate overflow", ErrInvalid)
	}
	if right < 0 && left < math.MinInt64-right {
		return 0, fmt.Errorf("%w: spatial coordinate overflow", ErrInvalid)
	}
	return left + right, nil
}

func checkedMulInt64(left, right int64) (product int64, err error) {
	if left == 0 || right == 0 {
		return 0, nil
	}
	if left > 0 && right > 0 && left > math.MaxInt64/right {
		return 0, fmt.Errorf("%w: spatial coordinate overflow", ErrInvalid)
	}
	return left * right, nil
}
