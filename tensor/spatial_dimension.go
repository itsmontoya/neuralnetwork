package tensor

import "fmt"

// spatialOutputDimension plans floor geometry without allocating numeric data.
func spatialOutputDimension(input, kernel, stride, dilation, padBegin, padEnd int64) (output int64, err error) {
	var (
		effective int64
		padded    int64
	)

	if _, err = validateDimensions([]int64{input, kernel, stride, dilation, padBegin, padEnd}); err != nil {
		return 0, err
	}
	if kernel == 0 || stride == 0 || dilation == 0 {
		err = fmt.Errorf("tensor: spatial kernel, stride, and dilation must be positive")
		return 0, err
	}
	if effective, err = checkedMultiply(kernel-1, dilation); err != nil {
		return 0, err
	}
	if effective, err = checkedAdd(effective, 1); err != nil {
		return 0, err
	}
	if padded, err = checkedAdd(input, padBegin); err != nil {
		return 0, err
	}
	if padded, err = checkedAdd(padded, padEnd); err != nil {
		return 0, err
	}
	if padded < effective {
		return 0, nil
	}
	if output, err = checkedAdd((padded-effective)/stride, 1); err != nil {
		return 0, err
	}
	if err = validateSpatialCoordinates(output, kernel, stride, dilation); err != nil {
		return 0, err
	}
	return output, nil
}

// validateSpatialCoordinates bounds the largest padded sampling coordinate.
// It also supports later ceil geometry, whose last sample can exceed padding.
func validateSpatialCoordinates(output, kernel, stride, dilation int64) (err error) {
	var start, extent int64

	if output == 0 {
		return nil
	}
	if start, err = checkedMultiply(output-1, stride); err != nil {
		return err
	}
	if extent, err = checkedMultiply(kernel-1, dilation); err != nil {
		return err
	}
	_, err = checkedAdd(start, extent)
	return err
}
