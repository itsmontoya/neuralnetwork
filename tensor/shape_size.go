package tensor

import "fmt"

func validateShapeSize(shape []int64, dataType DataType) (elements, bytes int64, err error) {
	var byteWidth int64

	if byteWidth, err = dataTypeByteSize(dataType); err != nil {
		return 0, 0, err
	}
	if elements, err = checkedProduct(shape); err != nil {
		return 0, 0, err
	}
	if bytes, err = checkedMultiply(elements, byteWidth); err != nil {
		err = fmt.Errorf("tensor: byte count is invalid: %w", err)
		return 0, 0, err
	}
	return elements, bytes, nil
}

func validateDimensions(shape []int64) (empty bool, err error) {
	var (
		maxHostIndex = int64(int(^uint(0) >> 1))
		dimension    int64
		axis         int
	)

	for axis, dimension = range shape {
		if dimension < 0 || dimension > maxHostIndex {
			err = fmt.Errorf("tensor: dimension cannot be a host index: axis=%d dimension=%d", axis, dimension)
			return false, err
		}
		if dimension == 0 {
			empty = true
		}
	}
	return empty, nil
}

// checkedProduct validates every dimension before handling an empty product.
func checkedProduct(dimensions []int64) (product int64, err error) {
	var (
		empty     bool
		dimension int64
		axis      int
	)

	if empty, err = validateDimensions(dimensions); err != nil {
		return 0, err
	}
	if empty {
		return 0, nil
	}
	product = 1
	for axis, dimension = range dimensions {
		if product, err = checkedMultiply(product, dimension); err != nil {
			err = fmt.Errorf("tensor: dimension product is invalid at axis %d: %w", axis, err)
			return 0, err
		}
	}
	return product, nil
}

func checkedAdd(left, right int64) (sum int64, err error) {
	var maxHostIndex = int64(int(^uint(0) >> 1))

	if left < 0 || right < 0 || left > maxHostIndex || right > maxHostIndex || left > maxHostIndex-right {
		err = fmt.Errorf("tensor: sum cannot be a host index: left=%d right=%d", left, right)
		return 0, err
	}
	return left + right, nil
}

func checkedMultiply(left, right int64) (product int64, err error) {
	var maxHostIndex = int64(int(^uint(0) >> 1))

	if left < 0 || right < 0 || left > maxHostIndex || right > maxHostIndex {
		err = fmt.Errorf("tensor: product operands cannot be host indices: left=%d right=%d", left, right)
		return 0, err
	}
	if right != 0 && left > maxHostIndex/right {
		err = fmt.Errorf("tensor: product cannot be a host index: left=%d right=%d", left, right)
		return 0, err
	}
	return left * right, nil
}
