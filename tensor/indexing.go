package tensor

import "fmt"

// contiguousStrides checks every suffix product it returns, including empties.
func contiguousStrides(shape []int64) (strides []int64, err error) {
	var (
		axis   int
		stride int64 = 1
	)

	if _, err = checkedProduct(shape); err != nil {
		return nil, err
	}
	strides = make([]int64, len(shape))
	for axis = len(shape) - 1; axis >= 0; axis-- {
		strides[axis] = stride
		if stride, err = checkedMultiply(stride, shape[axis]); err != nil {
			err = fmt.Errorf("tensor: contiguous stride is invalid at axis %d: %w", axis, err)
			return nil, err
		}
	}
	return strides, nil
}

func shapeOffset(shape, indices []int64) (offset int, err error) {
	var (
		strides []int64
		sum     int64
		term    int64
		axis    int
		index   int64
	)

	if len(shape) != len(indices) {
		err = fmt.Errorf("tensor: index rank does not match shape: indices=%d rank=%d", len(indices), len(shape))
		return 0, err
	}
	if strides, err = contiguousStrides(shape); err != nil {
		return 0, err
	}
	for axis, index = range indices {
		if index < 0 || index >= shape[axis] {
			err = fmt.Errorf("tensor: index is out of range: axis=%d index=%d dimension=%d", axis, index, shape[axis])
			return 0, err
		}
		if term, err = checkedMultiply(index, strides[axis]); err != nil {
			return 0, err
		}
		if sum, err = checkedAdd(sum, term); err != nil {
			return 0, err
		}
	}
	return int(sum), nil
}
