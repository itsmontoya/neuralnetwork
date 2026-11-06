package tensor

import "fmt"

// FlattenShape returns [prefix product, suffix product] at a split axis.
// Axes in [-rank, rank] are accepted, including both endpoints; scalar axis zero
// produces [1, 1]. Shapes support either tensor type without payload allocation.
func FlattenShape(input []int64, axis int64) (shape []int64, err error) {
	var (
		rank           = int64(len(input))
		prefix, suffix int64
	)

	if _, _, err = ValidateShape(input, Int64); err != nil {
		return nil, err
	}
	if axis < -rank || axis > rank {
		err = fmt.Errorf("tensor: flatten split axis is out of range: axis=%d rank=%d", axis, rank)
		return nil, err
	}
	if axis < 0 {
		axis += rank
	}
	if prefix, err = checkedProduct(input[:int(axis)]); err != nil {
		return nil, err
	}
	if suffix, err = checkedProduct(input[int(axis):]); err != nil {
		return nil, err
	}
	shape = []int64{prefix, suffix}
	if _, _, err = ValidateShape(shape, Int64); err != nil {
		return nil, err
	}
	return shape, nil
}

// FlattenInto copies input into the rank-two shape split at axis.
// Both types are preserved. Storage must be disjoint; errors preserve destination.
func FlattenInto(input, destination *Tensor, axis int64) (err error) {
	var shape []int64

	if err = validateTransformInput(input); err != nil {
		return err
	}
	if shape, err = FlattenShape(input.shape, axis); err != nil {
		return err
	}
	err = copyTransformInto(input, destination, shape)
	return err
}
