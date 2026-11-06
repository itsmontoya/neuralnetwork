package tensor

import "fmt"

// ValidateShape checks concrete dimensions and returns payload sizes without
// allocating numeric storage. Sizes must fit int64 and the host int index range.
func ValidateShape(shape []int64, dataType DataType) (elements, bytes int64, err error) {
	elements, bytes, err = validateShapeSize(shape, dataType)
	return elements, bytes, err
}

// NormalizeAxis resolves a negative axis against a positive rank.
// Scalars have no axis; both out-of-range axes and nonpositive ranks fail.
func NormalizeAxis(axis int64, rank int) (normalized int, err error) {
	var dimensions = int64(rank)

	if rank <= 0 || axis < -dimensions || axis >= dimensions {
		err = fmt.Errorf("tensor: axis is out of range: axis=%d rank=%d", axis, rank)
		return 0, err
	}
	if axis < 0 {
		axis += dimensions
	}
	return int(axis), nil
}

func shapesEqual(left, right []int64) (equal bool) {
	var axis int

	if len(left) != len(right) {
		return false
	}
	for axis = range left {
		if left[axis] != right[axis] {
			return false
		}
	}
	return true
}
