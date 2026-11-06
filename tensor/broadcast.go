package tensor

import "fmt"

// BroadcastShape returns an owned shape using trailing-axis multidirectional
// broadcasting. Shapes and float32 payload sizes are checked without allocation
// of numeric storage. A zero dimension broadcasts with one to zero.
func BroadcastShape(left, right []int64) (shape []int64, err error) {
	var (
		rank                          int
		axis, leftAxis, rightAxis     int
		leftDimension, rightDimension int64
	)

	if _, _, err = ValidateShape(left, Float32); err != nil {
		err = fmt.Errorf("tensor: left broadcast shape is invalid: %w", err)
		return nil, err
	}
	if _, _, err = ValidateShape(right, Float32); err != nil {
		err = fmt.Errorf("tensor: right broadcast shape is invalid: %w", err)
		return nil, err
	}
	rank = max(len(left), len(right))
	shape = make([]int64, rank)
	for axis = 0; axis < rank; axis++ {
		leftAxis = axis - (rank - len(left))
		rightAxis = axis - (rank - len(right))
		leftDimension, rightDimension = 1, 1
		if leftAxis >= 0 {
			leftDimension = left[leftAxis]
		}
		if rightAxis >= 0 {
			rightDimension = right[rightAxis]
		}
		switch {
		case leftDimension == rightDimension:
			shape[axis] = leftDimension
		case leftDimension == 1:
			shape[axis] = rightDimension
		case rightDimension == 1:
			shape[axis] = leftDimension
		default:
			err = fmt.Errorf("tensor: broadcast dimensions are incompatible: axis=%d left=%d right=%d", axis, leftDimension, rightDimension)
			return nil, err
		}
	}
	if _, _, err = ValidateShape(shape, Float32); err != nil {
		err = fmt.Errorf("tensor: broadcast result shape is invalid: %w", err)
		return nil, err
	}
	return shape, nil
}

// ValidateBroadcastTo checks that source can broadcast to exactly destination.
// It rejects extra source axes and never enlarges the destination shape.
func ValidateBroadcastTo(source, destination []int64) (err error) {
	var shape []int64

	if len(source) > len(destination) {
		err = fmt.Errorf("tensor: broadcast source rank exceeds destination rank: source=%d destination=%d", len(source), len(destination))
		return err
	}
	if shape, err = BroadcastShape(source, destination); err != nil {
		return err
	}
	if !shapesEqual(shape, destination) {
		err = fmt.Errorf("tensor: broadcast source would enlarge destination: source=%v destination=%v", source, destination)
		return err
	}
	return nil
}
