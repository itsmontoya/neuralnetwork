package tensor

import "fmt"

// Bound common-rank scratch to 64 bytes per shape; higher ranks remain supported.
const broadcastStackRank = 8

// BroadcastShape returns an owned shape using trailing-axis multidirectional
// broadcasting. Shapes and float32 payload sizes are checked without allocation
// of numeric storage. A zero dimension broadcasts with one to zero.
func BroadcastShape(left, right []int64) (shape []int64, err error) {
	return broadcastShapeInto(left, right, []int64{})
}

func broadcastShapeInto(left, right, buffer []int64) (shape []int64, err error) {
	if _, _, err = ValidateShape(left, Float32); err != nil {
		err = fmt.Errorf("tensor: left broadcast shape is invalid: %w", err)
		return nil, err
	}
	if _, _, err = ValidateShape(right, Float32); err != nil {
		err = fmt.Errorf("tensor: right broadcast shape is invalid: %w", err)
		return nil, err
	}
	if shape, err = broadcastDimensionsInto(left, right, buffer); err != nil {
		return nil, err
	}
	if _, _, err = ValidateShape(shape, Float32); err != nil {
		err = fmt.Errorf("tensor: broadcast result shape is invalid: %w", err)
		return nil, err
	}
	return shape, nil
}

// broadcastDimensions plans metadata such as batch prefixes, which do not
// themselves own numeric payloads. Callers validate their actual output size.
func broadcastDimensions(left, right []int64) (shape []int64, err error) {
	return broadcastDimensionsInto(left, right, []int64{})
}

func broadcastDimensionsInto(left, right, buffer []int64) (shape []int64, err error) {
	var (
		rank                          int
		axis, leftAxis, rightAxis     int
		leftDimension, rightDimension int64
	)

	if _, err = validateDimensions(left); err != nil {
		return nil, err
	}
	if _, err = validateDimensions(right); err != nil {
		return nil, err
	}
	rank = max(len(left), len(right))
	if cap(buffer) < rank {
		buffer = make([]int64, rank)
	}

	shape = buffer[:rank]
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
	return shape, nil
}

// ValidateBroadcastTo checks that source can broadcast to exactly destination.
// It rejects extra source axes and never enlarges the destination shape.
func ValidateBroadcastTo(source, destination []int64) (err error) {
	var (
		shape  []int64
		buffer [broadcastStackRank]int64
	)

	if len(source) > len(destination) {
		err = fmt.Errorf("tensor: broadcast source rank exceeds destination rank: source=%d destination=%d", len(source), len(destination))
		return err
	}
	if shape, err = broadcastShapeInto(source, destination, buffer[:]); err != nil {
		return err
	}
	if !shapesEqual(shape, destination) {
		err = fmt.Errorf("tensor: broadcast source would enlarge destination: source=%v destination=%v", source, destination)
		return err
	}
	return nil
}
