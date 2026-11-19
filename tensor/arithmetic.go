package tensor

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/float32elementwise"
)

// AddInto writes the broadcast float32 sum into a disjoint destination.
// All argument errors leave destination unchanged, including for empty results.
func AddInto(left, right, destination *Tensor) (err error) {
	err = binaryFloat32Into(left, right, destination, addValue, float32elementwise.Add, broadcastAddFloat32Into)
	return err
}

// SubInto writes the broadcast float32 difference into a disjoint destination.
// All argument errors leave destination unchanged, including for empty results.
func SubInto(left, right, destination *Tensor) (err error) {
	err = binaryFloat32Into(left, right, destination, subtractValue, float32elementwise.Subtract, nil)
	return err
}

// MulInto writes the broadcast float32 product into a disjoint destination.
// All argument errors leave destination unchanged, including for empty results.
func MulInto(left, right, destination *Tensor) (err error) {
	err = binaryFloat32Into(left, right, destination, multiplyValue, float32elementwise.Multiply, nil)
	return err
}

// DivInto writes the broadcast float32 quotient into a disjoint destination.
// Floating division preserves signed zero/infinity and yields NaN for invalid
// quotients, including zero divided by zero. NaN payloads are unspecified.
// All argument errors leave destination unchanged, including for empty results.
func DivInto(left, right, destination *Tensor) (err error) {
	err = binaryFloat32Into(left, right, destination, divideValue, nil, nil)
	return err
}

func binaryFloat32Into(left, right, destination *Tensor, operation func(float32, float32) float32, equalShapeOperation func(left, right, destination []float32), broadcastOperation func(*Tensor, *Tensor, *Tensor) error) (err error) {
	var (
		shape  []int64
		buffer [broadcastStackRank]int64
	)

	if err = validateTensorType("left", left, Float32); err != nil {
		return err
	}
	if err = validateTensorType("right", right, Float32); err != nil {
		return err
	}
	if shape, err = broadcastShapeInto(left.shape, right.shape, buffer[:]); err != nil {
		return err
	}
	if err = validateDestination(destination, Float32, shape, left, right); err != nil {
		return err
	}
	if len(destination.floats) == 0 {
		return nil
	}
	if equalShapeOperation != nil && shapesEqual(left.shape, right.shape) {
		equalShapeOperation(left.floats, right.floats, destination.floats)
		return nil
	}
	if broadcastOperation != nil {
		return broadcastOperation(left, right, destination)
	}
	err = broadcastFloat32Into(left, right, destination, operation)
	return err
}

func broadcastFloat32Into(left, right, destination *Tensor, operation func(float32, float32) float32) (err error) {
	var (
		coordinates, leftStrides, rightStrides []int64
		leftBuffer, rightBuffer                [broadcastStackRank]int64
		leftOffset, rightOffset                int64
		index, axis                            int
	)

	if leftStrides, err = broadcastStridesInto(left.shape, destination.shape, leftBuffer[:]); err != nil {
		return err
	}

	if rightStrides, err = broadcastStridesInto(right.shape, destination.shape, rightBuffer[:]); err != nil {
		return err
	}

	coordinates = make([]int64, len(destination.shape))
	for index = range destination.floats {
		destination.floats[index] = operation(left.floats[int(leftOffset)], right.floats[int(rightOffset)])
		// Carry from the trailing axis; broadcast axes have zero source stride.
		// Offsets stay within the validated input payloads, including on reset.
		for axis = len(coordinates) - 1; axis >= 0; axis-- {
			coordinates[axis]++
			if coordinates[axis] < destination.shape[axis] {
				leftOffset += leftStrides[axis]
				rightOffset += rightStrides[axis]
				break
			}

			coordinates[axis] = 0
			leftOffset -= (destination.shape[axis] - 1) * leftStrides[axis]
			rightOffset -= (destination.shape[axis] - 1) * rightStrides[axis]
		}
	}
	return nil
}

// broadcastStrides is for nonempty, validated broadcast operations only.
func broadcastStrides(input, output []int64) (strides []int64, err error) {
	return broadcastStridesInto(input, output, []int64{})
}

// broadcastStridesInto avoids a temporary contiguous-stride slice.
// Insufficient caller capacity falls back to an owned allocation.
func broadcastStridesInto(input, output, buffer []int64) (strides []int64, err error) {
	var stride int64 = 1
	if err = ValidateBroadcastTo(input, output); err != nil {
		return nil, err
	}

	if cap(buffer) < len(output) {
		buffer = make([]int64, len(output))
	}

	strides = buffer[:len(output)]
	clear(strides)
	shift := len(output) - len(input)
	for axis := len(input) - 1; axis >= 0; axis-- {
		if input[axis] != 1 {
			strides[axis+shift] = stride
		}

		if stride, err = checkedMultiply(stride, input[axis]); err != nil {
			return nil, fmt.Errorf("tensor: contiguous stride is invalid at axis %d: %w", axis, err)
		}
	}
	return strides, nil
}

func addValue(left, right float32) (value float32) {
	return left + right
}

func subtractValue(left, right float32) (value float32) {
	return left - right
}

func multiplyValue(left, right float32) (value float32) {
	return left * right
}

func divideValue(left, right float32) (value float32) {
	return left / right
}
