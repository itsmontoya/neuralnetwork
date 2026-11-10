package tensor

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// AddInto writes the broadcast float32 sum into a disjoint destination.
// All argument errors leave destination unchanged, including for empty results.
func AddInto(left, right, destination *Tensor) (err error) {
	err = binaryFloat32Into(left, right, destination, addValue, (*matrix.Matrix).AddInto)
	return err
}

// SubInto writes the broadcast float32 difference into a disjoint destination.
// All argument errors leave destination unchanged, including for empty results.
func SubInto(left, right, destination *Tensor) (err error) {
	err = binaryFloat32Into(left, right, destination, subtractValue, (*matrix.Matrix).SubtractInto)
	return err
}

// MulInto writes the broadcast float32 product into a disjoint destination.
// All argument errors leave destination unchanged, including for empty results.
func MulInto(left, right, destination *Tensor) (err error) {
	err = binaryFloat32Into(left, right, destination, multiplyValue, (*matrix.Matrix).MultiplyElementsInto)
	return err
}

// DivInto writes the broadcast float32 quotient into a disjoint destination.
// Floating division preserves signed zero/infinity and yields NaN for invalid
// quotients, including zero divided by zero. NaN payloads are unspecified.
// All argument errors leave destination unchanged, including for empty results.
func DivInto(left, right, destination *Tensor) (err error) {
	err = binaryFloat32Into(left, right, destination, divideValue, nil)
	return err
}

func binaryFloat32Into(left, right, destination *Tensor, operation func(float32, float32) float32, matrixOperation func(*matrix.Matrix, *matrix.Matrix, *matrix.Matrix) error) (err error) {
	var shape []int64

	if err = validateTensorType("left", left, Float32); err != nil {
		return err
	}
	if err = validateTensorType("right", right, Float32); err != nil {
		return err
	}
	if shape, err = BroadcastShape(left.shape, right.shape); err != nil {
		return err
	}
	if err = validateDestination(destination, Float32, shape, left, right); err != nil {
		return err
	}
	if len(destination.floats) == 0 {
		return nil
	}
	if matrixOperation != nil && shapesEqual(left.shape, right.shape) {
		err = matrixBinaryInto(left, right, destination, matrixOperation)
		return err
	}
	err = broadcastFloat32Into(left, right, destination, operation)
	return err
}

func matrixBinaryInto(left, right, destination *Tensor, operation func(*matrix.Matrix, *matrix.Matrix, *matrix.Matrix) error) (err error) {
	var (
		leftMatrix, rightMatrix, output *matrix.Matrix
		values                          []float32
	)

	if leftMatrix, err = matrix.FromSlice(1, len(left.floats), left.floats); err != nil {
		return fmt.Errorf("tensor: left matrix adapter: %w", err)
	}
	if rightMatrix, err = matrix.FromSlice(1, len(right.floats), right.floats); err != nil {
		return fmt.Errorf("tensor: right matrix adapter: %w", err)
	}
	if output, err = matrix.New(1, len(destination.floats)); err != nil {
		return fmt.Errorf("tensor: destination matrix adapter: %w", err)
	}
	if err = operation(leftMatrix, rightMatrix, output); err != nil {
		return fmt.Errorf("tensor: matrix arithmetic: %w", err)
	}
	if values, err = output.Values(); err != nil {
		return fmt.Errorf("tensor: matrix arithmetic values: %w", err)
	}
	copy(destination.floats, values)
	return nil
}

func broadcastFloat32Into(left, right, destination *Tensor, operation func(float32, float32) float32) (err error) {
	var (
		coordinates, leftStrides, rightStrides []int64
		leftOffset, rightOffset                int64
		index, axis                            int
	)

	if leftStrides, err = broadcastStrides(left.shape, destination.shape); err != nil {
		return err
	}

	if rightStrides, err = broadcastStrides(right.shape, destination.shape); err != nil {
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
	var (
		inputStrides []int64
		axis, shift  int
	)

	if err = ValidateBroadcastTo(input, output); err != nil {
		return nil, err
	}
	if inputStrides, err = contiguousStrides(input); err != nil {
		return nil, err
	}
	strides = make([]int64, len(output))
	shift = len(output) - len(input)
	for axis = range input {
		if input[axis] != 1 {
			strides[axis+shift] = inputStrides[axis]
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
