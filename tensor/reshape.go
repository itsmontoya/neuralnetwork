package tensor

import "fmt"

// ReshapeShape returns a copied concrete target with the same element count.
// Both shapes must support either tensor type; zero dimensions are literal and
// negative/inferred dimensions are rejected. No numeric payload is allocated.
func ReshapeShape(input, target []int64) (shape []int64, err error) {
	var inputSize, targetSize int64

	if inputSize, _, err = ValidateShape(input, Int64); err != nil {
		return nil, fmt.Errorf("tensor: reshape input shape is invalid: %w", err)
	}
	if targetSize, _, err = ValidateShape(target, Int64); err != nil {
		return nil, fmt.Errorf("tensor: reshape target shape is invalid: %w", err)
	}
	if inputSize != targetSize {
		err = fmt.Errorf("tensor: reshape element counts differ: input=%d target=%d", inputSize, targetSize)
		return nil, err
	}
	shape = make([]int64, len(target))
	copy(shape, target)
	return shape, nil
}

// ReshapeInto copies row-major values into destination's resolved shape.
// Types must match, storage must be disjoint, and all errors preserve destination.
func ReshapeInto(input, destination *Tensor) (err error) {
	var shape []int64

	if err = validateTransformInput(input); err != nil {
		return err
	}
	if err = destination.Validate(); err != nil {
		return fmt.Errorf("tensor: reshape destination is invalid: %w", err)
	}
	if shape, err = ReshapeShape(input.shape, destination.shape); err != nil {
		return err
	}
	err = copyTransformInto(input, destination, shape)
	return err
}
