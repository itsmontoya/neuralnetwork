package tensor

import "fmt"

// SqueezeShape returns an owned shape with exactly the listed unit axes removed.
// Negative axes normalize against input rank. Empty axes mean an identity shape,
// and duplicate/nonunit axes fail. Both types' sizes are checked without payloads.
func SqueezeShape(input, axes []int64) (shape []int64, err error) {
	var (
		normalized []int
		removed    []bool
		axis       int
		dimension  int64
	)

	if _, _, err = ValidateShape(input, Int64); err != nil {
		return nil, err
	}
	if normalized, err = normalizeAxes(axes, len(input)); err != nil {
		return nil, err
	}
	removed = make([]bool, len(input))
	for _, axis = range normalized {
		if input[axis] != 1 {
			err = fmt.Errorf("tensor: squeeze axis is not unit: axis=%d dimension=%d", axis, input[axis])
			return nil, err
		}
		removed[axis] = true
	}
	shape = make([]int64, 0, len(input)-len(normalized))
	for axis, dimension = range input {
		if !removed[axis] {
			shape = append(shape, dimension)
		}
	}
	return shape, nil
}

// SqueezeInto copies values while removing exactly the supplied unit axes.
// Both types are preserved. Storage must be disjoint; errors preserve destination.
func SqueezeInto(input, destination *Tensor, axes []int64) (err error) {
	var shape []int64

	if err = validateTransformInput(input); err != nil {
		return err
	}
	if shape, err = SqueezeShape(input.shape, axes); err != nil {
		return err
	}
	err = copyTransformInto(input, destination, shape)
	return err
}
