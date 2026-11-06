package tensor

import "fmt"

func validateTransformInput(input *Tensor) (err error) {
	if err = input.Validate(); err != nil {
		return fmt.Errorf("tensor: transform input is invalid: %w", err)
	}
	return nil
}

// copyTransformInto requires a validated input and checked output shape.
func copyTransformInto(input, destination *Tensor, shape []int64) (err error) {
	if err = validateDestination(destination, input.dataType, shape, input); err != nil {
		return err
	}
	if input.dataType == Float32 {
		copy(destination.floats, input.floats)
	} else {
		copy(destination.integers, input.integers)
	}
	return nil
}

// normalizeAxes preserves order and rejects duplicates after normalization.
func normalizeAxes(axes []int64, rank int) (normalized []int, err error) {
	var (
		seen  []bool
		index int
		axis  int64
	)

	if rank < 0 {
		err = fmt.Errorf("tensor: axes rank is negative: %d", rank)
		return nil, err
	}
	normalized = make([]int, len(axes))
	seen = make([]bool, rank)
	for index, axis = range axes {
		if normalized[index], err = NormalizeAxis(axis, rank); err != nil {
			return nil, err
		}
		if seen[normalized[index]] {
			err = fmt.Errorf("tensor: duplicate normalized axis: %d", normalized[index])
			return nil, err
		}
		seen[normalized[index]] = true
	}
	return normalized, nil
}
