package tensor

import "fmt"

// TransposeShape returns an owned shape for an explicit full permutation.
// Negative axes normalize against input rank. Duplicates and missing axes fail;
// only a scalar accepts an empty permutation. Both types' sizes are checked.
func TransposeShape(input, permutation []int64) (shape []int64, err error) {
	var (
		normalized []int
		index      int
		axis       int
	)

	if _, _, err = ValidateShape(input, Int64); err != nil {
		return nil, err
	}
	if len(permutation) != len(input) {
		err = fmt.Errorf("tensor: permutation length differs from rank: permutation=%d rank=%d", len(permutation), len(input))
		return nil, err
	}
	if normalized, err = normalizeAxes(permutation, len(input)); err != nil {
		return nil, err
	}
	shape = make([]int64, len(input))
	for index, axis = range normalized {
		shape[index] = input[axis]
	}
	if _, _, err = ValidateShape(shape, Int64); err != nil {
		return nil, err
	}
	return shape, nil
}

// TransposeInto materializes an explicit permutation into a disjoint destination.
// Float32 bits and int64 values are copied without conversion. All validation
// and stride planning precede writes, and errors preserve destination.
func TransposeInto(input, destination *Tensor, permutation []int64) (err error) {
	var (
		shape, inputStrides, outputStrides []int64
		normalized                         []int
		index, axis, elements              int
		offset, coordinate                 int64
	)

	if err = validateTransformInput(input); err != nil {
		return err
	}
	if shape, err = TransposeShape(input.shape, permutation); err != nil {
		return err
	}
	if err = validateDestination(destination, input.dataType, shape, input); err != nil {
		return err
	}
	elements = int(input.Len())
	if elements == 0 {
		return nil
	}
	if normalized, err = normalizeAxes(permutation, len(input.shape)); err != nil {
		return err
	}
	if inputStrides, err = contiguousStrides(input.shape); err != nil {
		return err
	}
	if outputStrides, err = contiguousStrides(shape); err != nil {
		return err
	}
	for index = 0; index < elements; index++ {
		offset = 0
		for axis = range shape {
			coordinate = int64(index) / outputStrides[axis] % shape[axis]
			// Validated permutation/strides bound each offset by input length.
			offset += coordinate * inputStrides[normalized[axis]]
		}
		if input.dataType == Float32 {
			destination.floats[index] = input.floats[int(offset)]
		} else {
			destination.integers[index] = input.integers[int(offset)]
		}
	}
	return nil
}
