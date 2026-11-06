package tensor

import "fmt"

// GatherShape returns data's prefix, the indices shape, and data's suffix.
// Data must have an axis. Scalar and empty indices shapes are supported; both
// data types' payload bounds are checked without allocating numeric storage.
func GatherShape(data, indices []int64, axis int64) (shape []int64, err error) {
	var (
		normalized int
		rank       int64
	)

	if _, _, err = ValidateShape(data, Int64); err != nil {
		return nil, fmt.Errorf("tensor: gather data shape is invalid: %w", err)
	}
	if _, _, err = ValidateShape(indices, Int64); err != nil {
		return nil, fmt.Errorf("tensor: gather indices shape is invalid: %w", err)
	}
	if normalized, err = NormalizeAxis(axis, len(data)); err != nil {
		return nil, err
	}
	if rank, err = checkedAdd(int64(len(data)-1), int64(len(indices))); err != nil {
		return nil, err
	}
	shape = make([]int64, 0, int(rank))
	shape = append(shape, data[:normalized]...)
	shape = append(shape, indices...)
	shape = append(shape, data[normalized+1:]...)
	if _, _, err = ValidateShape(shape, Int64); err != nil {
		return nil, fmt.Errorf("tensor: gather output shape is invalid: %w", err)
	}
	return shape, nil
}

// GatherInto copies float32 or int64 data selected by int64 indices along axis.
// Negative indices count from the axis end. Every index is validated even for
// empty results. Destination must be disjoint from data and indices; all returned
// errors leave it unchanged. Scalar and multidimensional indices preserve shape.
func GatherInto(data, indices, destination *Tensor, axis int64) (err error) {
	var (
		shape                    []int64
		normalized, outer, inner int
		dimension, index         int64
		position                 int
	)

	if err = data.Validate(); err != nil {
		return fmt.Errorf("tensor: gather data is invalid: %w", err)
	}
	if err = validateTensorType("gather indices", indices, Int64); err != nil {
		return err
	}
	if shape, err = GatherShape(data.shape, indices.shape, axis); err != nil {
		return err
	}
	if normalized, err = NormalizeAxis(axis, len(data.shape)); err != nil {
		return err
	}
	if err = validateDestination(destination, data.dataType, shape, data, indices); err != nil {
		return err
	}
	dimension = data.shape[normalized]
	for position, index = range indices.integers {
		if index < -dimension || index >= dimension {
			err = fmt.Errorf("tensor: gather index is out of range: position=%d index=%d dimension=%d", position, index, dimension)
			return err
		}
	}
	if destination.Len() == 0 {
		return nil
	}
	if outer, inner, err = gatherBlockSizes(data.shape, normalized, len(indices.integers)); err != nil {
		return err
	}
	gatherBlocksInto(data, indices, destination, normalized, outer, inner)
	return nil
}

func gatherBlockSizes(shape []int64, axis, indices int) (outer, inner int, err error) {
	var outerSize, innerSize int64

	if outerSize, err = checkedProduct(shape[:axis]); err != nil {
		return 0, 0, err
	}
	if innerSize, err = checkedProduct(shape[axis+1:]); err != nil {
		return 0, 0, err
	}
	if _, err = checkedMultiply(shape[axis], innerSize); err != nil {
		return 0, 0, err
	}
	if _, err = checkedMultiply(int64(indices), innerSize); err != nil {
		return 0, 0, err
	}
	return int(outerSize), int(innerSize), nil
}

func gatherBlocksInto(data, indices, destination *Tensor, axis, outer, inner int) {
	var (
		row, position, sourceOffset, outputOffset int
		dimension, index                          int64
	)

	dimension = data.shape[axis]
	for row = 0; row < outer; row++ {
		for position, index = range indices.integers {
			if index < 0 {
				index += dimension
			}
			// Validated indices and checked block geometry bound offsets and
			// conversions by source/destination payload lengths.
			sourceOffset = (row*int(dimension) + int(index)) * inner
			outputOffset = (row*len(indices.integers) + position) * inner
			if data.dataType == Float32 {
				copy(destination.floats[outputOffset:outputOffset+inner], data.floats[sourceOffset:sourceOffset+inner])
			} else {
				copy(destination.integers[outputOffset:outputOffset+inner], data.integers[sourceOffset:sourceOffset+inner])
			}
		}
	}
}
