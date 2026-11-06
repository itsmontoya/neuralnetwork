package tensor

import "fmt"

// ConcatShape returns an owned shape for a nonempty list along axis.
// Inputs must have equal ranks and non-axis dimensions. Axis sums and both
// supported types' payload bounds are checked without allocating numeric data.
func ConcatShape(inputs [][]int64, axis int64) (shape []int64, err error) {
	var (
		normalized int
		index      int
		dimension  int
		input      []int64
	)

	if len(inputs) == 0 {
		err = fmt.Errorf("tensor: concat requires at least one input")
		return nil, err
	}
	if normalized, err = NormalizeAxis(axis, len(inputs[0])); err != nil {
		return nil, err
	}
	shape = make([]int64, len(inputs[0]))
	copy(shape, inputs[0])
	shape[normalized] = 0
	for index, input = range inputs {
		if _, _, err = ValidateShape(input, Int64); err != nil {
			return nil, fmt.Errorf("tensor: concat input %d shape is invalid: %w", index, err)
		}
		if len(input) != len(shape) {
			err = fmt.Errorf("tensor: concat input %d rank is %d, want %d", index, len(input), len(shape))
			return nil, err
		}
		for dimension = range shape {
			if dimension != normalized && input[dimension] != shape[dimension] {
				err = fmt.Errorf("tensor: concat input %d dimension differs at axis %d", index, dimension)
				return nil, err
			}
		}
		if shape[normalized], err = checkedAdd(shape[normalized], input[normalized]); err != nil {
			return nil, fmt.Errorf("tensor: concat axis sum is invalid: %w", err)
		}
	}
	if _, _, err = ValidateShape(shape, Int64); err != nil {
		return nil, fmt.Errorf("tensor: concat output shape is invalid: %w", err)
	}
	return shape, nil
}

// ConcatInto copies compatible float32 or int64 inputs along axis.
// Destination must have the exact output type/shape and disjoint storage from
// every input. All returned errors leave destination unchanged.
func ConcatInto(inputs []*Tensor, destination *Tensor, axis int64) (err error) {
	var (
		shapes                   [][]int64
		shape                    []int64
		blocks                   []int
		dataType                 DataType
		normalized, outer, block int
		index                    int
		input                    *Tensor
	)

	if len(inputs) == 0 {
		err = fmt.Errorf("tensor: concat requires at least one input")
		return err
	}
	if err = inputs[0].Validate(); err != nil {
		return fmt.Errorf("tensor: concat input 0 is invalid: %w", err)
	}
	dataType = inputs[0].dataType
	shapes = make([][]int64, len(inputs))
	for index, input = range inputs {
		if err = validateTensorType(fmt.Sprintf("concat input %d", index), input, dataType); err != nil {
			return err
		}
		shapes[index] = input.shape
	}
	if shape, err = ConcatShape(shapes, axis); err != nil {
		return err
	}
	if err = validateDestination(destination, dataType, shape, inputs...); err != nil {
		return err
	}
	if destination.Len() == 0 {
		return nil
	}
	if normalized, err = NormalizeAxis(axis, len(shape)); err != nil {
		return err
	}
	if outer, block, blocks, err = concatBlockSizes(shapes, shape, normalized); err != nil {
		return err
	}
	concatBlocksInto(inputs, destination, outer, block, blocks)
	return nil
}

func concatBlockSizes(inputs [][]int64, output []int64, axis int) (outer, block int, blocks []int, err error) {
	var (
		outerSize, innerSize, size int64
		index                      int
	)

	if outerSize, err = checkedProduct(output[:axis]); err != nil {
		return 0, 0, nil, err
	}
	if innerSize, err = checkedProduct(output[axis+1:]); err != nil {
		return 0, 0, nil, err
	}
	if size, err = checkedMultiply(output[axis], innerSize); err != nil {
		return 0, 0, nil, err
	}
	outer, block = int(outerSize), int(size)
	blocks = make([]int, len(inputs))
	for index = range inputs {
		if size, err = checkedMultiply(inputs[index][axis], innerSize); err != nil {
			return 0, 0, nil, err
		}
		blocks[index] = int(size)
	}
	return outer, block, blocks, nil
}

func concatBlocksInto(inputs []*Tensor, destination *Tensor, outer, block int, blocks []int) {
	var (
		row, index, sourceOffset, outputOffset, size int
		input                                        *Tensor
	)

	for row = 0; row < outer; row++ {
		outputOffset = row * block
		for index, input = range inputs {
			size = blocks[index]
			// Checked block geometry bounds offsets and ends by payload lengths.
			sourceOffset = row * size
			if input.dataType == Float32 {
				copy(destination.floats[outputOffset:outputOffset+size], input.floats[sourceOffset:sourceOffset+size])
			} else {
				copy(destination.integers[outputOffset:outputOffset+size], input.integers[sourceOffset:sourceOffset+size])
			}
			outputOffset += size
		}
	}
}
