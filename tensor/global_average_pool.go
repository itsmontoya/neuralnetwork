package tensor

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
)

// GlobalAveragePoolShape returns [N,C,1,1] for a checked float32 NCHW shape.
// Zero spatial extents are valid; no numeric storage is allocated.
func GlobalAveragePoolShape(input []int64) (shape []int64, err error) {
	if len(input) != 4 {
		err = fmt.Errorf("tensor: global average pooling requires rank-four NCHW input")
		return nil, err
	}
	if _, _, err = ValidateShape(input, Float32); err != nil {
		return nil, fmt.Errorf("tensor: global average pooling input shape is invalid: %w", err)
	}
	shape = []int64{input[0], input[1], 1, 1}
	if _, _, err = ValidateShape(shape, Float32); err != nil {
		return nil, fmt.Errorf("tensor: global average pooling output shape is invalid: %w", err)
	}
	return shape, nil
}

// GlobalAveragePoolInto writes per-channel spatial means into [N,C,1,1].
// A zero spatial reduction produces NaN; input NaNs propagate. Float32 sums
// use row-major order. All errors preserve the disjoint caller-owned destination.
func GlobalAveragePoolInto(input, destination *Tensor) (err error) {
	var (
		shape                 []int64
		spatial               int64
		index, element, start int
		sum                   float32
	)

	if err = validateTensorType("global average pooling input", input, Float32); err != nil {
		return err
	}
	if shape, err = GlobalAveragePoolShape(input.shape); err != nil {
		return err
	}
	if err = validateDestination(destination, Float32, shape, input); err != nil {
		return err
	}
	if len(destination.floats) == 0 {
		return nil
	}
	if spatial, err = checkedProduct(input.shape[2:]); err != nil {
		return err
	}
	for index = range destination.floats {
		if spatial == 0 {
			destination.floats[index] = f32.NaN()
			continue
		}
		// Checked payload lengths bound every channel's contiguous block.
		start, sum = index*int(spatial), 0
		for element = 0; element < int(spatial); element++ {
			sum += input.floats[start+element]
		}
		destination.floats[index] = sum / float32(spatial)
	}
	return nil
}
