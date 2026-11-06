package tensor

import "github.com/itsmontoya/neuralnetwork/internal/f32"

// SoftmaxInto writes stable float32 normalized exponentials along axis.
// Negative axes are normalized; scalars have no axis. Empty values succeed
// after validation. A slice containing NaN, positive infinity, or only negative
// infinities produces NaNs. All errors preserve the disjoint destination.
func SoftmaxInto(input, destination *Tensor, axis int64) (err error) {
	var (
		normalized                 int
		outerSize, innerSize, span int64
	)

	if err = validateTensorType("input", input, Float32); err != nil {
		return err
	}
	if normalized, err = NormalizeAxis(axis, len(input.shape)); err != nil {
		return err
	}
	if err = validateDestination(destination, Float32, input.shape, input); err != nil {
		return err
	}
	if len(input.floats) == 0 {
		return nil
	}
	if outerSize, err = checkedProduct(input.shape[:normalized]); err != nil {
		return err
	}
	if innerSize, err = checkedProduct(input.shape[normalized+1:]); err != nil {
		return err
	}
	if span, err = checkedMultiply(input.shape[normalized], innerSize); err != nil {
		return err
	}
	softmaxFloat32Into(input.floats, destination.floats, int(outerSize), int(input.shape[normalized]), int(innerSize), int(span))
	return nil
}

func softmaxFloat32Into(input, destination []float32, outerSize, axisSize, innerSize, span int) {
	var (
		outer, inner, axis, base, offset int
		maximum, sum                     float32
	)

	for outer = 0; outer < outerSize; outer++ {
		// Checked geometry bounds every coordinate and offset by input length.
		base = outer * span
		for inner = 0; inner < innerSize; inner++ {
			maximum = input[base+inner]
			for axis = 1; axis < axisSize; axis++ {
				offset = base + axis*innerSize + inner
				if input[offset] > maximum {
					maximum = input[offset]
				}
			}
			sum = 0
			for axis = 0; axis < axisSize; axis++ {
				offset = base + axis*innerSize + inner
				destination[offset] = f32.Exp(input[offset] - maximum)
				sum += destination[offset]
			}
			for axis = 0; axis < axisSize; axis++ {
				offset = base + axis*innerSize + inner
				destination[offset] /= sum
			}
		}
	}
}
