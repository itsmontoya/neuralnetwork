package tensor

import "github.com/itsmontoya/neuralnetwork/internal/f32"

const (
	reductionSum = iota
	reductionMean
	reductionMax
)

// ReductionShape returns a checked float32 shape for explicit, distinct axes.
// Negative axes normalize against input rank. Empty axes mean identity; reduced
// dimensions become ones when keepDimensions is true, or are otherwise removed.
// No numeric payload is allocated.
func ReductionShape(input, axes []int64, keepDimensions bool) (shape []int64, err error) {
	shape, _, err = reductionShape(input, axes, keepDimensions)
	return shape, err
}

// ReduceSumInto writes row-major float32 sums into a disjoint destination.
// Empty reductions produce positive zero. Empty axes copy input unchanged.
// All argument errors preserve destination; input NaNs propagate.
func ReduceSumInto(input, destination *Tensor, axes []int64, keepDimensions bool) (err error) {
	err = reduceFloat32Into(input, destination, axes, keepDimensions, reductionSum)
	return err
}

// ReduceMeanInto writes row-major float32 means into a disjoint destination.
// Empty reductions produce NaN. Empty axes copy input unchanged. All argument
// errors preserve destination; input NaNs propagate.
func ReduceMeanInto(input, destination *Tensor, axes []int64, keepDimensions bool) (err error) {
	err = reduceFloat32Into(input, destination, axes, keepDimensions, reductionMean)
	return err
}

// ReduceMaxInto writes float32 maxima into a disjoint destination.
// Empty reductions produce negative infinity. Empty axes copy input unchanged.
// NaNs propagate; zero-tie signs are not promised across backends. All argument
// errors preserve destination.
func ReduceMaxInto(input, destination *Tensor, axes []int64, keepDimensions bool) (err error) {
	err = reduceFloat32Into(input, destination, axes, keepDimensions, reductionMax)
	return err
}

func reductionShape(input, axes []int64, keepDimensions bool) (shape []int64, normalized []int, err error) {
	var (
		reduced   []bool
		axis      int
		dimension int64
	)

	if _, _, err = ValidateShape(input, Float32); err != nil {
		return nil, nil, err
	}
	if normalized, err = normalizeAxes(axes, len(input)); err != nil {
		return nil, nil, err
	}
	reduced = make([]bool, len(input))
	for _, axis = range normalized {
		reduced[axis] = true
	}
	shape = make([]int64, 0, len(input))
	for axis, dimension = range input {
		if !reduced[axis] {
			shape = append(shape, dimension)
		} else if keepDimensions {
			shape = append(shape, 1)
		}
	}
	if _, _, err = ValidateShape(shape, Float32); err != nil {
		return nil, nil, err
	}
	return shape, normalized, nil
}

func reduceFloat32Into(input, destination *Tensor, axes []int64, keepDimensions bool, operation int) (err error) {
	var (
		shape, dimensions, inputStrides, outputStrides []int64
		normalized                                     []int
		count                                          int64
		index, axis                                    int
	)

	if err = validateTensorType("reduction input", input, Float32); err != nil {
		return err
	}
	if shape, normalized, err = reductionShape(input.shape, axes, keepDimensions); err != nil {
		return err
	}
	if err = validateDestination(destination, Float32, shape, input); err != nil {
		return err
	}
	if len(destination.floats) == 0 {
		return nil
	}
	if len(normalized) == 0 {
		copy(destination.floats, input.floats)
		return nil
	}
	dimensions = make([]int64, len(normalized))
	for index, axis = range normalized {
		dimensions[index] = input.shape[axis]
	}
	if count, err = checkedProduct(dimensions); err != nil {
		return err
	}
	if count > 0 {
		if inputStrides, outputStrides, err = reductionStrides(input.shape, shape, normalized, keepDimensions); err != nil {
			return err
		}
	}
	initializeReduction(destination.floats, operation, count)
	if count == 0 {
		return nil
	}
	reduceValuesInto(input, destination, inputStrides, outputStrides, operation)
	if operation == reductionMean {
		for index = range destination.floats {
			destination.floats[index] /= float32(count)
		}
	}
	return nil
}

// reductionStrides maps nonempty input coordinates to checked output offsets.
func reductionStrides(input, output []int64, normalized []int, keepDimensions bool) (inputStrides, mapped []int64, err error) {
	var (
		outputStrides []int64
		reduced       []bool
		axis          int
		outputAxis    int
	)

	if inputStrides, err = contiguousStrides(input); err != nil {
		return nil, nil, err
	}
	if outputStrides, err = contiguousStrides(output); err != nil {
		return nil, nil, err
	}
	reduced = make([]bool, len(input))
	for _, axis = range normalized {
		reduced[axis] = true
	}
	mapped = make([]int64, len(input))
	for axis = range input {
		if !reduced[axis] {
			mapped[axis] = outputStrides[outputAxis]
		}
		if !reduced[axis] || keepDimensions {
			outputAxis++
		}
	}
	return inputStrides, mapped, nil
}

func initializeReduction(values []float32, operation int, count int64) {
	var (
		index int
		value float32
	)

	if operation == reductionMax {
		value = f32.Inf(-1)
	} else if operation == reductionMean && count == 0 {
		value = f32.NaN()
	}
	for index = range values {
		values[index] = value
	}
}

func reduceValuesInto(input, destination *Tensor, inputStrides, outputStrides []int64, operation int) {
	var (
		index, axis int
		offset      int64
		coordinate  int64
		value       float32
	)

	for index, value = range input.floats {
		offset = 0
		for axis = range input.shape {
			coordinate = int64(index) / inputStrides[axis] % input.shape[axis]
			// Checked shapes and mapped strides bound offsets by output length.
			offset += coordinate * outputStrides[axis]
		}
		if operation != reductionMax {
			destination.floats[int(offset)] += value
		} else if value > destination.floats[int(offset)] || f32.IsNaN(value) {
			destination.floats[int(offset)] = value
		}
	}
}
