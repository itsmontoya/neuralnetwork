package tensor

import "github.com/itsmontoya/neuralnetwork/internal/f32"

// ExpInto writes float32 exponentials into a disjoint, shape-matched destination.
// Floating overflow produces infinity; all argument errors preserve destination.
func ExpInto(input, destination *Tensor) (err error) {
	err = mathFloat32Into(input, destination, f32.Exp)
	return err
}

// LogInto writes float32 natural logarithms into a disjoint destination.
// Negative inputs produce NaN and either zero sign produces negative infinity.
// All argument errors preserve destination.
func LogInto(input, destination *Tensor) (err error) {
	err = mathFloat32Into(input, destination, f32.Log)
	return err
}

// SqrtInto writes float32 square roots into a disjoint destination.
// Negative inputs produce NaN; negative zero is preserved. All argument errors
// preserve destination.
func SqrtInto(input, destination *Tensor) (err error) {
	err = mathFloat32Into(input, destination, f32.Sqrt)
	return err
}

// PowInto writes broadcast float32 powers into a disjoint destination.
// Scalars and empty outputs are supported. Math domains and nonfinite values
// follow the standard-library power function; NaN payloads are unspecified.
// All argument errors preserve destination.
func PowInto(left, right, destination *Tensor) (err error) {
	err = binaryFloat32Into(left, right, destination, f32.Pow, nil, nil)
	return err
}

func mathFloat32Into(input, destination *Tensor, operation func(float32) float32) (err error) {
	var (
		index int
		value float32
	)

	if err = validateTensorType("math input", input, Float32); err != nil {
		return err
	}
	if err = validateDestination(destination, Float32, input.shape, input); err != nil {
		return err
	}
	for index, value = range input.floats {
		destination.floats[index] = operation(value)
	}
	return nil
}
