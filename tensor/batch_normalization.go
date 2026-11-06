package tensor

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
)

// BatchNormalizationInto writes stored-statistics float32 normalization over
// [N,C,...] into a disjoint destination. Scale, bias, mean, and variance must
// have shape [C]. Epsilon must be finite and nonnegative; floating domains may
// produce NaN or infinity. Empty inputs succeed after full validation. All
// returned errors preserve destination. No statistics or training state change.
func BatchNormalizationInto(input, scale, bias, mean, variance, destination *Tensor, epsilon float32) (err error) {
	var (
		spatial        int64
		index, channel int
		value          float32
	)

	if err = validateTensorType("batch normalization input", input, Float32); err != nil {
		return err
	}
	if len(input.shape) < 2 {
		err = fmt.Errorf("tensor: batch normalization input must have rank at least two")
		return err
	}
	if epsilon < 0 || f32.IsNaN(epsilon) || f32.IsInf(epsilon, 0) {
		err = fmt.Errorf("tensor: batch normalization epsilon must be finite and nonnegative: %g", epsilon)
		return err
	}
	if err = validateNormalizationParameter("scale", scale, input.shape[1]); err != nil {
		return err
	}
	if err = validateNormalizationParameter("bias", bias, input.shape[1]); err != nil {
		return err
	}
	if err = validateNormalizationParameter("mean", mean, input.shape[1]); err != nil {
		return err
	}
	if err = validateNormalizationParameter("variance", variance, input.shape[1]); err != nil {
		return err
	}
	if err = validateDestination(destination, Float32, input.shape, input, scale, bias, mean, variance); err != nil {
		return err
	}
	if len(input.floats) == 0 {
		return nil
	}
	if spatial, err = checkedProduct(input.shape[2:]); err != nil {
		return err
	}
	for index, value = range input.floats {
		// Nonempty checked payloads have positive channel/spatial dimensions;
		// their sizes and this channel conversion are bounded by input length.
		channel = (index / int(spatial)) % int(input.shape[1])
		destination.floats[index] = scale.floats[channel]*(value-mean.floats[channel])/f32.Sqrt(variance.floats[channel]+epsilon) + bias.floats[channel]
	}
	return nil
}

func validateNormalizationParameter(name string, parameter *Tensor, channels int64) (err error) {
	if err = validateTensorType("batch normalization "+name, parameter, Float32); err != nil {
		return err
	}
	if !shapesEqual(parameter.shape, []int64{channels}) {
		err = fmt.Errorf("tensor: batch normalization %s shape is %v, want [%d]", name, parameter.shape, channels)
		return err
	}
	return nil
}
