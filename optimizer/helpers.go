package optimizer

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
)

func validateLearningRate(learningRate float32) (err error) {
	if learningRate <= 0 || f32.IsNaN(learningRate) || f32.IsInf(learningRate, 0) {
		err = fmt.Errorf("optimizer: learning rate must be positive and finite: learningRate=%g", learningRate)
		return err
	}

	return nil
}

func nilOptimizerError(name string) (err error) {
	err = fmt.Errorf("optimizer: %s optimizer is nil", name)
	return err
}

func validateUnitCoefficient(name string, value float32) (err error) {
	if value < 0 || value >= 1 || f32.IsNaN(value) || f32.IsInf(value, 0) {
		err = fmt.Errorf("optimizer: %s must be greater than or equal to 0 and less than 1: %s=%g", name, name, value)
		return err
	}

	return nil
}

func validatePositiveFinite(name string, value float32) (err error) {
	if value <= 0 || f32.IsNaN(value) || f32.IsInf(value, 0) {
		err = fmt.Errorf("optimizer: %s must be positive and finite: %s=%g", name, name, value)
		return err
	}

	return nil
}

func validateParameters(parameters []*Parameter) (err error) {
	var (
		index     int
		parameter *Parameter
	)

	for index, parameter = range parameters {
		if err = parameter.validate(); err != nil {
			err = fmt.Errorf("optimizer: parameter %d invalid: %w", index, err)
			return err
		}
	}

	return nil
}

func resetGradients(parameters []*Parameter) (err error) {
	var parameter *Parameter

	for _, parameter = range parameters {
		if err = parameter.ResetGradient(); err != nil {
			return err
		}
	}

	return nil
}
