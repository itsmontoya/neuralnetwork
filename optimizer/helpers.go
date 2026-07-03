package optimizer

import (
	"fmt"
	"math"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func validateLearningRate(learningRate float64) (err error) {
	if learningRate <= 0 || math.IsNaN(learningRate) || math.IsInf(learningRate, 0) {
		err = fmt.Errorf("optimizer: learning rate must be positive and finite: learningRate=%g", learningRate)
		return err
	}

	return nil
}

func nilOptimizerError(name string) (err error) {
	err = fmt.Errorf("optimizer: %s optimizer is nil", name)
	return err
}

func validateUnitCoefficient(name string, value float64) (err error) {
	if value < 0 || value >= 1 || math.IsNaN(value) || math.IsInf(value, 0) {
		err = fmt.Errorf("optimizer: %s must be greater than or equal to 0 and less than 1: %s=%g", name, name, value)
		return err
	}

	return nil
}

func validatePositiveFinite(name string, value float64) (err error) {
	if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
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

func parameterValues(parameter *Parameter) (rows, cols int, values, gradients []float64, err error) {
	if err = parameter.validate(); err != nil {
		return 0, 0, nil, nil, err
	}

	if values, err = parameter.Values().Values(); err != nil {
		return 0, 0, nil, nil, err
	}

	if gradients, err = parameter.Gradient().Values(); err != nil {
		return 0, 0, nil, nil, err
	}

	rows, cols = parameter.Values().Shape()
	return rows, cols, values, gradients, nil
}

func matrixValues(source *matrix.Matrix) (rows, cols int, values []float64, err error) {
	if values, err = source.Values(); err != nil {
		return 0, 0, nil, err
	}

	rows, cols = source.Shape()
	return rows, cols, values, nil
}

func copyMatrixValues(destination *matrix.Matrix, rows, cols int, values []float64) (err error) {
	var next *matrix.Matrix

	if next, err = matrix.FromSlice(rows, cols, values); err != nil {
		return err
	}

	err = destination.CopyFrom(next)
	return err
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
