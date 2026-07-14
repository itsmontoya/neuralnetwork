package optimizer

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
)

func validateRegularizers(regularizers []Regularizer) (err error) {
	var (
		index       int
		regularizer Regularizer
	)

	if len(regularizers) == 0 {
		err = errors.New("optimizer: regularized optimizer requires at least one regularizer")
		return err
	}

	for index, regularizer = range regularizers {
		if regularizer == nil {
			err = fmt.Errorf("optimizer: regularizer %d is nil", index)
			return err
		}
	}

	return nil
}

func validateRegularizationCoefficient(name string, coefficient float32) (err error) {
	if coefficient < 0 || f32.IsNaN(coefficient) || f32.IsInf(coefficient, 0) {
		err = fmt.Errorf("optimizer: %s coefficient must be non-negative and finite: coefficient=%g", name, coefficient)
		return err
	}

	return nil
}

func nilRegularizerError(name string) (err error) {
	err = fmt.Errorf("optimizer: %s regularizer is nil", name)
	return err
}

func applyRegularizationGradient(parameters []*Parameter, gradientForValue func(float32) float32) (err error) {
	var parameter *Parameter

	if gradientForValue == nil {
		err = errors.New("optimizer: regularization gradient function is nil")
		return err
	}

	if err = validateParameters(parameters); err != nil {
		return err
	}

	for _, parameter = range parameters {
		if err = parameter.Gradient().AddMappedInPlace(parameter.Values(), gradientForValue); err != nil {
			return err
		}
	}

	return nil
}
