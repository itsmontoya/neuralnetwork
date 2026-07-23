package optimizer

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// NewParameter constructs a Parameter with copied values and a zero gradient.
func NewParameter(values *matrix.Matrix) (out *Parameter, err error) {
	var (
		clonedValues *matrix.Matrix
		gradient     *matrix.Matrix
		p            Parameter
	)

	if values == nil {
		err = errors.New("optimizer: parameter values are nil")
		return nil, err
	}

	if clonedValues, err = values.Clone(); err != nil {
		return nil, err
	}

	if gradient, err = matrix.New(clonedValues.Rows(), clonedValues.Cols()); err != nil {
		return nil, err
	}

	p.values = clonedValues
	p.gradient = gradient
	return &p, nil
}

// Parameter stores trainable values and their accumulated gradient.
type Parameter struct {
	values   *matrix.Matrix
	gradient *matrix.Matrix
}

// Values returns the mutable parameter values used during optimization.
func (p *Parameter) Values() (values *matrix.Matrix) {
	if p == nil {
		return nil
	}

	values = p.values
	return values
}

// Gradient returns the mutable accumulated gradient for the parameter values.
func (p *Parameter) Gradient() (gradient *matrix.Matrix) {
	if p == nil {
		return nil
	}

	gradient = p.gradient
	return gradient
}

// AccumulateGradient adds gradient to the parameter's accumulated gradient.
func (p *Parameter) AccumulateGradient(gradient *matrix.Matrix) (err error) {
	if err = p.validate(); err != nil {
		return err
	}

	err = p.gradient.AddInPlace(gradient)
	return err
}

// ResetGradient sets every accumulated gradient value to zero.
func (p *Parameter) ResetGradient() (err error) {
	var handled bool

	if err = p.validate(); err != nil {
		return err
	}
	if handled, err = device.Reset(p.gradient); err != nil {
		return err
	}
	if handled {
		return nil
	}

	err = p.gradient.Fill(0)
	return err
}

func (p *Parameter) validate() (err error) {
	var (
		valueRows    int
		valueCols    int
		gradientRows int
		gradientCols int
	)

	if p == nil {
		err = errors.New("optimizer: parameter is nil")
		return err
	}

	if p.values == nil {
		err = errors.New("optimizer: parameter values are nil")
		return err
	}

	if p.gradient == nil {
		err = errors.New("optimizer: parameter gradient is nil")
		return err
	}

	valueRows, valueCols = p.values.Shape()
	gradientRows, gradientCols = p.gradient.Shape()
	if valueRows != gradientRows || valueCols != gradientCols {
		err = fmt.Errorf(
			"optimizer: parameter gradient shape mismatch: values %dx%d, gradient %dx%d",
			valueRows,
			valueCols,
			gradientRows,
			gradientCols,
		)
		return err
	}

	return nil
}
