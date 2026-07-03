package layer

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// NewDropout constructs an inverted dropout layer with training mode enabled.
func NewDropout(rate float64, random *rand.Rand) (out *Dropout, err error) {
	if err = validateDropoutRate(rate); err != nil {
		return nil, err
	}

	if random == nil {
		err = errors.New("layer: dropout random source is nil")
		return nil, err
	}

	var d Dropout
	d.rate = rate
	d.random = random
	d.training = true
	return &d, nil
}

// Dropout randomly zeros activations during training and acts as identity
// during evaluation.
//
// Training forward passes use inverted dropout: kept activations are scaled by
// 1/(1-rate), so evaluation passes do not need additional scaling.
type Dropout struct {
	rate            float64
	random          *rand.Rand
	training        bool
	maskCache       *matrix.Matrix
	forwardRows     int
	forwardCols     int
	forwardCalled   bool
	forwardTraining bool
}

// Forward applies dropout in training mode and identity in evaluation mode.
func (d *Dropout) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var (
		rows         int
		cols         int
		inputValues  []float64
		outputValues []float64
		maskValues   []float64
		index        int
		scale        float64
		mask         float64
	)

	if err = d.validate(); err != nil {
		return nil, err
	}

	if input == nil {
		err = errors.New("layer: dropout input is nil")
		return nil, err
	}

	if err = input.Validate(); err != nil {
		err = fmt.Errorf("layer: dropout input invalid: %w", err)
		return nil, err
	}

	rows, cols = input.Shape()
	d.forwardRows = rows
	d.forwardCols = cols
	d.forwardCalled = true
	d.forwardTraining = d.training
	d.maskCache = nil

	if !d.training {
		output, err = input.Clone()
		return output, err
	}

	if inputValues, err = input.Values(); err != nil {
		return nil, err
	}

	outputValues = make([]float64, len(inputValues))
	maskValues = make([]float64, len(inputValues))
	scale = 1 / (1 - d.rate)

	for index = range inputValues {
		mask = scale
		if d.rate > 0 && d.random.Float64() < d.rate {
			mask = 0
		}

		maskValues[index] = mask
		outputValues[index] = inputValues[index] * mask
	}

	if output, err = matrix.FromSlice(rows, cols, outputValues); err != nil {
		return nil, err
	}

	if d.maskCache, err = matrix.FromSlice(rows, cols, maskValues); err != nil {
		return nil, err
	}

	return output, nil
}

// Backward propagates gradients through the dropout mask from the last forward pass.
func (d *Dropout) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	if err = d.validate(); err != nil {
		return nil, err
	}

	if !d.forwardCalled {
		err = errors.New("layer: dropout backward called before forward")
		return nil, err
	}

	if err = d.validateOutputGradient(outputGradient); err != nil {
		return nil, err
	}

	if !d.forwardTraining {
		inputGradient, err = outputGradient.Clone()
		return inputGradient, err
	}

	if d.maskCache == nil {
		err = errors.New("layer: dropout mask cache is nil")
		return nil, err
	}

	inputGradient, err = outputGradient.MultiplyElements(d.maskCache)
	return inputGradient, err
}

// Rate returns the probability that an activation is dropped during training.
func (d *Dropout) Rate() (rate float64) {
	if d == nil {
		return 0
	}

	rate = d.rate
	return rate
}

// SetTraining updates whether forward passes apply dropout.
func (d *Dropout) SetTraining(training bool) {
	if d == nil {
		return
	}

	d.training = training
}

// Training reports whether forward passes apply dropout.
func (d *Dropout) Training() (training bool) {
	if d == nil {
		return false
	}

	training = d.training
	return training
}

func (d *Dropout) validate() (err error) {
	if d == nil {
		err = errors.New("layer: dropout layer is nil")
		return err
	}

	if err = validateDropoutRate(d.rate); err != nil {
		return err
	}

	if d.random == nil {
		err = errors.New("layer: dropout random source is nil")
		return err
	}

	return nil
}

func (d *Dropout) validateOutputGradient(outputGradient *matrix.Matrix) (err error) {
	var (
		rows int
		cols int
	)

	if outputGradient == nil {
		err = errors.New("layer: dropout output gradient is nil")
		return err
	}

	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("layer: dropout output gradient invalid: %w", err)
		return err
	}

	rows, cols = outputGradient.Shape()
	if rows != d.forwardRows || cols != d.forwardCols {
		err = fmt.Errorf(
			"layer: dropout output gradient shape mismatch: got %dx%d, want %dx%d",
			rows,
			cols,
			d.forwardRows,
			d.forwardCols,
		)
		return err
	}

	return nil
}

func validateDropoutRate(rate float64) (err error) {
	if rate < 0 || rate >= 1 || math.IsNaN(rate) || math.IsInf(rate, 0) {
		err = fmt.Errorf("layer: dropout rate must be greater than or equal to 0 and less than 1: rate=%g", rate)
		return err
	}

	return nil
}
