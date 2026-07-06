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
	rate                 float64
	random               *rand.Rand
	training             bool
	maskCache            *matrix.Matrix
	outputScratch        *matrix.Matrix
	inputGradientScratch *matrix.Matrix
	inputValues          []float64
	outputValues         []float64
	maskValues           []float64
	forwardRows          int
	forwardCols          int
	forwardCalled        bool
	forwardTraining      bool
}

// Forward applies dropout in training mode and identity in evaluation mode.
func (d *Dropout) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var (
		rows       int
		cols       int
		index      int
		scale      float64
		mask       float64
		valueCount int
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
	valueCount = rows * cols
	if d.outputScratch, err = matrixScratch(d.outputScratch, rows, cols); err != nil {
		return nil, err
	}

	d.forwardRows = rows
	d.forwardCols = cols
	d.forwardCalled = true
	d.forwardTraining = d.training

	if !d.training {
		d.maskCache = nil
		if err = d.outputScratch.CopyFrom(input); err != nil {
			return nil, err
		}

		output = d.outputScratch
		return output, nil
	}

	if d.maskCache, err = matrixScratch(d.maskCache, rows, cols); err != nil {
		return nil, err
	}

	if d.rate == 0 {
		if err = d.outputScratch.CopyFrom(input); err != nil {
			return nil, err
		}

		if err = d.maskCache.Fill(1); err != nil {
			return nil, err
		}

		output = d.outputScratch
		return output, nil
	}

	d.inputValues = floatScratch(d.inputValues, valueCount)
	d.outputValues = floatScratch(d.outputValues, valueCount)
	d.maskValues = floatScratch(d.maskValues, valueCount)

	if err = input.ValuesInto(d.inputValues); err != nil {
		return nil, err
	}

	scale = 1 / (1 - d.rate)
	for index = range d.inputValues {
		mask = scale
		if d.random.Float64() < d.rate {
			mask = 0
		}

		d.maskValues[index] = mask
		d.outputValues[index] = d.inputValues[index] * mask
	}

	if err = d.outputScratch.CopyValuesFrom(d.outputValues); err != nil {
		return nil, err
	}

	if err = d.maskCache.CopyValuesFrom(d.maskValues); err != nil {
		return nil, err
	}

	output = d.outputScratch
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
		if d.inputGradientScratch, err = matrixScratch(d.inputGradientScratch, d.forwardRows, d.forwardCols); err != nil {
			return nil, err
		}

		if err = d.inputGradientScratch.CopyFrom(outputGradient); err != nil {
			return nil, err
		}

		inputGradient = d.inputGradientScratch
		return inputGradient, nil
	}

	if d.maskCache == nil {
		err = errors.New("layer: dropout mask cache is nil")
		return nil, err
	}

	if d.inputGradientScratch, err = matrixScratch(d.inputGradientScratch, d.forwardRows, d.forwardCols); err != nil {
		return nil, err
	}

	if err = outputGradient.MultiplyElementsInto(d.maskCache, d.inputGradientScratch); err != nil {
		return nil, err
	}

	inputGradient = d.inputGradientScratch
	return inputGradient, nil
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
