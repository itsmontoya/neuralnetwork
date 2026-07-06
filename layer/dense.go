package layer

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

// NewDense constructs a fully connected layer with zero biases.
func NewDense(inputSize, outputSize int, initializer WeightInitializer) (out *Dense, err error) {
	var (
		weights         *matrix.Matrix
		biases          *matrix.Matrix
		weightParameter *optimizer.Parameter
		biasParameter   *optimizer.Parameter
		d               Dense
	)

	if inputSize <= 0 {
		err = fmt.Errorf("layer: dense input size must be positive: inputSize=%d", inputSize)
		return nil, err
	}

	if outputSize <= 0 {
		err = fmt.Errorf("layer: dense output size must be positive: outputSize=%d", outputSize)
		return nil, err
	}

	if initializer == nil {
		err = errors.New("layer: dense weight initializer is nil")
		return nil, err
	}

	if weights, err = initializer(inputSize, outputSize); err != nil {
		return nil, err
	}

	if err = validateMatrixShape("dense weights", weights, inputSize, outputSize); err != nil {
		return nil, err
	}

	if biases, err = matrix.New(1, outputSize); err != nil {
		return nil, err
	}

	if weightParameter, err = optimizer.NewParameter(weights); err != nil {
		return nil, err
	}

	if biasParameter, err = optimizer.NewParameter(biases); err != nil {
		return nil, err
	}

	d.inputSize = inputSize
	d.outputSize = outputSize
	d.weights = weightParameter
	d.biases = biasParameter
	return &d, nil
}

// Dense applies a fully connected affine transform to batched inputs.
//
// Backward accumulates summed batch gradients in the weight and bias
// parameters. Loss implementations can control mean scaling through the
// output gradients they pass into Backward.
type Dense struct {
	inputSize            int
	outputSize           int
	weights              *optimizer.Parameter
	biases               *optimizer.Parameter
	inputCache           *matrix.Matrix
	outputScratch        *matrix.Matrix
	weightGradient       *matrix.Matrix
	inputGradientScratch *matrix.Matrix
}

// Forward computes input*weights + biases for a batched input matrix.
func (d *Dense) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var batchRows int

	if err = d.validate(); err != nil {
		return nil, err
	}

	if err = d.validateInput(input); err != nil {
		return nil, err
	}

	batchRows = input.Rows()
	if d.outputScratch, err = matrixScratch(d.outputScratch, batchRows, d.outputSize); err != nil {
		return nil, err
	}

	if d.outputScratch == input {
		if d.outputScratch, err = matrix.New(batchRows, d.outputSize); err != nil {
			return nil, err
		}
	}

	if d.inputCache, err = matrixScratch(d.inputCache, batchRows, d.inputSize); err != nil {
		return nil, err
	}

	if err = input.MatMulInto(d.weights.Values(), d.outputScratch); err != nil {
		return nil, err
	}

	if err = d.outputScratch.AddRowVectorInPlace(d.biases.Values()); err != nil {
		return nil, err
	}

	if err = d.inputCache.CopyFrom(input); err != nil {
		return nil, err
	}

	output = d.outputScratch
	return output, nil
}

// Backward accumulates parameter gradients and returns the input gradient.
func (d *Dense) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var batchRows int

	if err = d.validate(); err != nil {
		return nil, err
	}

	if d.inputCache == nil {
		err = errors.New("layer: dense backward called before forward")
		return nil, err
	}

	if err = d.validateOutputGradient(outputGradient); err != nil {
		return nil, err
	}

	batchRows = outputGradient.Rows()
	if d.weightGradient, err = matrixScratch(d.weightGradient, d.inputSize, d.outputSize); err != nil {
		return nil, err
	}

	if d.inputGradientScratch, err = matrixScratch(d.inputGradientScratch, batchRows, d.inputSize); err != nil {
		return nil, err
	}

	if d.inputGradientScratch == outputGradient {
		if d.inputGradientScratch, err = matrix.New(batchRows, d.inputSize); err != nil {
			return nil, err
		}
	}

	if err = d.inputCache.MatMulLeftTransposeInto(outputGradient, d.weightGradient); err != nil {
		return nil, err
	}

	if err = outputGradient.MatMulRightTransposeInto(d.weights.Values(), d.inputGradientScratch); err != nil {
		return nil, err
	}

	if err = d.weights.AccumulateGradient(d.weightGradient); err != nil {
		return nil, err
	}

	if err = outputGradient.AccumulateColumnSumsInto(d.biases.Gradient()); err != nil {
		return nil, err
	}

	inputGradient = d.inputGradientScratch
	return inputGradient, nil
}

// InputSize returns the expected number of input columns.
func (d *Dense) InputSize() (inputSize int) {
	if d == nil {
		return 0
	}

	inputSize = d.inputSize
	return inputSize
}

// OutputSize returns the number of output columns.
func (d *Dense) OutputSize() (outputSize int) {
	if d == nil {
		return 0
	}

	outputSize = d.outputSize
	return outputSize
}

// Weights returns the trainable weight parameter.
func (d *Dense) Weights() (weights *optimizer.Parameter) {
	if d == nil {
		return nil
	}

	weights = d.weights
	return weights
}

// Biases returns the trainable bias parameter.
func (d *Dense) Biases() (biases *optimizer.Parameter) {
	if d == nil {
		return nil
	}

	biases = d.biases
	return biases
}

// Parameters returns the trainable parameters in weight, bias order.
func (d *Dense) Parameters() (parameters []*optimizer.Parameter) {
	if d == nil {
		return nil
	}

	parameters = []*optimizer.Parameter{d.weights, d.biases}
	return parameters
}

// ResetGradients clears all accumulated parameter gradients.
func (d *Dense) ResetGradients() (err error) {
	if err = d.validate(); err != nil {
		return err
	}

	if err = d.weights.ResetGradient(); err != nil {
		return err
	}

	err = d.biases.ResetGradient()
	return err
}

func (d *Dense) validate() (err error) {
	if d == nil {
		err = errors.New("layer: dense layer is nil")
		return err
	}

	if d.inputSize <= 0 {
		err = fmt.Errorf("layer: dense input size must be positive: inputSize=%d", d.inputSize)
		return err
	}

	if d.outputSize <= 0 {
		err = fmt.Errorf("layer: dense output size must be positive: outputSize=%d", d.outputSize)
		return err
	}

	if d.weights == nil {
		err = errors.New("layer: dense weights parameter is nil")
		return err
	}

	if d.biases == nil {
		err = errors.New("layer: dense biases parameter is nil")
		return err
	}

	if err = validateMatrixShape("dense weights", d.weights.Values(), d.inputSize, d.outputSize); err != nil {
		return err
	}

	if err = validateMatrixShape("dense weight gradient", d.weights.Gradient(), d.inputSize, d.outputSize); err != nil {
		return err
	}

	if err = validateMatrixShape("dense biases", d.biases.Values(), 1, d.outputSize); err != nil {
		return err
	}

	if err = validateMatrixShape("dense bias gradient", d.biases.Gradient(), 1, d.outputSize); err != nil {
		return err
	}

	return nil
}

func (d *Dense) validateInput(input *matrix.Matrix) (err error) {
	var (
		rows int
		cols int
	)

	if input == nil {
		err = errors.New("layer: dense input is nil")
		return err
	}

	if err = input.Validate(); err != nil {
		err = fmt.Errorf("layer: dense input invalid: %w", err)
		return err
	}

	rows, cols = input.Shape()
	if cols != d.inputSize {
		err = fmt.Errorf("layer: dense input shape mismatch: got %dx%d, want batch rows x %d", rows, cols, d.inputSize)
		return err
	}

	return nil
}

func (d *Dense) validateOutputGradient(outputGradient *matrix.Matrix) (err error) {
	var (
		gradientRows int
		gradientCols int
		inputRows    int
	)

	if outputGradient == nil {
		err = errors.New("layer: dense output gradient is nil")
		return err
	}

	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("layer: dense output gradient invalid: %w", err)
		return err
	}

	gradientRows, gradientCols = outputGradient.Shape()
	inputRows = d.inputCache.Rows()

	if gradientRows != inputRows || gradientCols != d.outputSize {
		err = fmt.Errorf(
			"layer: dense output gradient shape mismatch: got %dx%d, want %dx%d",
			gradientRows,
			gradientCols,
			inputRows,
			d.outputSize,
		)
		return err
	}

	return nil
}

func validateMatrixShape(name string, m *matrix.Matrix, rows, cols int) (err error) {
	var (
		matrixRows int
		matrixCols int
	)

	if m == nil {
		err = fmt.Errorf("layer: %s is nil", name)
		return err
	}

	if err = m.Validate(); err != nil {
		return err
	}

	matrixRows, matrixCols = m.Shape()
	if matrixRows != rows || matrixCols != cols {
		err = fmt.Errorf("layer: %s shape mismatch: got %dx%d, want %dx%d", name, matrixRows, matrixCols, rows, cols)
		return err
	}

	return nil
}
