package layer

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/scratch"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// NewLastStep constructs a validating sequence-to-dense adapter.
func NewLastStep(inputShape SequenceShape) (out *LastStep, err error) {
	if err = inputShape.validate(); err != nil {
		err = fmt.Errorf(
			"layer: last step input shape invalid: steps=%d featureSize=%d: %w",
			inputShape.Steps(),
			inputShape.FeatureSize(),
			err,
		)
		return nil, err
	}

	var l LastStep
	l.inputShape = inputShape
	return &l, nil
}

// LastStep selects the final feature vector from each flattened sequence row.
type LastStep struct {
	inputShape               SequenceShape
	outputPool               scratch.MatrixPool
	outputScratch            *matrix.Matrix
	inputGradientPool        scratch.MatrixPool
	inputGradientScratch     *matrix.Matrix
	inputValuesPool          scratch.Float32Pool
	inputValues              []float32
	outputValuesPool         scratch.Float32Pool
	outputValues             []float32
	outputGradientValuesPool scratch.Float32Pool
	outputGradientValues     []float32
	inputGradientValuesPool  scratch.Float32Pool
	inputGradientValues      []float32
	forwardRows              int
	forwardCalled            bool
}

// Forward copies the final step from each flattened sequence row.
func (l *LastStep) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var rows int

	if err = l.validate(); err != nil {
		return nil, err
	}

	if rows, err = l.validateInput(input); err != nil {
		return nil, err
	}

	if err = l.ensureForwardScratch(rows, input); err != nil {
		return nil, err
	}

	if err = input.ValuesInto(l.inputValues); err != nil {
		err = fmt.Errorf("layer: last step copy input values: %w", err)
		return nil, err
	}

	l.forwardInto(rows)
	if err = l.outputScratch.CopyValuesFrom(l.outputValues); err != nil {
		err = fmt.Errorf("layer: last step store output values: %w", err)
		return nil, err
	}

	l.forwardRows = rows
	l.forwardCalled = true
	output = l.outputScratch
	return output, nil
}

// Backward routes gradients only to the final sequence step.
func (l *LastStep) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var rows int

	if err = l.validate(); err != nil {
		return nil, err
	}

	if !l.forwardCalled {
		err = errors.New("layer: last step backward called before forward")
		return nil, err
	}

	if rows, err = l.validateOutputGradient(outputGradient); err != nil {
		return nil, err
	}

	if err = l.ensureBackwardScratch(rows, outputGradient); err != nil {
		return nil, err
	}

	if err = outputGradient.ValuesInto(l.outputGradientValues); err != nil {
		err = fmt.Errorf("layer: last step copy output gradient values: %w", err)
		return nil, err
	}

	l.backwardInto(rows)
	if err = l.inputGradientScratch.CopyValuesFrom(l.inputGradientValues); err != nil {
		err = fmt.Errorf("layer: last step store input gradient values: %w", err)
		return nil, err
	}

	inputGradient = l.inputGradientScratch
	return inputGradient, nil
}

// InputShape returns the configured sequence input shape.
func (l *LastStep) InputShape() (shape SequenceShape) {
	if l == nil {
		return shape
	}

	shape = l.inputShape
	return shape
}

// OutputSize returns the feature count in the selected final step.
func (l *LastStep) OutputSize() (size int) {
	if l == nil {
		return 0
	}

	size = l.inputShape.FeatureSize()
	return size
}

func (l *LastStep) validate() (err error) {
	if l == nil {
		err = errors.New("layer: last step layer is nil")
		return err
	}

	if err = l.inputShape.validate(); err != nil {
		err = fmt.Errorf("layer: last step input shape invalid: %w", err)
		return err
	}

	return nil
}

func (l *LastStep) validateInput(input *matrix.Matrix) (rows int, err error) {
	var cols int

	if input == nil {
		err = errors.New("layer: last step input is nil")
		return 0, err
	}

	if err = input.Validate(); err != nil {
		err = fmt.Errorf("layer: last step input invalid: %w", err)
		return 0, err
	}

	rows, cols = input.Shape()
	if cols != l.inputShape.Size() {
		err = fmt.Errorf(
			"layer: last step input shape mismatch: got %dx%d, want batch rows x %d",
			rows,
			cols,
			l.inputShape.Size(),
		)
		return 0, err
	}

	return rows, nil
}

func (l *LastStep) validateOutputGradient(outputGradient *matrix.Matrix) (rows int, err error) {
	var cols int

	if outputGradient == nil {
		err = errors.New("layer: last step output gradient is nil")
		return 0, err
	}

	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("layer: last step output gradient invalid: %w", err)
		return 0, err
	}

	rows, cols = outputGradient.Shape()
	if rows != l.forwardRows || cols != l.OutputSize() {
		err = fmt.Errorf(
			"layer: last step output gradient shape mismatch: got %dx%d, want %dx%d",
			rows,
			cols,
			l.forwardRows,
			l.OutputSize(),
		)
		return 0, err
	}

	return rows, nil
}

func (l *LastStep) ensureForwardScratch(rows int, input *matrix.Matrix) (err error) {
	var (
		inputValueCount  int
		outputValueCount int
	)

	inputValueCount = rows * l.inputShape.Size()
	outputValueCount = rows * l.OutputSize()
	if l.outputScratch, _, err = l.outputPool.Get(rows, l.OutputSize()); err != nil {
		err = fmt.Errorf("layer: last step allocate output: %w", err)
		return err
	}

	if l.outputScratch == input {
		if l.outputScratch, err = matrix.New(rows, l.OutputSize()); err != nil {
			err = fmt.Errorf("layer: last step allocate non-aliasing output: %w", err)
			return err
		}
	}

	if l.inputValues, _, err = l.inputValuesPool.Get(inputValueCount); err != nil {
		err = fmt.Errorf("layer: last step allocate input values: %w", err)
		return err
	}

	if l.outputValues, _, err = l.outputValuesPool.Get(outputValueCount); err != nil {
		err = fmt.Errorf("layer: last step allocate output values: %w", err)
		return err
	}

	return nil
}

func (l *LastStep) ensureBackwardScratch(rows int, outputGradient *matrix.Matrix) (err error) {
	var (
		inputValueCount  int
		outputValueCount int
	)

	inputValueCount = rows * l.inputShape.Size()
	outputValueCount = rows * l.OutputSize()
	if l.outputGradientValues, _, err = l.outputGradientValuesPool.Get(outputValueCount); err != nil {
		err = fmt.Errorf("layer: last step allocate output gradient values: %w", err)
		return err
	}

	if l.inputGradientValues, _, err = l.inputGradientValuesPool.Get(inputValueCount); err != nil {
		err = fmt.Errorf("layer: last step allocate input gradient values: %w", err)
		return err
	}

	if l.inputGradientScratch, _, err = l.inputGradientPool.Get(rows, l.inputShape.Size()); err != nil {
		err = fmt.Errorf("layer: last step allocate input gradient: %w", err)
		return err
	}

	if l.inputGradientScratch == outputGradient {
		if l.inputGradientScratch, err = matrix.New(rows, l.inputShape.Size()); err != nil {
			err = fmt.Errorf("layer: last step allocate non-aliasing input gradient: %w", err)
			return err
		}
	}

	return nil
}

func (l *LastStep) forwardInto(rows int) {
	var (
		inputSize   int
		outputSize  int
		finalOffset int
		row         int
		inputStart  int
		outputStart int
	)

	inputSize = l.inputShape.Size()
	outputSize = l.OutputSize()
	finalOffset = inputSize - outputSize
	for row = 0; row < rows; row++ {
		inputStart = row*inputSize + finalOffset
		outputStart = row * outputSize
		copy(l.outputValues[outputStart:outputStart+outputSize], l.inputValues[inputStart:inputStart+outputSize])
	}
}

func (l *LastStep) backwardInto(rows int) {
	var (
		inputSize   int
		outputSize  int
		finalOffset int
		row         int
		inputStart  int
		outputStart int
	)

	clear(l.inputGradientValues)
	inputSize = l.inputShape.Size()
	outputSize = l.OutputSize()
	finalOffset = inputSize - outputSize
	for row = 0; row < rows; row++ {
		inputStart = row*inputSize + finalOffset
		outputStart = row * outputSize
		copy(l.inputGradientValues[inputStart:inputStart+outputSize], l.outputGradientValues[outputStart:outputStart+outputSize])
	}
}
