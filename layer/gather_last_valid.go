package layer

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/scratch"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// NewGatherLastValid constructs a length-aware sequence-to-dense adapter.
func NewGatherLastValid(inputShape SequenceShape) (out *GatherLastValid, err error) {
	if err = inputShape.validate(); err != nil {
		err = fmt.Errorf(
			"layer: gather last valid input shape invalid: steps=%d featureSize=%d: %w",
			inputShape.Steps(),
			inputShape.FeatureSize(),
			err,
		)
		return nil, err
	}

	var g GatherLastValid
	g.inputShape = inputShape
	return &g, nil
}

// GatherLastValid selects each row's final valid sequence feature vector.
type GatherLastValid struct {
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
	forwardLengths           []int
	forwardCalled            bool
}

// Forward rejects calls that do not explicitly provide sequence lengths.
func (g *GatherLastValid) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	if g != nil {
		g.invalidateForward()
	}

	if err = g.validate(); err != nil {
		return nil, err
	}

	err = errors.New("layer: gather last valid forward requires ForwardWithLengths")
	return nil, err
}

// Backward rejects calls outside the explicit length-aware path.
func (g *GatherLastValid) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	if err = g.validate(); err != nil {
		return nil, err
	}

	err = errors.New("layer: gather last valid backward requires BackwardWithLengths")
	return nil, err
}

// ForwardWithLengths copies each row's final valid step.
func (g *GatherLastValid) ForwardWithLengths(
	input *matrix.Matrix,
	lengths []int,
) (output *matrix.Matrix, err error) {
	var rows int

	if g != nil {
		g.invalidateForward()
	}

	if err = g.validate(); err != nil {
		return nil, err
	}

	if rows, err = g.validateInput(input); err != nil {
		return nil, err
	}

	if err = g.validateLengths(lengths, rows); err != nil {
		return nil, err
	}

	if err = g.ensureForwardScratch(rows, input); err != nil {
		return nil, err
	}

	if err = input.ValuesInto(g.inputValues); err != nil {
		err = fmt.Errorf("layer: gather last valid copy input values: %w", err)
		return nil, err
	}

	g.snapshotLengths(lengths)
	g.forwardInto(rows)
	if err = g.outputScratch.CopyValuesFrom(g.outputValues); err != nil {
		err = fmt.Errorf("layer: gather last valid store output values: %w", err)
		return nil, err
	}

	g.forwardCalled = true
	output = g.outputScratch
	return output, nil
}

// BackwardWithLengths routes gradients only to each row's selected step.
func (g *GatherLastValid) BackwardWithLengths(
	outputGradient *matrix.Matrix,
) (inputGradient *matrix.Matrix, err error) {
	var rows int

	if err = g.validate(); err != nil {
		return nil, err
	}

	if !g.forwardCalled {
		err = errors.New("layer: gather last valid backward called before length-aware forward")
		return nil, err
	}

	if rows, err = g.validateOutputGradient(outputGradient); err != nil {
		return nil, err
	}

	if err = g.ensureBackwardScratch(rows, outputGradient); err != nil {
		return nil, err
	}

	if err = outputGradient.ValuesInto(g.outputGradientValues); err != nil {
		err = fmt.Errorf("layer: gather last valid copy output gradient values: %w", err)
		return nil, err
	}

	g.backwardInto(rows)
	if err = g.inputGradientScratch.CopyValuesFrom(g.inputGradientValues); err != nil {
		err = fmt.Errorf("layer: gather last valid store input gradient values: %w", err)
		return nil, err
	}

	inputGradient = g.inputGradientScratch
	return inputGradient, nil
}

// InputShape returns the configured sequence input shape.
func (g *GatherLastValid) InputShape() (shape SequenceShape) {
	if g == nil {
		return shape
	}

	shape = g.inputShape
	return shape
}

// OutputSize returns the feature count in each selected step.
func (g *GatherLastValid) OutputSize() (size int) {
	if g == nil {
		return 0
	}

	size = g.inputShape.FeatureSize()
	return size
}

func (g *GatherLastValid) validate() (err error) {
	if g == nil {
		err = errors.New("layer: gather last valid layer is nil")
		return err
	}

	if err = g.inputShape.validate(); err != nil {
		err = fmt.Errorf("layer: gather last valid input shape invalid: %w", err)
		return err
	}

	return nil
}

func (g *GatherLastValid) validateInput(input *matrix.Matrix) (rows int, err error) {
	var cols int

	if input == nil {
		err = errors.New("layer: gather last valid input is nil")
		return 0, err
	}

	if err = input.Validate(); err != nil {
		err = fmt.Errorf("layer: gather last valid input invalid: %w", err)
		return 0, err
	}

	rows, cols = input.Shape()
	if cols != g.inputShape.Size() {
		err = fmt.Errorf(
			"layer: gather last valid input shape mismatch: got %dx%d, want batch rows x %d",
			rows,
			cols,
			g.inputShape.Size(),
		)
		return 0, err
	}

	return rows, nil
}

func (g *GatherLastValid) validateLengths(lengths []int, rows int) (err error) {
	var (
		row    int
		length int
	)

	if len(lengths) != rows {
		err = fmt.Errorf(
			"layer: gather last valid length count mismatch: got=%d want=%d",
			len(lengths),
			rows,
		)
		return err
	}

	for row, length = range lengths {
		if length < 1 || length > g.inputShape.Steps() {
			err = fmt.Errorf(
				"layer: gather last valid length out of range: row=%d value=%d want=1..%d",
				row,
				length,
				g.inputShape.Steps(),
			)
			return err
		}
	}

	return nil
}

func (g *GatherLastValid) validateOutputGradient(outputGradient *matrix.Matrix) (rows int, err error) {
	var cols int

	if outputGradient == nil {
		err = errors.New("layer: gather last valid output gradient is nil")
		return 0, err
	}

	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("layer: gather last valid output gradient invalid: %w", err)
		return 0, err
	}

	rows, cols = outputGradient.Shape()
	if rows != len(g.forwardLengths) || cols != g.OutputSize() {
		err = fmt.Errorf(
			"layer: gather last valid output gradient shape mismatch: got %dx%d, want %dx%d",
			rows,
			cols,
			len(g.forwardLengths),
			g.OutputSize(),
		)
		return 0, err
	}

	return rows, nil
}

func (g *GatherLastValid) ensureForwardScratch(rows int, input *matrix.Matrix) (err error) {
	var (
		inputValueCount  int
		outputValueCount int
	)

	inputValueCount = rows * g.inputShape.Size()
	outputValueCount = rows * g.OutputSize()
	if g.outputScratch, _, err = g.outputPool.Get(rows, g.OutputSize()); err != nil {
		err = fmt.Errorf("layer: gather last valid allocate output: %w", err)
		return err
	}

	if g.outputScratch == input {
		if g.outputScratch, err = matrix.New(rows, g.OutputSize()); err != nil {
			err = fmt.Errorf("layer: gather last valid allocate non-aliasing output: %w", err)
			return err
		}
	}

	if g.inputValues, _, err = g.inputValuesPool.Get(inputValueCount); err != nil {
		err = fmt.Errorf("layer: gather last valid allocate input values: %w", err)
		return err
	}

	if g.outputValues, _, err = g.outputValuesPool.Get(outputValueCount); err != nil {
		err = fmt.Errorf("layer: gather last valid allocate output values: %w", err)
		return err
	}

	return nil
}

func (g *GatherLastValid) ensureBackwardScratch(rows int, outputGradient *matrix.Matrix) (err error) {
	var (
		inputValueCount  int
		outputValueCount int
	)

	inputValueCount = rows * g.inputShape.Size()
	outputValueCount = rows * g.OutputSize()
	if g.outputGradientValues, _, err = g.outputGradientValuesPool.Get(outputValueCount); err != nil {
		err = fmt.Errorf("layer: gather last valid allocate output gradient values: %w", err)
		return err
	}

	if g.inputGradientValues, _, err = g.inputGradientValuesPool.Get(inputValueCount); err != nil {
		err = fmt.Errorf("layer: gather last valid allocate input gradient values: %w", err)
		return err
	}

	if g.inputGradientScratch, _, err = g.inputGradientPool.Get(rows, g.inputShape.Size()); err != nil {
		err = fmt.Errorf("layer: gather last valid allocate input gradient: %w", err)
		return err
	}

	if g.inputGradientScratch == outputGradient {
		if g.inputGradientScratch, err = matrix.New(rows, g.inputShape.Size()); err != nil {
			err = fmt.Errorf("layer: gather last valid allocate non-aliasing input gradient: %w", err)
			return err
		}
	}

	return nil
}

func (g *GatherLastValid) snapshotLengths(lengths []int) {
	if cap(g.forwardLengths) < len(lengths) {
		g.forwardLengths = make([]int, len(lengths))
	} else {
		g.forwardLengths = g.forwardLengths[:len(lengths)]
	}

	copy(g.forwardLengths, lengths)
}

func (g *GatherLastValid) invalidateForward() {
	g.forwardCalled = false
	g.forwardLengths = g.forwardLengths[:0]
}

func (g *GatherLastValid) forwardInto(rows int) {
	var (
		inputSize   int
		outputSize  int
		row         int
		inputStart  int
		outputStart int
	)

	inputSize = g.inputShape.Size()
	outputSize = g.OutputSize()
	for row = 0; row < rows; row++ {
		inputStart = row*inputSize + (g.forwardLengths[row]-1)*outputSize
		outputStart = row * outputSize
		copy(g.outputValues[outputStart:outputStart+outputSize], g.inputValues[inputStart:inputStart+outputSize])
	}
}

func (g *GatherLastValid) backwardInto(rows int) {
	var (
		inputSize   int
		outputSize  int
		row         int
		inputStart  int
		outputStart int
	)

	clear(g.inputGradientValues)
	inputSize = g.inputShape.Size()
	outputSize = g.OutputSize()
	for row = 0; row < rows; row++ {
		inputStart = row*inputSize + (g.forwardLengths[row]-1)*outputSize
		outputStart = row * outputSize
		copy(g.inputGradientValues[inputStart:inputStart+outputSize], g.outputGradientValues[outputStart:outputStart+outputSize])
	}
}
