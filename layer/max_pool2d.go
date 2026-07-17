package layer

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/scratch"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// NewMaxPool2D constructs a parameter-free two-dimensional max-pooling layer.
func NewMaxPool2D(config MaxPool2DConfig) (out *MaxPool2D, err error) {
	var m MaxPool2D
	if err = config.validate(); err != nil {
		err = fmt.Errorf("layer: max pool2d configuration invalid: %w", err)
		return nil, err
	}

	m.config = config
	return &m, nil
}

// MaxPool2D applies valid two-dimensional max pooling independently to each
// channel of flattened channels-first spatial inputs.
type MaxPool2D struct {
	config                   MaxPool2DConfig
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
	argmax                   []int
	forwardRows              int
	forwardCalled            bool
}

// Forward selects the first maximum in each pooling window.
func (m *MaxPool2D) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var rows int

	if err = m.validate(); err != nil {
		return nil, err
	}

	if rows, err = m.validateInput(input); err != nil {
		return nil, err
	}

	if err = m.ensureForwardScratch(rows, input); err != nil {
		return nil, err
	}

	if err = input.ValuesInto(m.inputValues); err != nil {
		err = fmt.Errorf("layer: max pool2d copy input values: %w", err)
		return nil, err
	}

	m.forwardInto(rows)
	if err = m.outputScratch.CopyValuesFrom(m.outputValues); err != nil {
		err = fmt.Errorf("layer: max pool2d store output values: %w", err)
		return nil, err
	}

	m.forwardRows = rows
	m.forwardCalled = true
	output = m.outputScratch
	return output, nil
}

// Backward routes output gradients to the positions selected by Forward.
func (m *MaxPool2D) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var rows int

	if err = m.validate(); err != nil {
		return nil, err
	}

	if !m.forwardCalled {
		err = errors.New("layer: max pool2d backward called before forward")
		return nil, err
	}

	if rows, err = m.validateOutputGradient(outputGradient); err != nil {
		return nil, err
	}

	if err = m.ensureBackwardScratch(rows, outputGradient); err != nil {
		return nil, err
	}

	if err = outputGradient.ValuesInto(m.outputGradientValues); err != nil {
		err = fmt.Errorf("layer: max pool2d copy output gradient values: %w", err)
		return nil, err
	}

	m.backwardInto()
	if err = m.inputGradientScratch.CopyValuesFrom(m.inputGradientValues); err != nil {
		err = fmt.Errorf("layer: max pool2d store input gradient values: %w", err)
		return nil, err
	}

	inputGradient = m.inputGradientScratch
	return inputGradient, nil
}

// Config returns the immutable pooling configuration.
func (m *MaxPool2D) Config() (config MaxPool2DConfig) {
	if m == nil {
		return config
	}

	config = m.config
	return config
}

// InputShape returns the configured input shape.
func (m *MaxPool2D) InputShape() (shape SpatialShape) {
	if m == nil {
		return shape
	}

	shape = m.config.InputShape()
	return shape
}

// OutputShape returns the derived output shape.
func (m *MaxPool2D) OutputShape() (shape SpatialShape) {
	if m == nil {
		return shape
	}

	shape = m.config.OutputShape()
	return shape
}

func (m *MaxPool2D) validate() (err error) {
	if m == nil {
		err = errors.New("layer: max pool2d layer is nil")
		return err
	}

	if err = m.config.validate(); err != nil {
		err = fmt.Errorf("layer: max pool2d configuration invalid: %w", err)
		return err
	}

	return nil
}

func (m *MaxPool2D) validateInput(input *matrix.Matrix) (rows int, err error) {
	var cols int

	if input == nil {
		err = errors.New("layer: max pool2d input is nil")
		return 0, err
	}

	if err = input.Validate(); err != nil {
		err = fmt.Errorf("layer: max pool2d input invalid: %w", err)
		return 0, err
	}

	rows, cols = input.Shape()
	if cols != m.config.InputShape().Size() {
		err = fmt.Errorf(
			"layer: max pool2d input shape mismatch: got %dx%d, want batch rows x %d",
			rows,
			cols,
			m.config.InputShape().Size(),
		)
		return 0, err
	}

	return rows, nil
}

func (m *MaxPool2D) validateOutputGradient(outputGradient *matrix.Matrix) (rows int, err error) {
	var cols int

	if outputGradient == nil {
		err = errors.New("layer: max pool2d output gradient is nil")
		return 0, err
	}

	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("layer: max pool2d output gradient invalid: %w", err)
		return 0, err
	}

	rows, cols = outputGradient.Shape()
	if rows != m.forwardRows || cols != m.config.OutputShape().Size() {
		err = fmt.Errorf(
			"layer: max pool2d output gradient shape mismatch: got %dx%d, want %dx%d",
			rows,
			cols,
			m.forwardRows,
			m.config.OutputShape().Size(),
		)
		return 0, err
	}

	if err = m.validateArgmaxCache(); err != nil {
		return 0, err
	}

	return rows, nil
}

func (m *MaxPool2D) validateArgmaxCache() (err error) {
	var (
		expectedLength int
		inputLength    int
		outputIndex    int
		inputIndex     int
	)

	expectedLength = m.forwardRows * m.config.OutputShape().Size()
	if len(m.argmax) != expectedLength {
		err = fmt.Errorf(
			"layer: max pool2d argmax cache length mismatch: got=%d want=%d",
			len(m.argmax),
			expectedLength,
		)
		return err
	}

	inputLength = m.forwardRows * m.config.InputShape().Size()
	for outputIndex, inputIndex = range m.argmax {
		if inputIndex < 0 || inputIndex >= inputLength {
			err = fmt.Errorf(
				"layer: max pool2d argmax cache position out of range: outputIndex=%d got=%d want=0..%d",
				outputIndex,
				inputIndex,
				inputLength-1,
			)
			return err
		}
	}

	return nil
}

func (m *MaxPool2D) ensureForwardScratch(rows int, input *matrix.Matrix) (err error) {
	var (
		inputValueCount  int
		outputValueCount int
	)

	inputValueCount = rows * m.config.InputShape().Size()
	outputValueCount = rows * m.config.OutputShape().Size()
	if m.outputScratch, _, err = m.outputPool.Get(rows, m.config.OutputShape().Size()); err != nil {
		err = fmt.Errorf("layer: max pool2d allocate output: %w", err)
		return err
	}

	if m.outputScratch == input {
		if m.outputScratch, err = matrix.New(rows, m.config.OutputShape().Size()); err != nil {
			err = fmt.Errorf("layer: max pool2d allocate non-aliasing output: %w", err)
			return err
		}
	}

	if m.inputValues, _, err = m.inputValuesPool.Get(inputValueCount); err != nil {
		err = fmt.Errorf("layer: max pool2d allocate input values: %w", err)
		return err
	}

	if m.outputValues, _, err = m.outputValuesPool.Get(outputValueCount); err != nil {
		err = fmt.Errorf("layer: max pool2d allocate output values: %w", err)
		return err
	}

	if cap(m.argmax) < outputValueCount {
		m.argmax = make([]int, outputValueCount)
	} else {
		m.argmax = m.argmax[:outputValueCount]
	}

	return nil
}

func (m *MaxPool2D) ensureBackwardScratch(rows int, outputGradient *matrix.Matrix) (err error) {
	var (
		inputValueCount  int
		outputValueCount int
	)

	inputValueCount = rows * m.config.InputShape().Size()
	outputValueCount = rows * m.config.OutputShape().Size()
	if m.outputGradientValues, _, err = m.outputGradientValuesPool.Get(outputValueCount); err != nil {
		err = fmt.Errorf("layer: max pool2d allocate output gradient values: %w", err)
		return err
	}

	if m.inputGradientValues, _, err = m.inputGradientValuesPool.Get(inputValueCount); err != nil {
		err = fmt.Errorf("layer: max pool2d allocate input gradient values: %w", err)
		return err
	}

	if m.inputGradientScratch, _, err = m.inputGradientPool.Get(rows, m.config.InputShape().Size()); err != nil {
		err = fmt.Errorf("layer: max pool2d allocate input gradient: %w", err)
		return err
	}

	if m.inputGradientScratch == outputGradient {
		if m.inputGradientScratch, err = matrix.New(rows, m.config.InputShape().Size()); err != nil {
			err = fmt.Errorf("layer: max pool2d allocate non-aliasing input gradient: %w", err)
			return err
		}
	}

	return nil
}

func (m *MaxPool2D) forwardInto(rows int) {
	var (
		inputShape    SpatialShape
		outputShape   SpatialShape
		inputHeight   int
		inputWidth    int
		outputHeight  int
		outputWidth   int
		windowHeight  int
		windowWidth   int
		strideHeight  int
		strideWidth   int
		inputSize     int
		outputSize    int
		batch         int
		channel       int
		outputRow     int
		outputCol     int
		windowRow     int
		windowCol     int
		inputRow      int
		inputCol      int
		inputIndex    int
		outputIndex   int
		selectedIndex int
		value         float32
		maximum       float32
	)

	inputShape = m.config.InputShape()
	outputShape = m.config.OutputShape()
	inputHeight = inputShape.Height()
	inputWidth = inputShape.Width()
	outputHeight = outputShape.Height()
	outputWidth = outputShape.Width()
	windowHeight = m.config.WindowHeight()
	windowWidth = m.config.WindowWidth()
	strideHeight = m.config.StrideHeight()
	strideWidth = m.config.StrideWidth()
	inputSize = inputShape.Size()
	outputSize = outputShape.Size()

	for batch = 0; batch < rows; batch++ {
		for channel = 0; channel < inputShape.Channels(); channel++ {
			for outputRow = 0; outputRow < outputHeight; outputRow++ {
				for outputCol = 0; outputCol < outputWidth; outputCol++ {
					selectedIndex = -1
					for windowRow = 0; windowRow < windowHeight; windowRow++ {
						inputRow = outputRow*strideHeight + windowRow
						for windowCol = 0; windowCol < windowWidth; windowCol++ {
							inputCol = outputCol*strideWidth + windowCol
							inputIndex = batch*inputSize + (channel*inputHeight+inputRow)*inputWidth + inputCol
							value = m.inputValues[inputIndex]
							if selectedIndex < 0 || value > maximum {
								selectedIndex = inputIndex
								maximum = value
							}
						}
					}

					outputIndex = batch*outputSize + (channel*outputHeight+outputRow)*outputWidth + outputCol
					m.outputValues[outputIndex] = maximum
					m.argmax[outputIndex] = selectedIndex
				}
			}
		}
	}
}

func (m *MaxPool2D) backwardInto() {
	var (
		outputIndex int
		inputIndex  int
	)

	clear(m.inputGradientValues)
	for outputIndex, inputIndex = range m.argmax {
		m.inputGradientValues[inputIndex] += m.outputGradientValues[outputIndex]
	}
}
