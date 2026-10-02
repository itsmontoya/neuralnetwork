package layer

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/scratch"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// NewAveragePool2D constructs a parameter-free two-dimensional average-pooling
// layer.
func NewAveragePool2D(config AveragePool2DConfig) (out *AveragePool2D, err error) {
	var a AveragePool2D

	if err = config.validate(); err != nil {
		err = fmt.Errorf("layer: average pool2d configuration invalid: %w", err)
		return nil, err
	}
	a.config = config
	return &a, nil
}

// AveragePool2D applies valid rectangular average pooling independently to
// each channel of flattened channels-first spatial inputs. Only complete
// windows are emitted; uncovered trailing edges do not contribute.
type AveragePool2D struct {
	config                   AveragePool2DConfig
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

// Forward averages each complete pooling window.
func (a *AveragePool2D) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var rows int

	if err = a.validate(); err != nil {
		return nil, err
	}
	if rows, err = a.validateInput(input); err != nil {
		return nil, err
	}
	if err = a.ensureForwardScratch(rows, input); err != nil {
		return nil, err
	}
	if err = input.ValuesInto(a.inputValues); err != nil {
		err = fmt.Errorf("layer: average pool2d copy input values: %w", err)
		return nil, err
	}

	a.forwardInto(rows)
	if err = a.outputScratch.CopyValuesFrom(a.outputValues); err != nil {
		err = fmt.Errorf("layer: average pool2d store output values: %w", err)
		return nil, err
	}
	a.forwardRows = rows
	a.forwardCalled = true
	return a.outputScratch, nil
}

// Backward distributes each output gradient uniformly across its window and
// accumulates contributions from overlapping windows.
func (a *AveragePool2D) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var rows int

	if err = a.validate(); err != nil {
		return nil, err
	}
	if !a.forwardCalled {
		err = errors.New("layer: average pool2d backward called before forward")
		return nil, err
	}
	if rows, err = a.validateOutputGradient(outputGradient); err != nil {
		return nil, err
	}
	if err = a.ensureBackwardScratch(rows, outputGradient); err != nil {
		return nil, err
	}
	if err = outputGradient.ValuesInto(a.outputGradientValues); err != nil {
		err = fmt.Errorf("layer: average pool2d copy output gradient values: %w", err)
		return nil, err
	}

	a.backwardInto(rows)
	if err = a.inputGradientScratch.CopyValuesFrom(a.inputGradientValues); err != nil {
		err = fmt.Errorf("layer: average pool2d store input gradient values: %w", err)
		return nil, err
	}
	return a.inputGradientScratch, nil
}

// Config returns the immutable pooling configuration.
func (a *AveragePool2D) Config() (config AveragePool2DConfig) {
	if a == nil {
		return config
	}
	return a.config
}

// InputShape returns the configured input shape.
func (a *AveragePool2D) InputShape() (shape SpatialShape) {
	if a == nil {
		return shape
	}
	return a.config.InputShape()
}

// OutputShape returns the derived output shape.
func (a *AveragePool2D) OutputShape() (shape SpatialShape) {
	if a == nil {
		return shape
	}
	return a.config.OutputShape()
}

func (a *AveragePool2D) validate() (err error) {
	if a == nil {
		err = errors.New("layer: average pool2d layer is nil")
		return err
	}
	if err = a.config.validate(); err != nil {
		err = fmt.Errorf("layer: average pool2d configuration invalid: %w", err)
		return err
	}
	return nil
}

func (a *AveragePool2D) validateInput(input *matrix.Matrix) (rows int, err error) {
	var cols int

	if input == nil {
		err = errors.New("layer: average pool2d input is nil")
		return 0, err
	}
	if err = input.Validate(); err != nil {
		err = fmt.Errorf("layer: average pool2d input invalid: %w", err)
		return 0, err
	}
	rows, cols = input.Shape()
	if cols != a.config.InputShape().Size() {
		err = fmt.Errorf(
			"layer: average pool2d input shape mismatch: got %dx%d, want batch rows x %d",
			rows,
			cols,
			a.config.InputShape().Size(),
		)
		return 0, err
	}
	return rows, nil
}

func (a *AveragePool2D) validateOutputGradient(outputGradient *matrix.Matrix) (rows int, err error) {
	var cols int

	if outputGradient == nil {
		err = errors.New("layer: average pool2d output gradient is nil")
		return 0, err
	}
	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("layer: average pool2d output gradient invalid: %w", err)
		return 0, err
	}
	rows, cols = outputGradient.Shape()
	if rows != a.forwardRows || cols != a.config.OutputShape().Size() {
		err = fmt.Errorf(
			"layer: average pool2d output gradient shape mismatch: got %dx%d, want %dx%d",
			rows,
			cols,
			a.forwardRows,
			a.config.OutputShape().Size(),
		)
		return 0, err
	}
	return rows, nil
}

func (a *AveragePool2D) ensureForwardScratch(rows int, input *matrix.Matrix) (err error) {
	var (
		inputValues  int
		outputValues int
	)

	inputValues = rows * a.config.InputShape().Size()
	outputValues = rows * a.config.OutputShape().Size()
	if a.outputScratch, _, err = a.outputPool.Get(rows, a.config.OutputShape().Size()); err != nil {
		err = fmt.Errorf("layer: average pool2d allocate output: %w", err)
		return err
	}
	if a.outputScratch == input {
		if a.outputScratch, err = matrix.New(rows, a.config.OutputShape().Size()); err != nil {
			err = fmt.Errorf("layer: average pool2d allocate non-aliasing output: %w", err)
			return err
		}
	}
	if a.inputValues, _, err = a.inputValuesPool.Get(inputValues); err != nil {
		err = fmt.Errorf("layer: average pool2d allocate input values: %w", err)
		return err
	}
	if a.outputValues, _, err = a.outputValuesPool.Get(outputValues); err != nil {
		err = fmt.Errorf("layer: average pool2d allocate output values: %w", err)
		return err
	}
	return nil
}

func (a *AveragePool2D) ensureBackwardScratch(rows int, outputGradient *matrix.Matrix) (err error) {
	var (
		inputValues  int
		outputValues int
	)

	inputValues = rows * a.config.InputShape().Size()
	outputValues = rows * a.config.OutputShape().Size()
	if a.outputGradientValues, _, err = a.outputGradientValuesPool.Get(outputValues); err != nil {
		err = fmt.Errorf("layer: average pool2d allocate output gradient values: %w", err)
		return err
	}
	if a.inputGradientValues, _, err = a.inputGradientValuesPool.Get(inputValues); err != nil {
		err = fmt.Errorf("layer: average pool2d allocate input gradient values: %w", err)
		return err
	}
	if a.inputGradientScratch, _, err = a.inputGradientPool.Get(rows, a.config.InputShape().Size()); err != nil {
		err = fmt.Errorf("layer: average pool2d allocate input gradient: %w", err)
		return err
	}
	if a.inputGradientScratch == outputGradient {
		if a.inputGradientScratch, err = matrix.New(rows, a.config.InputShape().Size()); err != nil {
			err = fmt.Errorf("layer: average pool2d allocate non-aliasing input gradient: %w", err)
			return err
		}
	}
	return nil
}

func (a *AveragePool2D) forwardInto(rows int) {
	var (
		inputShape   SpatialShape
		outputShape  SpatialShape
		inputHeight  int
		inputWidth   int
		outputHeight int
		outputWidth  int
		windowHeight int
		windowWidth  int
		strideHeight int
		strideWidth  int
		inputSize    int
		outputSize   int
		windowSize   float32
		batch        int
		channel      int
		outputRow    int
		outputCol    int
		windowRow    int
		windowCol    int
		inputRow     int
		inputCol     int
		inputIndex   int
		outputIndex  int
		sum          float32
	)

	inputShape = a.config.InputShape()
	outputShape = a.config.OutputShape()
	inputHeight = inputShape.Height()
	inputWidth = inputShape.Width()
	outputHeight = outputShape.Height()
	outputWidth = outputShape.Width()
	windowHeight = a.config.WindowHeight()
	windowWidth = a.config.WindowWidth()
	strideHeight = a.config.StrideHeight()
	strideWidth = a.config.StrideWidth()
	inputSize = inputShape.Size()
	outputSize = outputShape.Size()
	windowSize = float32(windowHeight * windowWidth)

	for batch = 0; batch < rows; batch++ {
		for channel = 0; channel < inputShape.Channels(); channel++ {
			for outputRow = 0; outputRow < outputHeight; outputRow++ {
				for outputCol = 0; outputCol < outputWidth; outputCol++ {
					sum = 0
					for windowRow = 0; windowRow < windowHeight; windowRow++ {
						inputRow = outputRow*strideHeight + windowRow
						for windowCol = 0; windowCol < windowWidth; windowCol++ {
							inputCol = outputCol*strideWidth + windowCol
							inputIndex = batch*inputSize + (channel*inputHeight+inputRow)*inputWidth + inputCol
							sum += a.inputValues[inputIndex]
						}
					}
					outputIndex = batch*outputSize + (channel*outputHeight+outputRow)*outputWidth + outputCol
					a.outputValues[outputIndex] = sum / windowSize
				}
			}
		}
	}
}

func (a *AveragePool2D) backwardInto(rows int) {
	var (
		inputShape   SpatialShape
		outputShape  SpatialShape
		inputHeight  int
		inputWidth   int
		outputHeight int
		outputWidth  int
		windowHeight int
		windowWidth  int
		strideHeight int
		strideWidth  int
		inputSize    int
		outputSize   int
		windowSize   float32
		batch        int
		channel      int
		outputRow    int
		outputCol    int
		windowRow    int
		windowCol    int
		inputRow     int
		inputCol     int
		inputIndex   int
		outputIndex  int
		gradient     float32
	)

	clear(a.inputGradientValues)
	inputShape = a.config.InputShape()
	outputShape = a.config.OutputShape()
	inputHeight = inputShape.Height()
	inputWidth = inputShape.Width()
	outputHeight = outputShape.Height()
	outputWidth = outputShape.Width()
	windowHeight = a.config.WindowHeight()
	windowWidth = a.config.WindowWidth()
	strideHeight = a.config.StrideHeight()
	strideWidth = a.config.StrideWidth()
	inputSize = inputShape.Size()
	outputSize = outputShape.Size()
	windowSize = float32(windowHeight * windowWidth)

	for batch = 0; batch < rows; batch++ {
		for channel = 0; channel < inputShape.Channels(); channel++ {
			for outputRow = 0; outputRow < outputHeight; outputRow++ {
				for outputCol = 0; outputCol < outputWidth; outputCol++ {
					outputIndex = batch*outputSize + (channel*outputHeight+outputRow)*outputWidth + outputCol
					gradient = a.outputGradientValues[outputIndex] / windowSize
					for windowRow = 0; windowRow < windowHeight; windowRow++ {
						inputRow = outputRow*strideHeight + windowRow
						for windowCol = 0; windowCol < windowWidth; windowCol++ {
							inputCol = outputCol*strideWidth + windowCol
							inputIndex = batch*inputSize + (channel*inputHeight+inputRow)*inputWidth + inputCol
							a.inputGradientValues[inputIndex] += gradient
						}
					}
				}
			}
		}
	}
}
