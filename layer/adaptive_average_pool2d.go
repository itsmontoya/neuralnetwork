package layer

import (
	"errors"
	"fmt"
	"math/bits"

	"github.com/itsmontoya/neuralnetwork/internal/scratch"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// NewAdaptiveAveragePool2D constructs a parameter-free adaptive average-pooling
// layer.
func NewAdaptiveAveragePool2D(config AdaptiveAveragePool2DConfig) (out *AdaptiveAveragePool2D, err error) {
	var a AdaptiveAveragePool2D

	if err = config.validate(); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d configuration invalid: %w", err)
		return nil, err
	}
	a.config = config
	return &a, nil
}

// AdaptiveAveragePool2D partitions every input axis into deterministic bins and
// averages each bin independently for every NCHW channel. Bin starts use floor
// and bin ends use ceil, so all input positions are covered and bins may overlap
// when dimensions do not divide evenly or the output is larger than the input.
type AdaptiveAveragePool2D struct {
	config                   AdaptiveAveragePool2DConfig
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

// Forward averages each adaptive spatial bin.
func (a *AdaptiveAveragePool2D) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
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
		err = fmt.Errorf("layer: adaptive average pool2d copy input values: %w", err)
		return nil, err
	}

	a.forwardInto(rows)
	if err = a.outputScratch.CopyValuesFrom(a.outputValues); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d store output values: %w", err)
		return nil, err
	}
	a.forwardRows = rows
	a.forwardCalled = true
	return a.outputScratch, nil
}

// Backward distributes each output gradient uniformly across its adaptive bin
// and accumulates contributions where bins overlap.
func (a *AdaptiveAveragePool2D) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var rows int

	if err = a.validate(); err != nil {
		return nil, err
	}
	if !a.forwardCalled {
		err = errors.New("layer: adaptive average pool2d backward called before forward")
		return nil, err
	}
	if rows, err = a.validateOutputGradient(outputGradient); err != nil {
		return nil, err
	}
	if err = a.ensureBackwardScratch(rows, outputGradient); err != nil {
		return nil, err
	}
	if err = outputGradient.ValuesInto(a.outputGradientValues); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d copy output gradient values: %w", err)
		return nil, err
	}

	a.backwardInto(rows)
	if err = a.inputGradientScratch.CopyValuesFrom(a.inputGradientValues); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d store input gradient values: %w", err)
		return nil, err
	}
	return a.inputGradientScratch, nil
}

// Config returns the immutable pooling configuration.
func (a *AdaptiveAveragePool2D) Config() (config AdaptiveAveragePool2DConfig) {
	if a == nil {
		return config
	}
	return a.config
}

// InputShape returns the configured input shape.
func (a *AdaptiveAveragePool2D) InputShape() (shape SpatialShape) {
	if a == nil {
		return shape
	}
	return a.config.InputShape()
}

// OutputShape returns the configured output shape.
func (a *AdaptiveAveragePool2D) OutputShape() (shape SpatialShape) {
	if a == nil {
		return shape
	}
	return a.config.OutputShape()
}

func (a *AdaptiveAveragePool2D) validate() (err error) {
	if a == nil {
		err = errors.New("layer: adaptive average pool2d layer is nil")
		return err
	}
	if err = a.config.validate(); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d configuration invalid: %w", err)
		return err
	}
	return nil
}

func (a *AdaptiveAveragePool2D) validateInput(input *matrix.Matrix) (rows int, err error) {
	var cols int

	if input == nil {
		err = errors.New("layer: adaptive average pool2d input is nil")
		return 0, err
	}
	if err = input.Validate(); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d input invalid: %w", err)
		return 0, err
	}
	rows, cols = input.Shape()
	if cols != a.config.InputShape().Size() {
		err = fmt.Errorf(
			"layer: adaptive average pool2d input shape mismatch: got %dx%d, want batch rows x %d",
			rows,
			cols,
			a.config.InputShape().Size(),
		)
		return 0, err
	}
	return rows, nil
}

func (a *AdaptiveAveragePool2D) validateOutputGradient(outputGradient *matrix.Matrix) (rows int, err error) {
	var cols int

	if outputGradient == nil {
		err = errors.New("layer: adaptive average pool2d output gradient is nil")
		return 0, err
	}
	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d output gradient invalid: %w", err)
		return 0, err
	}
	rows, cols = outputGradient.Shape()
	if rows != a.forwardRows || cols != a.config.OutputShape().Size() {
		err = fmt.Errorf(
			"layer: adaptive average pool2d output gradient shape mismatch: got %dx%d, want %dx%d",
			rows,
			cols,
			a.forwardRows,
			a.config.OutputShape().Size(),
		)
		return 0, err
	}
	return rows, nil
}

func (a *AdaptiveAveragePool2D) ensureForwardScratch(rows int, input *matrix.Matrix) (err error) {
	var (
		inputValues  int
		outputValues int
	)

	inputValues = rows * a.config.InputShape().Size()
	outputValues = rows * a.config.OutputShape().Size()
	if a.outputScratch, _, err = a.outputPool.Get(rows, a.config.OutputShape().Size()); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d allocate output: %w", err)
		return err
	}
	if a.outputScratch == input {
		if a.outputScratch, err = matrix.New(rows, a.config.OutputShape().Size()); err != nil {
			err = fmt.Errorf("layer: adaptive average pool2d allocate non-aliasing output: %w", err)
			return err
		}
	}
	if a.inputValues, _, err = a.inputValuesPool.Get(inputValues); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d allocate input values: %w", err)
		return err
	}
	if a.outputValues, _, err = a.outputValuesPool.Get(outputValues); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d allocate output values: %w", err)
		return err
	}
	return nil
}

func (a *AdaptiveAveragePool2D) ensureBackwardScratch(rows int, outputGradient *matrix.Matrix) (err error) {
	var (
		inputValues  int
		outputValues int
	)

	inputValues = rows * a.config.InputShape().Size()
	outputValues = rows * a.config.OutputShape().Size()
	if a.outputGradientValues, _, err = a.outputGradientValuesPool.Get(outputValues); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d allocate output gradient values: %w", err)
		return err
	}
	if a.inputGradientValues, _, err = a.inputGradientValuesPool.Get(inputValues); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d allocate input gradient values: %w", err)
		return err
	}
	if a.inputGradientScratch, _, err = a.inputGradientPool.Get(rows, a.config.InputShape().Size()); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d allocate input gradient: %w", err)
		return err
	}
	if a.inputGradientScratch == outputGradient {
		if a.inputGradientScratch, err = matrix.New(rows, a.config.InputShape().Size()); err != nil {
			err = fmt.Errorf("layer: adaptive average pool2d allocate non-aliasing input gradient: %w", err)
			return err
		}
	}
	return nil
}

func (a *AdaptiveAveragePool2D) forwardInto(rows int) {
	var (
		inputShape   SpatialShape
		outputShape  SpatialShape
		inputHeight  int
		inputWidth   int
		outputHeight int
		outputWidth  int
		inputSize    int
		outputSize   int
		batch        int
		channel      int
		outputRow    int
		outputCol    int
		startRow     int
		endRow       int
		startCol     int
		endCol       int
		inputRow     int
		inputCol     int
		inputIndex   int
		outputIndex  int
		binSize      float32
		sum          float32
	)

	inputShape = a.config.InputShape()
	outputShape = a.config.OutputShape()
	inputHeight = inputShape.Height()
	inputWidth = inputShape.Width()
	outputHeight = outputShape.Height()
	outputWidth = outputShape.Width()
	inputSize = inputShape.Size()
	outputSize = outputShape.Size()
	for batch = 0; batch < rows; batch++ {
		for channel = 0; channel < inputShape.Channels(); channel++ {
			for outputRow = 0; outputRow < outputHeight; outputRow++ {
				startRow = adaptivePoolStart(outputRow, inputHeight, outputHeight)
				endRow = adaptivePoolEnd(outputRow, inputHeight, outputHeight)
				for outputCol = 0; outputCol < outputWidth; outputCol++ {
					startCol = adaptivePoolStart(outputCol, inputWidth, outputWidth)
					endCol = adaptivePoolEnd(outputCol, inputWidth, outputWidth)
					sum = 0
					for inputRow = startRow; inputRow < endRow; inputRow++ {
						for inputCol = startCol; inputCol < endCol; inputCol++ {
							inputIndex = batch*inputSize + (channel*inputHeight+inputRow)*inputWidth + inputCol
							sum += a.inputValues[inputIndex]
						}
					}
					binSize = float32((endRow - startRow) * (endCol - startCol))
					outputIndex = batch*outputSize + (channel*outputHeight+outputRow)*outputWidth + outputCol
					a.outputValues[outputIndex] = sum / binSize
				}
			}
		}
	}
}

func (a *AdaptiveAveragePool2D) backwardInto(rows int) {
	var (
		inputShape   SpatialShape
		outputShape  SpatialShape
		inputHeight  int
		inputWidth   int
		outputHeight int
		outputWidth  int
		inputSize    int
		outputSize   int
		batch        int
		channel      int
		outputRow    int
		outputCol    int
		startRow     int
		endRow       int
		startCol     int
		endCol       int
		inputRow     int
		inputCol     int
		inputIndex   int
		outputIndex  int
		binSize      float32
		gradient     float32
	)

	clear(a.inputGradientValues)
	inputShape = a.config.InputShape()
	outputShape = a.config.OutputShape()
	inputHeight = inputShape.Height()
	inputWidth = inputShape.Width()
	outputHeight = outputShape.Height()
	outputWidth = outputShape.Width()
	inputSize = inputShape.Size()
	outputSize = outputShape.Size()
	for batch = 0; batch < rows; batch++ {
		for channel = 0; channel < inputShape.Channels(); channel++ {
			for outputRow = 0; outputRow < outputHeight; outputRow++ {
				startRow = adaptivePoolStart(outputRow, inputHeight, outputHeight)
				endRow = adaptivePoolEnd(outputRow, inputHeight, outputHeight)
				for outputCol = 0; outputCol < outputWidth; outputCol++ {
					startCol = adaptivePoolStart(outputCol, inputWidth, outputWidth)
					endCol = adaptivePoolEnd(outputCol, inputWidth, outputWidth)
					binSize = float32((endRow - startRow) * (endCol - startCol))
					outputIndex = batch*outputSize + (channel*outputHeight+outputRow)*outputWidth + outputCol
					gradient = a.outputGradientValues[outputIndex] / binSize
					for inputRow = startRow; inputRow < endRow; inputRow++ {
						for inputCol = startCol; inputCol < endCol; inputCol++ {
							inputIndex = batch*inputSize + (channel*inputHeight+inputRow)*inputWidth + inputCol
							a.inputGradientValues[inputIndex] += gradient
						}
					}
				}
			}
		}
	}
}

func adaptivePoolStart(index, inputSize, outputSize int) (start int) {
	var (
		high     uint64
		low      uint64
		quotient uint64
	)

	high, low = bits.Mul64(uint64(index), uint64(inputSize))
	quotient, _ = bits.Div64(high, low, uint64(outputSize))
	start = int(quotient)
	return start
}

func adaptivePoolEnd(index, inputSize, outputSize int) (end int) {
	var (
		high      uint64
		low       uint64
		quotient  uint64
		remainder uint64
	)

	high, low = bits.Mul64(uint64(index+1), uint64(inputSize))
	quotient, remainder = bits.Div64(high, low, uint64(outputSize))
	if remainder != 0 {
		quotient++
	}
	end = int(quotient)
	return end
}
