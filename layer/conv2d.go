package layer

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/scratch"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

// NewConv2D constructs a trainable two-dimensional cross-correlation layer.
func NewConv2D(config Conv2DConfig, initializer WeightInitializer) (out *Conv2D, err error) {
	var (
		weights         *matrix.Matrix
		biases          *matrix.Matrix
		weightParameter *optimizer.Parameter
		biasParameter   *optimizer.Parameter
		fanIn           int
		c               Conv2D
	)

	if err = config.validate(); err != nil {
		err = fmt.Errorf("layer: conv2d configuration invalid: %w", err)
		return nil, err
	}

	if initializer == nil {
		err = errors.New("layer: conv2d weight initializer is nil")
		return nil, err
	}

	fanIn = config.InputShape().Channels() * config.KernelHeight() * config.KernelWidth()
	if weights, err = initializer(fanIn, config.OutputChannels()); err != nil {
		err = fmt.Errorf("layer: conv2d initialize weights: %w", err)
		return nil, err
	}

	if err = validateConv2DMatrix("initializer weights", weights, fanIn, config.OutputChannels()); err != nil {
		return nil, err
	}

	if biases, err = matrix.New(1, config.OutputChannels()); err != nil {
		err = fmt.Errorf("layer: conv2d initialize biases: %w", err)
		return nil, err
	}

	if weightParameter, err = optimizer.NewParameter(weights); err != nil {
		err = fmt.Errorf("layer: conv2d construct weights parameter: %w", err)
		return nil, err
	}

	if biasParameter, err = optimizer.NewParameter(biases); err != nil {
		err = fmt.Errorf("layer: conv2d construct biases parameter: %w", err)
		return nil, err
	}

	c.config = config
	c.weights = weightParameter
	c.biases = biasParameter
	c.weightValues = make([]float32, fanIn*config.OutputChannels())
	c.biasValues = make([]float32, config.OutputChannels())
	c.weightGradientValues = make([]float32, fanIn*config.OutputChannels())
	c.biasGradientValues = make([]float32, config.OutputChannels())
	return &c, nil
}

// Conv2D applies a trainable two-dimensional cross-correlation to flattened
// channels-first spatial inputs.
//
// Backward accumulates summed batch and spatial gradients without mean
// scaling. Loss implementations control scaling through their output gradient.
type Conv2D struct {
	config                  Conv2DConfig
	weights                 *optimizer.Parameter
	biases                  *optimizer.Parameter
	inputCachePool          scratch.MatrixPool
	inputCache              *matrix.Matrix
	outputPool              scratch.MatrixPool
	outputScratch           *matrix.Matrix
	inputGradientPool       scratch.MatrixPool
	inputGradientScratch    *matrix.Matrix
	inputValuesPool         scratch.Float32Pool
	inputValues             []float32
	outputValuesPool        scratch.Float32Pool
	outputValues            []float32
	outputGradientPool      scratch.Float32Pool
	outputGradientValues    []float32
	inputGradientValuesPool scratch.Float32Pool
	inputGradientValues     []float32
	weightValues            []float32
	biasValues              []float32
	weightGradientValues    []float32
	biasGradientValues      []float32
	weightGradientScratch   *matrix.Matrix
	biasGradientScratch     *matrix.Matrix
	forwardRows             int
	forwardCalled           bool
}

// Forward applies batched multi-channel cross-correlation and shared biases.
func (c *Conv2D) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var rows int

	if err = c.validate(); err != nil {
		return nil, err
	}

	if rows, err = c.validateInput(input); err != nil {
		return nil, err
	}

	if err = c.ensureForwardScratch(rows, input); err != nil {
		return nil, err
	}

	if err = input.ValuesInto(c.inputValues); err != nil {
		err = fmt.Errorf("layer: conv2d copy input values: %w", err)
		return nil, err
	}

	if err = c.weights.Values().ValuesInto(c.weightValues); err != nil {
		err = fmt.Errorf("layer: conv2d copy weight values: %w", err)
		return nil, err
	}

	if err = c.biases.Values().ValuesInto(c.biasValues); err != nil {
		err = fmt.Errorf("layer: conv2d copy bias values: %w", err)
		return nil, err
	}

	c.forwardInto(rows)
	if err = c.outputScratch.CopyValuesFrom(c.outputValues); err != nil {
		err = fmt.Errorf("layer: conv2d store output values: %w", err)
		return nil, err
	}

	if err = c.inputCache.CopyValuesFrom(c.inputValues); err != nil {
		err = fmt.Errorf("layer: conv2d cache input values: %w", err)
		return nil, err
	}

	c.forwardRows = rows
	c.forwardCalled = true
	output = c.outputScratch
	return output, nil
}

// Backward accumulates parameter gradients and returns the input gradient.
func (c *Conv2D) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var rows int

	if err = c.validate(); err != nil {
		return nil, err
	}

	if !c.forwardCalled {
		err = errors.New("layer: conv2d backward called before forward")
		return nil, err
	}

	if rows, err = c.validateOutputGradient(outputGradient); err != nil {
		return nil, err
	}

	if err = c.ensureBackwardScratch(rows, outputGradient); err != nil {
		return nil, err
	}

	if err = c.inputCache.ValuesInto(c.inputValues); err != nil {
		err = fmt.Errorf("layer: conv2d copy cached input values: %w", err)
		return nil, err
	}

	if err = outputGradient.ValuesInto(c.outputGradientValues); err != nil {
		err = fmt.Errorf("layer: conv2d copy output gradient values: %w", err)
		return nil, err
	}

	if err = c.weights.Values().ValuesInto(c.weightValues); err != nil {
		err = fmt.Errorf("layer: conv2d copy weight values: %w", err)
		return nil, err
	}

	c.backwardInto(rows)
	if err = c.inputGradientScratch.CopyValuesFrom(c.inputGradientValues); err != nil {
		err = fmt.Errorf("layer: conv2d store input gradient values: %w", err)
		return nil, err
	}

	if err = c.weightGradientScratch.CopyValuesFrom(c.weightGradientValues); err != nil {
		err = fmt.Errorf("layer: conv2d store weight gradient values: %w", err)
		return nil, err
	}

	if err = c.biasGradientScratch.CopyValuesFrom(c.biasGradientValues); err != nil {
		err = fmt.Errorf("layer: conv2d store bias gradient values: %w", err)
		return nil, err
	}

	if err = c.weights.AccumulateGradient(c.weightGradientScratch); err != nil {
		err = fmt.Errorf("layer: conv2d accumulate weight gradients: %w", err)
		return nil, err
	}

	if err = c.biases.AccumulateGradient(c.biasGradientScratch); err != nil {
		err = fmt.Errorf("layer: conv2d accumulate bias gradients: %w", err)
		return nil, err
	}

	inputGradient = c.inputGradientScratch
	return inputGradient, nil
}

// Config returns the immutable convolution configuration.
func (c *Conv2D) Config() (config Conv2DConfig) {
	if c == nil {
		return config
	}

	config = c.config
	return config
}

// InputShape returns the configured input shape.
func (c *Conv2D) InputShape() (shape SpatialShape) {
	if c == nil {
		return shape
	}

	shape = c.config.InputShape()
	return shape
}

// OutputShape returns the derived output shape.
func (c *Conv2D) OutputShape() (shape SpatialShape) {
	if c == nil {
		return shape
	}

	shape = c.config.OutputShape()
	return shape
}

// Weights returns the trainable cross-correlation weights.
func (c *Conv2D) Weights() (weights *optimizer.Parameter) {
	if c == nil {
		return nil
	}

	weights = c.weights
	return weights
}

// Biases returns the trainable output-channel biases.
func (c *Conv2D) Biases() (biases *optimizer.Parameter) {
	if c == nil {
		return nil
	}

	biases = c.biases
	return biases
}

// Parameters returns trainable parameters in weight, bias order.
func (c *Conv2D) Parameters() (parameters []*optimizer.Parameter) {
	if c == nil {
		return nil
	}

	parameters = []*optimizer.Parameter{c.weights, c.biases}
	return parameters
}

// AppendParameters appends trainable parameters in weight, bias order.
// The returned slice is caller-owned, and Conv2D does not retain it.
func (c *Conv2D) AppendParameters(parameters []*optimizer.Parameter) (out []*optimizer.Parameter) {
	if c == nil {
		return parameters
	}

	out = append(parameters, c.weights, c.biases)
	return out
}

// ResetGradients clears all accumulated parameter gradients.
func (c *Conv2D) ResetGradients() (err error) {
	if err = c.validate(); err != nil {
		return err
	}

	if err = c.weights.ResetGradient(); err != nil {
		err = fmt.Errorf("layer: conv2d reset weight gradients: %w", err)
		return err
	}

	if err = c.biases.ResetGradient(); err != nil {
		err = fmt.Errorf("layer: conv2d reset bias gradients: %w", err)
		return err
	}

	return nil
}

func (c *Conv2D) validate() (err error) {
	var fanIn int

	if c == nil {
		err = errors.New("layer: conv2d layer is nil")
		return err
	}

	if err = c.config.validate(); err != nil {
		err = fmt.Errorf("layer: conv2d configuration invalid: %w", err)
		return err
	}

	fanIn = c.config.InputShape().Channels() * c.config.KernelHeight() * c.config.KernelWidth()
	if err = validateConv2DParameter(
		"weights",
		"weights gradient",
		c.weights,
		fanIn,
		c.config.OutputChannels(),
	); err != nil {
		return err
	}

	if err = validateConv2DParameter(
		"biases",
		"biases gradient",
		c.biases,
		1,
		c.config.OutputChannels(),
	); err != nil {
		return err
	}

	return nil
}

func (c *Conv2D) validateInput(input *matrix.Matrix) (rows int, err error) {
	var cols int

	if input == nil {
		err = errors.New("layer: conv2d input is nil")
		return 0, err
	}

	if err = input.Validate(); err != nil {
		err = fmt.Errorf("layer: conv2d input invalid: %w", err)
		return 0, err
	}

	rows, cols = input.Shape()
	if cols != c.config.InputShape().Size() {
		err = fmt.Errorf(
			"layer: conv2d input shape mismatch: got %dx%d, want batch rows x %d",
			rows,
			cols,
			c.config.InputShape().Size(),
		)
		return 0, err
	}

	return rows, nil
}

func (c *Conv2D) validateOutputGradient(outputGradient *matrix.Matrix) (rows int, err error) {
	var cols int

	if outputGradient == nil {
		err = errors.New("layer: conv2d output gradient is nil")
		return 0, err
	}

	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("layer: conv2d output gradient invalid: %w", err)
		return 0, err
	}

	rows, cols = outputGradient.Shape()
	if rows != c.forwardRows || cols != c.config.OutputShape().Size() {
		err = fmt.Errorf(
			"layer: conv2d output gradient shape mismatch: got %dx%d, want %dx%d",
			rows,
			cols,
			c.forwardRows,
			c.config.OutputShape().Size(),
		)
		return 0, err
	}

	if c.inputCache == nil {
		err = errors.New("layer: conv2d input cache is nil")
		return 0, err
	}

	if err = validateConv2DMatrix(
		"input cache",
		c.inputCache,
		c.forwardRows,
		c.config.InputShape().Size(),
	); err != nil {
		return 0, err
	}

	return rows, nil
}

func (c *Conv2D) ensureForwardScratch(rows int, input *matrix.Matrix) (err error) {
	var (
		inputSize        int
		outputSize       int
		inputValueCount  int
		outputValueCount int
	)

	inputSize = c.config.InputShape().Size()
	outputSize = c.config.OutputShape().Size()
	inputValueCount = rows * inputSize

	if c.inputCache, _, err = c.inputCachePool.Get(rows, inputSize); err != nil {
		err = fmt.Errorf("layer: conv2d allocate input cache: %w", err)
		return err
	}

	if c.outputScratch, _, err = c.outputPool.Get(rows, outputSize); err != nil {
		err = fmt.Errorf("layer: conv2d allocate output: %w", err)
		return err
	}

	if c.outputScratch == input {
		if c.outputScratch, err = matrix.New(rows, outputSize); err != nil {
			err = fmt.Errorf("layer: conv2d allocate non-aliasing output: %w", err)
			return err
		}
	}

	outputValueCount = rows * outputSize
	if c.inputValues, _, err = c.inputValuesPool.Get(inputValueCount); err != nil {
		err = fmt.Errorf("layer: conv2d allocate input values: %w", err)
		return err
	}

	if c.outputValues, _, err = c.outputValuesPool.Get(outputValueCount); err != nil {
		err = fmt.Errorf("layer: conv2d allocate output values: %w", err)
		return err
	}

	return nil
}

func (c *Conv2D) ensureBackwardScratch(rows int, outputGradient *matrix.Matrix) (err error) {
	var (
		fanIn            int
		inputSize        int
		outputSize       int
		inputValueCount  int
		outputValueCount int
	)

	inputSize = c.config.InputShape().Size()
	outputSize = c.config.OutputShape().Size()
	inputValueCount = rows * inputSize
	outputValueCount = rows * outputSize
	fanIn = c.config.InputShape().Channels() * c.config.KernelHeight() * c.config.KernelWidth()

	if c.outputGradientValues, _, err = c.outputGradientPool.Get(outputValueCount); err != nil {
		err = fmt.Errorf("layer: conv2d allocate output gradient values: %w", err)
		return err
	}

	if c.inputGradientValues, _, err = c.inputGradientValuesPool.Get(inputValueCount); err != nil {
		err = fmt.Errorf("layer: conv2d allocate input gradient values: %w", err)
		return err
	}

	if c.inputGradientScratch, _, err = c.inputGradientPool.Get(rows, inputSize); err != nil {
		err = fmt.Errorf("layer: conv2d allocate input gradient: %w", err)
		return err
	}

	if c.inputGradientScratch == outputGradient {
		if c.inputGradientScratch, err = matrix.New(rows, inputSize); err != nil {
			err = fmt.Errorf("layer: conv2d allocate non-aliasing input gradient: %w", err)
			return err
		}
	}

	if c.weightGradientScratch == nil {
		if c.weightGradientScratch, err = matrix.New(fanIn, c.config.OutputChannels()); err != nil {
			err = fmt.Errorf("layer: conv2d allocate weight gradient: %w", err)
			return err
		}
	}

	if c.biasGradientScratch == nil {
		if c.biasGradientScratch, err = matrix.New(1, c.config.OutputChannels()); err != nil {
			err = fmt.Errorf("layer: conv2d allocate bias gradient: %w", err)
			return err
		}
	}

	return nil
}

func (c *Conv2D) forwardInto(rows int) {
	var (
		inputShape     SpatialShape
		outputShape    SpatialShape
		inputHeight    int
		inputWidth     int
		outputHeight   int
		outputWidth    int
		outputChannels int
		kernelHeight   int
		kernelWidth    int
		strideHeight   int
		strideWidth    int
		paddingHeight  int
		paddingWidth   int
		inputSize      int
		outputSize     int
		batch          int
		outputChannel  int
		outputRow      int
		outputCol      int
		inputChannel   int
		kernelRow      int
		kernelCol      int
		inputRow       int
		inputCol       int
		inputIndex     int
		outputIndex    int
		weightIndex    int
		sum            float32
	)

	inputShape = c.config.InputShape()
	outputShape = c.config.OutputShape()
	inputHeight = inputShape.Height()
	inputWidth = inputShape.Width()
	outputHeight = outputShape.Height()
	outputWidth = outputShape.Width()
	outputChannels = outputShape.Channels()
	kernelHeight = c.config.KernelHeight()
	kernelWidth = c.config.KernelWidth()
	strideHeight = c.config.StrideHeight()
	strideWidth = c.config.StrideWidth()
	paddingHeight = c.config.PaddingHeight()
	paddingWidth = c.config.PaddingWidth()
	inputSize = inputShape.Size()
	outputSize = outputShape.Size()

	for batch = 0; batch < rows; batch++ {
		for outputChannel = 0; outputChannel < outputChannels; outputChannel++ {
			for outputRow = 0; outputRow < outputHeight; outputRow++ {
				for outputCol = 0; outputCol < outputWidth; outputCol++ {
					sum = c.biasValues[outputChannel]
					for inputChannel = 0; inputChannel < inputShape.Channels(); inputChannel++ {
						for kernelRow = 0; kernelRow < kernelHeight; kernelRow++ {
							inputRow = outputRow*strideHeight + kernelRow - paddingHeight
							if inputRow < 0 || inputRow >= inputHeight {
								continue
							}

							for kernelCol = 0; kernelCol < kernelWidth; kernelCol++ {
								inputCol = outputCol*strideWidth + kernelCol - paddingWidth
								if inputCol < 0 || inputCol >= inputWidth {
									continue
								}

								inputIndex = batch*inputSize + (inputChannel*inputHeight+inputRow)*inputWidth + inputCol
								weightIndex = ((inputChannel*kernelHeight+kernelRow)*kernelWidth+kernelCol)*outputChannels + outputChannel
								sum += c.inputValues[inputIndex] * c.weightValues[weightIndex]
							}
						}
					}

					outputIndex = batch*outputSize + (outputChannel*outputHeight+outputRow)*outputWidth + outputCol
					c.outputValues[outputIndex] = sum
				}
			}
		}
	}
}

func (c *Conv2D) backwardInto(rows int) {
	var (
		inputShape     SpatialShape
		outputShape    SpatialShape
		inputHeight    int
		inputWidth     int
		outputHeight   int
		outputWidth    int
		outputChannels int
		kernelHeight   int
		kernelWidth    int
		strideHeight   int
		strideWidth    int
		paddingHeight  int
		paddingWidth   int
		inputSize      int
		outputSize     int
		batch          int
		outputChannel  int
		outputRow      int
		outputCol      int
		inputChannel   int
		kernelRow      int
		kernelCol      int
		inputRow       int
		inputCol       int
		inputIndex     int
		outputIndex    int
		weightIndex    int
		gradient       float32
	)

	clear(c.inputGradientValues)
	clear(c.weightGradientValues)
	clear(c.biasGradientValues)
	inputShape = c.config.InputShape()
	outputShape = c.config.OutputShape()
	inputHeight = inputShape.Height()
	inputWidth = inputShape.Width()
	outputHeight = outputShape.Height()
	outputWidth = outputShape.Width()
	outputChannels = outputShape.Channels()
	kernelHeight = c.config.KernelHeight()
	kernelWidth = c.config.KernelWidth()
	strideHeight = c.config.StrideHeight()
	strideWidth = c.config.StrideWidth()
	paddingHeight = c.config.PaddingHeight()
	paddingWidth = c.config.PaddingWidth()
	inputSize = inputShape.Size()
	outputSize = outputShape.Size()

	for batch = 0; batch < rows; batch++ {
		for outputChannel = 0; outputChannel < outputChannels; outputChannel++ {
			for outputRow = 0; outputRow < outputHeight; outputRow++ {
				for outputCol = 0; outputCol < outputWidth; outputCol++ {
					outputIndex = batch*outputSize + (outputChannel*outputHeight+outputRow)*outputWidth + outputCol
					gradient = c.outputGradientValues[outputIndex]
					c.biasGradientValues[outputChannel] += gradient

					for inputChannel = 0; inputChannel < inputShape.Channels(); inputChannel++ {
						for kernelRow = 0; kernelRow < kernelHeight; kernelRow++ {
							inputRow = outputRow*strideHeight + kernelRow - paddingHeight
							if inputRow < 0 || inputRow >= inputHeight {
								continue
							}

							for kernelCol = 0; kernelCol < kernelWidth; kernelCol++ {
								inputCol = outputCol*strideWidth + kernelCol - paddingWidth
								if inputCol < 0 || inputCol >= inputWidth {
									continue
								}

								inputIndex = batch*inputSize + (inputChannel*inputHeight+inputRow)*inputWidth + inputCol
								weightIndex = ((inputChannel*kernelHeight+kernelRow)*kernelWidth+kernelCol)*outputChannels + outputChannel
								c.inputGradientValues[inputIndex] += gradient * c.weightValues[weightIndex]
								c.weightGradientValues[weightIndex] += c.inputValues[inputIndex] * gradient
							}
						}
					}
				}
			}
		}
	}
}

func validateConv2DParameter(
	name,
	gradientName string,
	parameter *optimizer.Parameter,
	rows,
	cols int,
) (err error) {
	if parameter == nil {
		err = fmt.Errorf("layer: conv2d %s parameter is nil", name)
		return err
	}

	if err = validateConv2DMatrix(name, parameter.Values(), rows, cols); err != nil {
		return err
	}

	if err = validateConv2DMatrix(gradientName, parameter.Gradient(), rows, cols); err != nil {
		return err
	}

	return nil
}

func validateConv2DMatrix(name string, value *matrix.Matrix, rows, cols int) (err error) {
	var (
		valueRows int
		valueCols int
	)

	if value == nil {
		err = fmt.Errorf("layer: conv2d %s is nil", name)
		return err
	}

	if err = value.Validate(); err != nil {
		err = fmt.Errorf("layer: conv2d %s invalid: %w", name, err)
		return err
	}

	valueRows, valueCols = value.Shape()
	if valueRows != rows || valueCols != cols {
		err = fmt.Errorf(
			"layer: conv2d %s shape mismatch: got %dx%d, want %dx%d",
			name,
			valueRows,
			valueCols,
			rows,
			cols,
		)
		return err
	}

	return nil
}
