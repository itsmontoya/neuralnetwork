package layer

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/internal/scratch"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

// NewSimpleRNN constructs a stateless tanh recurrent layer with zero biases.
func NewSimpleRNN(config SimpleRNNConfig, inputInitializer WeightInitializer, recurrentInitializer WeightInitializer) (out *SimpleRNN, err error) {
	var (
		inputWeightMatrix        *matrix.Matrix
		recurrentWeightMatrix    *matrix.Matrix
		biasMatrix               *matrix.Matrix
		inputWeightParameter     *optimizer.Parameter
		recurrentWeightParameter *optimizer.Parameter
		biasParameter            *optimizer.Parameter
		inputFeatureSize         int
		hiddenSize               int
		r                        SimpleRNN
	)

	if err = config.validate(); err != nil {
		err = fmt.Errorf("layer: simple rnn configuration invalid: %w", err)
		return nil, err
	}

	if inputInitializer == nil {
		err = errors.New("layer: simple rnn input weight initializer is nil")
		return nil, err
	}

	if recurrentInitializer == nil {
		err = errors.New("layer: simple rnn recurrent weight initializer is nil")
		return nil, err
	}

	inputFeatureSize = config.InputShape().FeatureSize()
	hiddenSize = config.HiddenSize()
	if inputWeightMatrix, err = inputInitializer(inputFeatureSize, hiddenSize); err != nil {
		err = fmt.Errorf("layer: simple rnn initialize input weights: %w", err)
		return nil, err
	}

	if err = validateSimpleRNNMatrix(
		"initializer input weights",
		inputWeightMatrix,
		inputFeatureSize,
		hiddenSize,
	); err != nil {
		return nil, err
	}

	if recurrentWeightMatrix, err = recurrentInitializer(hiddenSize, hiddenSize); err != nil {
		err = fmt.Errorf("layer: simple rnn initialize recurrent weights: %w", err)
		return nil, err
	}

	if err = validateSimpleRNNMatrix(
		"initializer recurrent weights",
		recurrentWeightMatrix,
		hiddenSize,
		hiddenSize,
	); err != nil {
		return nil, err
	}

	if biasMatrix, err = matrix.New(1, hiddenSize); err != nil {
		err = fmt.Errorf("layer: simple rnn initialize biases: %w", err)
		return nil, err
	}

	if inputWeightParameter, err = optimizer.NewParameter(inputWeightMatrix); err != nil {
		err = fmt.Errorf("layer: simple rnn construct input weights parameter: %w", err)
		return nil, err
	}

	if recurrentWeightParameter, err = optimizer.NewParameter(recurrentWeightMatrix); err != nil {
		err = fmt.Errorf("layer: simple rnn construct recurrent weights parameter: %w", err)
		return nil, err
	}

	if biasParameter, err = optimizer.NewParameter(biasMatrix); err != nil {
		err = fmt.Errorf("layer: simple rnn construct biases parameter: %w", err)
		return nil, err
	}

	r.config = config
	r.inputWeights = inputWeightParameter
	r.recurrentWeights = recurrentWeightParameter
	r.biases = biasParameter
	r.inputWeightValues = make([]float32, inputFeatureSize*hiddenSize)
	r.recurrentWeightValues = make([]float32, hiddenSize*hiddenSize)
	r.biasValues = make([]float32, hiddenSize)
	r.inputWeightGradientValues = make([]float32, inputFeatureSize*hiddenSize)
	r.recurrentWeightGradientValues = make([]float32, hiddenSize*hiddenSize)
	r.biasGradientValues = make([]float32, hiddenSize)
	r.hiddenGradientValues = make([]float32, hiddenSize)
	r.previousHiddenGradientValues = make([]float32, hiddenSize)
	r.stepGradientValues = make([]float32, hiddenSize)
	return &r, nil
}

// SimpleRNN applies a stateless Elman recurrence and returns every hidden step.
//
// Backward performs full backpropagation through time and accumulates summed
// batch and step gradients without mean scaling or clipping.
type SimpleRNN struct {
	config                         SimpleRNNConfig
	inputWeights                   *optimizer.Parameter
	recurrentWeights               *optimizer.Parameter
	biases                         *optimizer.Parameter
	inputCachePool                 scratch.MatrixPool
	inputCache                     *matrix.Matrix
	hiddenCachePool                scratch.MatrixPool
	hiddenCache                    *matrix.Matrix
	outputPool                     scratch.MatrixPool
	outputScratch                  *matrix.Matrix
	inputGradientPool              scratch.MatrixPool
	inputGradientScratch           *matrix.Matrix
	inputValuesPool                scratch.Float32Pool
	inputValues                    []float32
	hiddenValuesPool               scratch.Float32Pool
	hiddenValues                   []float32
	outputGradientValuesPool       scratch.Float32Pool
	outputGradientValues           []float32
	inputGradientValuesPool        scratch.Float32Pool
	inputGradientValues            []float32
	inputWeightValues              []float32
	recurrentWeightValues          []float32
	biasValues                     []float32
	inputWeightGradientValues      []float32
	recurrentWeightGradientValues  []float32
	biasGradientValues             []float32
	hiddenGradientValues           []float32
	previousHiddenGradientValues   []float32
	stepGradientValues             []float32
	inputWeightGradientScratch     *matrix.Matrix
	recurrentWeightGradientScratch *matrix.Matrix
	biasGradientScratch            *matrix.Matrix
	forwardRows                    int
	forwardCalled                  bool
}

// Forward applies the tanh recurrence from a zero hidden state for every row.
func (r *SimpleRNN) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var rows int

	if err = r.validate(); err != nil {
		return nil, err
	}

	if rows, err = r.validateInput(input); err != nil {
		return nil, err
	}

	if err = r.ensureForwardScratch(rows, input); err != nil {
		return nil, err
	}

	if err = input.ValuesInto(r.inputValues); err != nil {
		err = fmt.Errorf("layer: simple rnn copy input values: %w", err)
		return nil, err
	}

	if err = r.inputWeights.Values().ValuesInto(r.inputWeightValues); err != nil {
		err = fmt.Errorf("layer: simple rnn copy input weight values: %w", err)
		return nil, err
	}

	if err = r.recurrentWeights.Values().ValuesInto(r.recurrentWeightValues); err != nil {
		err = fmt.Errorf("layer: simple rnn copy recurrent weight values: %w", err)
		return nil, err
	}

	if err = r.biases.Values().ValuesInto(r.biasValues); err != nil {
		err = fmt.Errorf("layer: simple rnn copy bias values: %w", err)
		return nil, err
	}

	r.forwardInto(rows)
	if err = r.outputScratch.CopyValuesFrom(r.hiddenValues); err != nil {
		err = fmt.Errorf("layer: simple rnn store output values: %w", err)
		return nil, err
	}

	if err = r.inputCache.CopyValuesFrom(r.inputValues); err != nil {
		err = fmt.Errorf("layer: simple rnn cache input values: %w", err)
		return nil, err
	}

	if err = r.hiddenCache.CopyValuesFrom(r.hiddenValues); err != nil {
		err = fmt.Errorf("layer: simple rnn cache hidden values: %w", err)
		return nil, err
	}

	r.forwardRows = rows
	r.forwardCalled = true
	output = r.outputScratch
	return output, nil
}

// Backward performs full reverse-time propagation and accumulates gradients.
func (r *SimpleRNN) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var rows int

	if err = r.validate(); err != nil {
		return nil, err
	}

	if !r.forwardCalled {
		err = errors.New("layer: simple rnn backward called before forward")
		return nil, err
	}

	if rows, err = r.validateOutputGradient(outputGradient); err != nil {
		return nil, err
	}

	if err = r.ensureBackwardScratch(rows, outputGradient); err != nil {
		return nil, err
	}

	if err = r.inputCache.ValuesInto(r.inputValues); err != nil {
		err = fmt.Errorf("layer: simple rnn copy cached input values: %w", err)
		return nil, err
	}

	if err = r.hiddenCache.ValuesInto(r.hiddenValues); err != nil {
		err = fmt.Errorf("layer: simple rnn copy cached hidden values: %w", err)
		return nil, err
	}

	if err = outputGradient.ValuesInto(r.outputGradientValues); err != nil {
		err = fmt.Errorf("layer: simple rnn copy output gradient values: %w", err)
		return nil, err
	}

	if err = r.inputWeights.Values().ValuesInto(r.inputWeightValues); err != nil {
		err = fmt.Errorf("layer: simple rnn copy input weight values: %w", err)
		return nil, err
	}

	if err = r.recurrentWeights.Values().ValuesInto(r.recurrentWeightValues); err != nil {
		err = fmt.Errorf("layer: simple rnn copy recurrent weight values: %w", err)
		return nil, err
	}

	r.backwardInto(rows)
	if err = r.inputGradientScratch.CopyValuesFrom(r.inputGradientValues); err != nil {
		err = fmt.Errorf("layer: simple rnn store input gradient values: %w", err)
		return nil, err
	}

	if err = r.inputWeightGradientScratch.CopyValuesFrom(r.inputWeightGradientValues); err != nil {
		err = fmt.Errorf("layer: simple rnn store input weight gradient values: %w", err)
		return nil, err
	}

	if err = r.recurrentWeightGradientScratch.CopyValuesFrom(r.recurrentWeightGradientValues); err != nil {
		err = fmt.Errorf("layer: simple rnn store recurrent weight gradient values: %w", err)
		return nil, err
	}

	if err = r.biasGradientScratch.CopyValuesFrom(r.biasGradientValues); err != nil {
		err = fmt.Errorf("layer: simple rnn store bias gradient values: %w", err)
		return nil, err
	}

	if err = r.inputWeights.AccumulateGradient(r.inputWeightGradientScratch); err != nil {
		err = fmt.Errorf("layer: simple rnn accumulate input weight gradients: %w", err)
		return nil, err
	}

	if err = r.recurrentWeights.AccumulateGradient(r.recurrentWeightGradientScratch); err != nil {
		err = fmt.Errorf("layer: simple rnn accumulate recurrent weight gradients: %w", err)
		return nil, err
	}

	if err = r.biases.AccumulateGradient(r.biasGradientScratch); err != nil {
		err = fmt.Errorf("layer: simple rnn accumulate bias gradients: %w", err)
		return nil, err
	}

	inputGradient = r.inputGradientScratch
	return inputGradient, nil
}

// Config returns the immutable recurrent configuration.
func (r *SimpleRNN) Config() (config SimpleRNNConfig) {
	if r == nil {
		return config
	}

	config = r.config
	return config
}

// InputShape returns the configured input sequence shape.
func (r *SimpleRNN) InputShape() (shape SequenceShape) {
	if r == nil {
		return shape
	}

	shape = r.config.InputShape()
	return shape
}

// OutputShape returns the derived hidden sequence shape.
func (r *SimpleRNN) OutputShape() (shape SequenceShape) {
	if r == nil {
		return shape
	}

	shape = r.config.OutputShape()
	return shape
}

// InputWeights returns the trainable input weight parameter.
func (r *SimpleRNN) InputWeights() (weights *optimizer.Parameter) {
	if r == nil {
		return nil
	}

	weights = r.inputWeights
	return weights
}

// RecurrentWeights returns the trainable recurrent weight parameter.
func (r *SimpleRNN) RecurrentWeights() (weights *optimizer.Parameter) {
	if r == nil {
		return nil
	}

	weights = r.recurrentWeights
	return weights
}

// Biases returns the trainable hidden bias parameter.
func (r *SimpleRNN) Biases() (biases *optimizer.Parameter) {
	if r == nil {
		return nil
	}

	biases = r.biases
	return biases
}

// Parameters returns parameters in input weight, recurrent weight, bias order.
func (r *SimpleRNN) Parameters() (parameters []*optimizer.Parameter) {
	if r == nil {
		return nil
	}

	parameters = []*optimizer.Parameter{r.inputWeights, r.recurrentWeights, r.biases}
	return parameters
}

// AppendParameters appends parameters in input weight, recurrent weight, bias order.
// The returned slice is caller-owned, and SimpleRNN does not retain it.
func (r *SimpleRNN) AppendParameters(parameters []*optimizer.Parameter) (out []*optimizer.Parameter) {
	if r == nil {
		return parameters
	}

	out = append(parameters, r.inputWeights, r.recurrentWeights, r.biases)
	return out
}

// ResetGradients clears all accumulated parameter gradients.
func (r *SimpleRNN) ResetGradients() (err error) {
	if err = r.validate(); err != nil {
		return err
	}

	if err = r.inputWeights.ResetGradient(); err != nil {
		err = fmt.Errorf("layer: simple rnn reset input weight gradients: %w", err)
		return err
	}

	if err = r.recurrentWeights.ResetGradient(); err != nil {
		err = fmt.Errorf("layer: simple rnn reset recurrent weight gradients: %w", err)
		return err
	}

	if err = r.biases.ResetGradient(); err != nil {
		err = fmt.Errorf("layer: simple rnn reset bias gradients: %w", err)
		return err
	}

	return nil
}

func (r *SimpleRNN) validate() (err error) {
	var (
		inputFeatureSize int
		hiddenSize       int
	)

	if r == nil {
		err = errors.New("layer: simple rnn layer is nil")
		return err
	}

	if err = r.config.validate(); err != nil {
		err = fmt.Errorf("layer: simple rnn configuration invalid: %w", err)
		return err
	}

	inputFeatureSize = r.config.InputShape().FeatureSize()
	hiddenSize = r.config.HiddenSize()
	if err = validateSimpleRNNParameter(
		"input weights",
		"input weight gradient",
		r.inputWeights,
		inputFeatureSize,
		hiddenSize,
	); err != nil {
		return err
	}

	if err = validateSimpleRNNParameter(
		"recurrent weights",
		"recurrent weight gradient",
		r.recurrentWeights,
		hiddenSize,
		hiddenSize,
	); err != nil {
		return err
	}

	if err = validateSimpleRNNParameter(
		"biases",
		"bias gradient",
		r.biases,
		1,
		hiddenSize,
	); err != nil {
		return err
	}

	return nil
}

func (r *SimpleRNN) validateInput(input *matrix.Matrix) (rows int, err error) {
	var cols int

	if input == nil {
		err = errors.New("layer: simple rnn input is nil")
		return 0, err
	}

	if err = input.Validate(); err != nil {
		err = fmt.Errorf("layer: simple rnn input invalid: %w", err)
		return 0, err
	}

	rows, cols = input.Shape()
	if cols != r.config.InputShape().Size() {
		err = fmt.Errorf(
			"layer: simple rnn input shape mismatch: got %dx%d, want batch rows x %d",
			rows,
			cols,
			r.config.InputShape().Size(),
		)
		return 0, err
	}

	return rows, nil
}

func (r *SimpleRNN) validateOutputGradient(outputGradient *matrix.Matrix) (rows int, err error) {
	var cols int

	if outputGradient == nil {
		err = errors.New("layer: simple rnn output gradient is nil")
		return 0, err
	}

	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("layer: simple rnn output gradient invalid: %w", err)
		return 0, err
	}

	rows, cols = outputGradient.Shape()
	if rows != r.forwardRows || cols != r.config.OutputShape().Size() {
		err = fmt.Errorf(
			"layer: simple rnn output gradient shape mismatch: got %dx%d, want %dx%d",
			rows,
			cols,
			r.forwardRows,
			r.config.OutputShape().Size(),
		)
		return 0, err
	}

	if err = validateSimpleRNNMatrix(
		"input cache",
		r.inputCache,
		r.forwardRows,
		r.config.InputShape().Size(),
	); err != nil {
		return 0, err
	}

	if err = validateSimpleRNNMatrix(
		"hidden cache",
		r.hiddenCache,
		r.forwardRows,
		r.config.OutputShape().Size(),
	); err != nil {
		return 0, err
	}

	return rows, nil
}

func (r *SimpleRNN) ensureForwardScratch(rows int, input *matrix.Matrix) (err error) {
	var (
		inputSize        int
		outputSize       int
		inputValueCount  int
		outputValueCount int
	)

	inputSize = r.config.InputShape().Size()
	outputSize = r.config.OutputShape().Size()
	inputValueCount = rows * inputSize
	if r.inputCache, _, err = r.inputCachePool.Get(rows, inputSize); err != nil {
		err = fmt.Errorf("layer: simple rnn allocate input cache: %w", err)
		return err
	}

	if r.hiddenCache, _, err = r.hiddenCachePool.Get(rows, outputSize); err != nil {
		err = fmt.Errorf("layer: simple rnn allocate hidden cache: %w", err)
		return err
	}

	if r.outputScratch, _, err = r.outputPool.Get(rows, outputSize); err != nil {
		err = fmt.Errorf("layer: simple rnn allocate output: %w", err)
		return err
	}

	if r.outputScratch == input {
		if r.outputScratch, err = matrix.New(rows, outputSize); err != nil {
			err = fmt.Errorf("layer: simple rnn allocate non-aliasing output: %w", err)
			return err
		}
	}

	outputValueCount = rows * outputSize
	if r.inputValues, _, err = r.inputValuesPool.Get(inputValueCount); err != nil {
		err = fmt.Errorf("layer: simple rnn allocate input values: %w", err)
		return err
	}

	if r.hiddenValues, _, err = r.hiddenValuesPool.Get(outputValueCount); err != nil {
		err = fmt.Errorf("layer: simple rnn allocate hidden values: %w", err)
		return err
	}

	return nil
}

func (r *SimpleRNN) ensureBackwardScratch(rows int, outputGradient *matrix.Matrix) (err error) {
	var (
		inputFeatureSize int
		hiddenSize       int
		inputSize        int
		outputSize       int
		inputValueCount  int
		outputValueCount int
	)

	inputFeatureSize = r.config.InputShape().FeatureSize()
	hiddenSize = r.config.HiddenSize()
	inputSize = r.config.InputShape().Size()
	outputSize = r.config.OutputShape().Size()
	inputValueCount = rows * inputSize
	outputValueCount = rows * outputSize
	if r.outputGradientValues, _, err = r.outputGradientValuesPool.Get(outputValueCount); err != nil {
		err = fmt.Errorf("layer: simple rnn allocate output gradient values: %w", err)
		return err
	}

	if r.inputGradientValues, _, err = r.inputGradientValuesPool.Get(inputValueCount); err != nil {
		err = fmt.Errorf("layer: simple rnn allocate input gradient values: %w", err)
		return err
	}

	if r.inputGradientScratch, _, err = r.inputGradientPool.Get(rows, inputSize); err != nil {
		err = fmt.Errorf("layer: simple rnn allocate input gradient: %w", err)
		return err
	}

	if r.inputGradientScratch == outputGradient {
		if r.inputGradientScratch, err = matrix.New(rows, inputSize); err != nil {
			err = fmt.Errorf("layer: simple rnn allocate non-aliasing input gradient: %w", err)
			return err
		}
	}

	if r.inputWeightGradientScratch == nil {
		if r.inputWeightGradientScratch, err = matrix.New(inputFeatureSize, hiddenSize); err != nil {
			err = fmt.Errorf("layer: simple rnn allocate input weight gradient: %w", err)
			return err
		}
	}

	if r.recurrentWeightGradientScratch == nil {
		if r.recurrentWeightGradientScratch, err = matrix.New(hiddenSize, hiddenSize); err != nil {
			err = fmt.Errorf("layer: simple rnn allocate recurrent weight gradient: %w", err)
			return err
		}
	}

	if r.biasGradientScratch == nil {
		if r.biasGradientScratch, err = matrix.New(1, hiddenSize); err != nil {
			err = fmt.Errorf("layer: simple rnn allocate bias gradient: %w", err)
			return err
		}
	}

	return nil
}

func (r *SimpleRNN) forwardInto(rows int) {
	var (
		steps                int
		inputFeatureSize     int
		hiddenSize           int
		inputSize            int
		outputSize           int
		batch                int
		step                 int
		feature              int
		hidden               int
		previousHidden       int
		inputIndex           int
		hiddenIndex          int
		previousHiddenIndex  int
		inputWeightIndex     int
		recurrentWeightIndex int
		sum                  float32
	)

	steps = r.config.InputShape().Steps()
	inputFeatureSize = r.config.InputShape().FeatureSize()
	hiddenSize = r.config.HiddenSize()
	inputSize = r.config.InputShape().Size()
	outputSize = r.config.OutputShape().Size()
	for batch = 0; batch < rows; batch++ {
		for step = 0; step < steps; step++ {
			for hidden = 0; hidden < hiddenSize; hidden++ {
				sum = r.biasValues[hidden]
				for feature = 0; feature < inputFeatureSize; feature++ {
					inputIndex = batch*inputSize + step*inputFeatureSize + feature
					inputWeightIndex = feature*hiddenSize + hidden
					sum += r.inputValues[inputIndex] * r.inputWeightValues[inputWeightIndex]
				}

				if step > 0 {
					for previousHidden = 0; previousHidden < hiddenSize; previousHidden++ {
						previousHiddenIndex = batch*outputSize + (step-1)*hiddenSize + previousHidden
						recurrentWeightIndex = previousHidden*hiddenSize + hidden
						sum += r.hiddenValues[previousHiddenIndex] * r.recurrentWeightValues[recurrentWeightIndex]
					}
				}

				hiddenIndex = batch*outputSize + step*hiddenSize + hidden
				r.hiddenValues[hiddenIndex] = f32.Tanh(sum)
			}
		}
	}
}

func (r *SimpleRNN) backwardInto(rows int) {
	var (
		steps                   int
		inputFeatureSize        int
		hiddenSize              int
		inputSize               int
		outputSize              int
		batch                   int
		step                    int
		feature                 int
		hidden                  int
		previousHidden          int
		inputIndex              int
		hiddenIndex             int
		previousHiddenIndex     int
		inputWeightIndex        int
		recurrentWeightIndex    int
		hiddenValue             float32
		gradient                float32
		inputValue              float32
		previousHiddenValue     float32
		temporaryGradientValues []float32
	)

	clear(r.inputGradientValues)
	clear(r.inputWeightGradientValues)
	clear(r.recurrentWeightGradientValues)
	clear(r.biasGradientValues)
	steps = r.config.InputShape().Steps()
	inputFeatureSize = r.config.InputShape().FeatureSize()
	hiddenSize = r.config.HiddenSize()
	inputSize = r.config.InputShape().Size()
	outputSize = r.config.OutputShape().Size()
	for batch = 0; batch < rows; batch++ {
		clear(r.hiddenGradientValues)
		clear(r.previousHiddenGradientValues)
		for step = steps - 1; step >= 0; step-- {
			for hidden = 0; hidden < hiddenSize; hidden++ {
				hiddenIndex = batch*outputSize + step*hiddenSize + hidden
				hiddenValue = r.hiddenValues[hiddenIndex]
				gradient = r.outputGradientValues[hiddenIndex] + r.hiddenGradientValues[hidden]
				gradient *= 1 - hiddenValue*hiddenValue
				r.stepGradientValues[hidden] = gradient
				r.biasGradientValues[hidden] += gradient
			}

			for feature = 0; feature < inputFeatureSize; feature++ {
				inputIndex = batch*inputSize + step*inputFeatureSize + feature
				inputValue = r.inputValues[inputIndex]
				for hidden = 0; hidden < hiddenSize; hidden++ {
					inputWeightIndex = feature*hiddenSize + hidden
					gradient = r.stepGradientValues[hidden]
					r.inputGradientValues[inputIndex] += gradient * r.inputWeightValues[inputWeightIndex]
					r.inputWeightGradientValues[inputWeightIndex] += inputValue * gradient
				}
			}

			clear(r.previousHiddenGradientValues)
			if step > 0 {
				for previousHidden = 0; previousHidden < hiddenSize; previousHidden++ {
					previousHiddenIndex = batch*outputSize + (step-1)*hiddenSize + previousHidden
					previousHiddenValue = r.hiddenValues[previousHiddenIndex]
					for hidden = 0; hidden < hiddenSize; hidden++ {
						recurrentWeightIndex = previousHidden*hiddenSize + hidden
						gradient = r.stepGradientValues[hidden]
						r.recurrentWeightGradientValues[recurrentWeightIndex] += previousHiddenValue * gradient
						r.previousHiddenGradientValues[previousHidden] += gradient * r.recurrentWeightValues[recurrentWeightIndex]
					}
				}
			}

			temporaryGradientValues = r.hiddenGradientValues
			r.hiddenGradientValues = r.previousHiddenGradientValues
			r.previousHiddenGradientValues = temporaryGradientValues
		}
	}
}

func validateSimpleRNNParameter(
	name,
	gradientName string,
	parameter *optimizer.Parameter,
	rows,
	cols int,
) (err error) {
	if parameter == nil {
		err = fmt.Errorf("layer: simple rnn %s parameter is nil", name)
		return err
	}

	if err = validateSimpleRNNMatrix(name, parameter.Values(), rows, cols); err != nil {
		return err
	}

	if err = validateSimpleRNNMatrix(gradientName, parameter.Gradient(), rows, cols); err != nil {
		return err
	}

	return nil
}

func validateSimpleRNNMatrix(name string, value *matrix.Matrix, rows, cols int) (err error) {
	var (
		valueRows int
		valueCols int
	)

	if value == nil {
		err = fmt.Errorf("layer: simple rnn %s is nil", name)
		return err
	}

	if err = value.Validate(); err != nil {
		err = fmt.Errorf("layer: simple rnn %s invalid: %w", name, err)
		return err
	}

	valueRows, valueCols = value.Shape()
	if valueRows != rows || valueCols != cols {
		err = fmt.Errorf(
			"layer: simple rnn %s shape mismatch: got %dx%d, want %dx%d",
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
