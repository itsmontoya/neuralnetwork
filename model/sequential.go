// Package model provides neural network composition and training orchestration.
package model

import (
	"errors"
	"fmt"
	"io"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/internal/scratch"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

// NewSequential constructs a Sequential model with training mode enabled.
func NewSequential(layers ...layer.Layer) (out *Sequential, err error) {
	var current layer.Layer
	var s Sequential

	s.training = true

	for _, current = range layers {
		if err = s.Add(current); err != nil {
			return nil, err
		}
	}

	return &s, nil
}

// LoadSequential restores a Sequential model from the v1 JSON contract.
//
// The document must use format "neuralnetwork.sequential", version 1, and
// a supported dense, activation, dropout, batch-normalization, convolution,
// max-pooling, flatten, simple recurrent, last-step, or gather-last-valid layer
// type. Loading restores architecture, parameter values, and
// batch-normalization running statistics only. Optimizer state, accumulated
// gradients, forward caches, recurrent hidden histories, gathered length
// snapshots, training history, callbacks, learning-rate schedules, and
// original random source state are not serialized; dropout layers use
// deterministic local random sources, and recurrent and length-aware layers
// begin with fresh forward state. Existing version 1 documents remain
// compatible; older readers reject documents containing unknown additive
// layer types.
func LoadSequential(reader io.Reader) (out *Sequential, err error) {
	if reader == nil {
		err = errors.New("model: load reader is nil")
		return nil, err
	}

	if out, err = decodeSequential(reader); err != nil {
		err = fmt.Errorf("model: load sequential: %w", err)
		return nil, err
	}

	return out, nil
}

// Sequential applies an ordered list of layers.
type Sequential struct {
	layers                    []layer.Layer
	parameterBuffer           []*optimizer.Parameter
	gradientPool              scratch.MatrixPool
	execution                 *device.Execution
	lengthValues              []int
	lengthAwareForward        bool
	lengthAwareForwardRows    int
	lengthAwareForwardColumns int
	training                  bool
}

type sequenceLengthLayer interface {
	ForwardWithLengths(
		input *matrix.Matrix,
		lengths []int,
	) (output *matrix.Matrix, err error)
	BackwardWithLengths(
		outputGradient *matrix.Matrix,
	) (inputGradient *matrix.Matrix, err error)
	InputShape() layer.SequenceShape
}

// Add appends a layer to the model.
func (s *Sequential) Add(next layer.Layer) (err error) {
	var modeLayer trainingModeLayer
	var ok bool

	s.invalidateLengthAwareForward()
	if err = s.validate(); err != nil {
		return err
	}

	if next == nil {
		err = errors.New("model: layer is nil")
		return err
	}

	s.layers = append(s.layers, next)
	modeLayer, ok = next.(trainingModeLayer)
	if ok {
		modeLayer.SetTraining(s.training)
	}

	return nil
}

// Predict runs a forward pass through every layer.
func (s *Sequential) Predict(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var (
		execution *device.Execution
		owned     bool
	)

	s.invalidateLengthAwareForward()
	if err = s.validateOrdinaryGraph(
		"prediction",
		"PredictWithLengths (which invokes ForwardWithLengths)",
	); err != nil {
		return nil, err
	}

	if execution, owned, err = s.beginExecution(input); err != nil {
		return nil, fmt.Errorf("model: begin prediction execution: %w", err)
	}
	defer func() {
		var (
			panicValue any
			cleanupErr error
		)

		panicValue = recover()
		if panicValue != nil {
			if owned {
				execution.Abort(errors.New("model: prediction panicked"))
			}
			panic(panicValue)
		}
		if !owned {
			return
		}
		if err != nil {
			if cleanupErr = execution.Abort(err); cleanupErr != nil {
				err = errors.Join(err, fmt.Errorf("model: abort prediction execution: %w", cleanupErr))
			}
			output = nil
			return
		}
		if cleanupErr = execution.Finish(); cleanupErr != nil {
			err = fmt.Errorf("model: finish prediction execution: %w", cleanupErr)
			output = nil
		}
	}()

	output, err = s.predict(input, execution)
	return output, err
}

// PredictWithLengths runs a forward pass using one logical length per input row.
func (s *Sequential) PredictWithLengths(
	input *matrix.Matrix,
	lengths *data.SequenceLengths,
) (output *matrix.Matrix, err error) {
	var (
		selector      sequenceLengthLayer
		selectorIndex int
		lengthValues  []int
	)

	s.invalidateLengthAwareForward()
	if selector, selectorIndex, lengthValues, err = s.prepareLengthAwareInput(
		input,
		lengths,
		"prediction",
	); err != nil {
		return nil, err
	}

	output, err = s.runLengthAwarePrediction(
		input,
		lengthValues,
		selector,
		selectorIndex,
	)
	return output, err
}

func (s *Sequential) predict(
	input *matrix.Matrix,
	execution *device.Execution,
) (output *matrix.Matrix, err error) {
	var (
		index   int
		current layer.Layer
	)

	if err = s.validateReady(); err != nil {
		return nil, err
	}

	if input == nil {
		err = errors.New("model: input is nil")
		return nil, err
	}

	if err = input.Validate(); err != nil {
		err = fmt.Errorf("model: input matrix invalid: %w", err)
		return nil, err
	}
	if execution != nil {
		if err = execution.Bind(input); err != nil {
			return nil, fmt.Errorf("model: bind prediction input: %w", err)
		}
	}

	output = input
	for index, current = range s.layers {
		if output, err = current.Forward(output); err != nil {
			err = fmt.Errorf("model: layer %d forward failed: %w", index, err)
			return nil, err
		}
		if execution != nil {
			if err = execution.Bind(output); err != nil {
				return nil, fmt.Errorf("model: bind layer %d output: %w", index, err)
			}
		}
	}

	return output, nil
}

func (s *Sequential) runLengthAwarePrediction(
	input *matrix.Matrix,
	lengths []int,
	selector sequenceLengthLayer,
	selectorIndex int,
) (output *matrix.Matrix, err error) {
	var (
		execution *device.Execution
		owned     bool
	)

	if execution, owned, err = s.beginExecution(input); err != nil {
		return nil, fmt.Errorf("model: begin length-aware prediction execution: %w", err)
	}
	defer func() {
		var (
			panicValue any
			cleanupErr error
		)

		panicValue = recover()
		if panicValue != nil {
			s.invalidateLengthAwareForward()
			if owned {
				execution.Abort(errors.New("model: length-aware prediction panicked"))
			}
			panic(panicValue)
		}
		if owned && err != nil {
			if cleanupErr = execution.Abort(err); cleanupErr != nil {
				err = errors.Join(
					err,
					fmt.Errorf("model: abort length-aware prediction execution: %w", cleanupErr),
				)
			}
			output = nil
		} else if owned {
			if cleanupErr = execution.Finish(); cleanupErr != nil {
				err = fmt.Errorf(
					"model: finish length-aware prediction execution: %w",
					cleanupErr,
				)
				output = nil
			}
		}

		if err != nil {
			s.invalidateLengthAwareForward()
			return
		}

		s.recordLengthAwareForward(output.Rows(), output.Cols())
	}()

	output, err = s.predictWithLengths(
		input,
		lengths,
		selector,
		selectorIndex,
		execution,
	)
	return output, err
}

func (s *Sequential) predictWithLengths(
	input *matrix.Matrix,
	lengths []int,
	selector sequenceLengthLayer,
	selectorIndex int,
	execution *device.Execution,
) (output *matrix.Matrix, err error) {
	var (
		index   int
		current layer.Layer
	)

	if execution != nil {
		if err = execution.Bind(input); err != nil {
			return nil, fmt.Errorf("model: bind length-aware prediction input: %w", err)
		}
	}

	output = input
	for index, current = range s.layers {
		if index == selectorIndex {
			if output, err = selector.ForwardWithLengths(output, lengths); err != nil {
				err = fmt.Errorf("model: layer %d length-aware forward failed: %w", index, err)
				return nil, err
			}
		} else if output, err = current.Forward(output); err != nil {
			err = fmt.Errorf("model: layer %d forward failed: %w", index, err)
			return nil, err
		}

		if execution != nil {
			if err = execution.Bind(output); err != nil {
				return nil, fmt.Errorf("model: bind layer %d output: %w", index, err)
			}
		}
	}

	return output, nil
}

// Backward runs a backward pass through every layer in reverse order.
func (s *Sequential) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var (
		execution *device.Execution
		owned     bool
	)

	s.invalidateLengthAwareForward()
	if err = s.validateOrdinaryGraph("backward", "BackwardWithLengths"); err != nil {
		return nil, err
	}

	if execution, owned, err = s.beginExecution(outputGradient); err != nil {
		return nil, fmt.Errorf("model: begin backward execution: %w", err)
	}
	defer func() {
		var (
			panicValue any
			cleanupErr error
		)

		panicValue = recover()
		if panicValue != nil {
			if owned {
				execution.Abort(errors.New("model: backward panicked"))
			}
			panic(panicValue)
		}
		if !owned {
			return
		}
		if err != nil {
			if cleanupErr = execution.Abort(err); cleanupErr != nil {
				err = errors.Join(err, fmt.Errorf("model: abort backward execution: %w", cleanupErr))
			}
			inputGradient = nil
			return
		}
		if cleanupErr = execution.Finish(); cleanupErr != nil {
			err = fmt.Errorf("model: finish backward execution: %w", cleanupErr)
			inputGradient = nil
		}
	}()

	inputGradient, err = s.backward(outputGradient, execution)
	return inputGradient, err
}

// BackwardWithLengths runs backward through a matching length-aware prediction.
func (s *Sequential) BackwardWithLengths(
	outputGradient *matrix.Matrix,
) (inputGradient *matrix.Matrix, err error) {
	var (
		selector      sequenceLengthLayer
		selectorIndex int
		execution     *device.Execution
		owned         bool
	)

	if err = s.validateLengthAwareBackward(outputGradient); err != nil {
		s.invalidateLengthAwareForward()
		return nil, err
	}
	s.invalidateLengthAwareForward()

	if selector, selectorIndex, err = s.lengthAwareGraph("backward"); err != nil {
		return nil, err
	}

	if execution, owned, err = s.beginExecution(outputGradient); err != nil {
		return nil, fmt.Errorf("model: begin length-aware backward execution: %w", err)
	}
	defer func() {
		var (
			panicValue any
			cleanupErr error
		)

		panicValue = recover()
		if panicValue != nil {
			s.invalidateLengthAwareForward()
			if owned {
				execution.Abort(errors.New("model: length-aware backward panicked"))
			}
			panic(panicValue)
		}
		if owned && err != nil {
			if cleanupErr = execution.Abort(err); cleanupErr != nil {
				err = errors.Join(
					err,
					fmt.Errorf("model: abort length-aware backward execution: %w", cleanupErr),
				)
			}
			inputGradient = nil
		} else if owned {
			if cleanupErr = execution.Finish(); cleanupErr != nil {
				err = fmt.Errorf(
					"model: finish length-aware backward execution: %w",
					cleanupErr,
				)
				inputGradient = nil
			}
		}

		if err != nil {
			s.invalidateLengthAwareForward()
			return
		}

		s.recordLengthAwareForward(inputGradient.Rows(), outputGradient.Cols())
	}()

	inputGradient, err = s.backwardWithLengths(
		outputGradient,
		selector,
		selectorIndex,
		execution,
	)
	return inputGradient, err
}

func (s *Sequential) backward(
	outputGradient *matrix.Matrix,
	execution *device.Execution,
) (inputGradient *matrix.Matrix, err error) {
	var index int

	if err = s.validateReady(); err != nil {
		return nil, err
	}

	if outputGradient == nil {
		err = errors.New("model: output gradient is nil")
		return nil, err
	}

	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("model: output gradient matrix invalid: %w", err)
		return nil, err
	}
	if execution != nil {
		if err = execution.Bind(outputGradient); err != nil {
			return nil, fmt.Errorf("model: bind output gradient: %w", err)
		}
	}

	inputGradient = outputGradient
	for index = len(s.layers) - 1; index >= 0; index-- {
		if inputGradient, err = s.layers[index].Backward(inputGradient); err != nil {
			err = fmt.Errorf("model: layer %d backward failed: %w", index, err)
			return nil, err
		}
		if execution != nil {
			if err = execution.Bind(inputGradient); err != nil {
				return nil, fmt.Errorf("model: bind layer %d input gradient: %w", index, err)
			}
		}
	}

	return inputGradient, nil
}

func (s *Sequential) backwardWithLengths(
	outputGradient *matrix.Matrix,
	selector sequenceLengthLayer,
	selectorIndex int,
	execution *device.Execution,
) (inputGradient *matrix.Matrix, err error) {
	var index int

	if execution != nil {
		if err = execution.Bind(outputGradient); err != nil {
			return nil, fmt.Errorf("model: bind length-aware output gradient: %w", err)
		}
	}

	inputGradient = outputGradient
	for index = len(s.layers) - 1; index >= 0; index-- {
		if index == selectorIndex {
			if inputGradient, err = selector.BackwardWithLengths(inputGradient); err != nil {
				err = fmt.Errorf("model: layer %d length-aware backward failed: %w", index, err)
				return nil, err
			}
		} else if inputGradient, err = s.layers[index].Backward(inputGradient); err != nil {
			err = fmt.Errorf("model: layer %d backward failed: %w", index, err)
			return nil, err
		}

		if execution != nil {
			if err = execution.Bind(inputGradient); err != nil {
				return nil, fmt.Errorf("model: bind layer %d input gradient: %w", index, err)
			}
		}
	}

	return inputGradient, nil
}

// Parameters returns a caller-owned slice of trainable parameters in layer order.
// Mutating the returned slice does not change the model's parameter enumeration.
func (s *Sequential) Parameters() (parameters []*optimizer.Parameter) {
	var internalParameters []*optimizer.Parameter

	internalParameters = s.rebuildParameters()
	if len(internalParameters) == 0 {
		return nil
	}

	parameters = make([]*optimizer.Parameter, len(internalParameters))
	copy(parameters, internalParameters)
	return parameters
}

func (s *Sequential) rebuildParameters() (parameters []*optimizer.Parameter) {
	var (
		current         layer.Layer
		appendLayer     parameterAppender
		parameterLayer  parameterProvider
		layerParameters []*optimizer.Parameter
		ok              bool
	)

	if s == nil {
		return nil
	}

	clear(s.parameterBuffer)
	s.parameterBuffer = s.parameterBuffer[:0]
	for _, current = range s.layers {
		appendLayer, ok = current.(parameterAppender)
		if ok {
			s.parameterBuffer = appendLayer.AppendParameters(s.parameterBuffer)
			continue
		}

		parameterLayer, ok = current.(parameterProvider)
		if !ok {
			continue
		}

		layerParameters = parameterLayer.Parameters()
		s.parameterBuffer = append(s.parameterBuffer, layerParameters...)
	}

	parameters = s.parameterBuffer
	return parameters
}

// SetTraining updates the model training flag and propagates it to compatible layers.
func (s *Sequential) SetTraining(training bool) (err error) {
	var (
		current   layer.Layer
		modeLayer trainingModeLayer
		ok        bool
	)

	if err = s.validate(); err != nil {
		return err
	}

	s.training = training
	for _, current = range s.layers {
		modeLayer, ok = current.(trainingModeLayer)
		if !ok {
			continue
		}

		modeLayer.SetTraining(training)
	}

	return nil
}

// Training reports whether the model is in training mode.
func (s *Sequential) Training() (training bool) {
	if s == nil {
		return false
	}

	training = s.training
	return training
}

// TrainBatch runs one supervised training step and updates trainable parameters.
func (s *Sequential) TrainBatch(
	input,
	targets *matrix.Matrix,
	lossFunc loss.Loss,
	optimizerRule optimizer.Optimizer,
) (metrics TrainMetrics, err error) {
	var (
		previousTraining bool
		predictions      *matrix.Matrix
		gradient         *matrix.Matrix
		execution        *device.Execution
		owned            bool
	)

	s.invalidateLengthAwareForward()
	if lossFunc == nil {
		err = errors.New("model: loss is nil")
		return metrics, err
	}

	if optimizerRule == nil {
		err = errors.New("model: optimizer is nil")
		return metrics, err
	}

	if err = s.validateOrdinaryGraph("training", "TrainBatchWithLengths"); err != nil {
		return metrics, err
	}

	previousTraining = s.Training()
	if err = s.SetTraining(true); err != nil {
		return metrics, err
	}
	defer func() {
		var restoreErr error

		if restoreErr = s.SetTraining(previousTraining); restoreErr != nil && err == nil {
			err = restoreErr
		}
	}()
	if execution, owned, err = s.beginExecution(input); err != nil {
		return metrics, fmt.Errorf("model: begin training execution: %w", err)
	}
	defer func() {
		var (
			panicValue any
			cleanupErr error
		)

		panicValue = recover()
		if panicValue != nil {
			if owned {
				execution.Abort(errors.New("model: training panicked"))
			}
			panic(panicValue)
		}
		if !owned {
			return
		}
		if err != nil {
			if cleanupErr = execution.Abort(err); cleanupErr != nil {
				err = errors.Join(err, fmt.Errorf("model: abort training execution: %w", cleanupErr))
			}
			return
		}
		if cleanupErr = execution.Finish(); cleanupErr != nil {
			err = fmt.Errorf("model: finish training execution: %w", cleanupErr)
		}
	}()

	if predictions, err = s.predict(input, execution); err != nil {
		return metrics, err
	}
	if execution != nil && targets != nil {
		if err = execution.Bind(targets); err != nil {
			return metrics, fmt.Errorf("model: bind training targets: %w", err)
		}
	}

	if metrics.Loss, err = lossFunc.Value(predictions, targets); err != nil {
		err = fmt.Errorf("model: loss value failed: %w", err)
		return metrics, err
	}

	if gradient, err = s.lossGradient(lossFunc, predictions, targets); err != nil {
		err = fmt.Errorf("model: loss gradient failed: %w", err)
		return metrics, err
	}

	if _, err = s.backward(gradient, execution); err != nil {
		err = fmt.Errorf("model: backward failed: %w", err)
		return metrics, err
	}
	if execution != nil && !supportsResidentUpdate(optimizerRule) {
		if err = execution.Barrier(device.BoundaryCPUFallback); err != nil {
			err = fmt.Errorf("model: complete backward execution before optimizer update: %w", err)
			return metrics, err
		}
	}

	if err = optimizerRule.Update(s.rebuildParameters()); err != nil {
		err = fmt.Errorf("model: optimizer update failed: %w", err)
		return metrics, err
	}

	return metrics, nil
}

// TrainBatchWithLengths runs one supervised step with aligned logical lengths.
func (s *Sequential) TrainBatchWithLengths(
	input,
	targets *matrix.Matrix,
	lengths *data.SequenceLengths,
	lossFunc loss.Loss,
	optimizerRule optimizer.Optimizer,
) (metrics TrainMetrics, err error) {
	var (
		selector      sequenceLengthLayer
		selectorIndex int
		lengthValues  []int
	)

	s.invalidateLengthAwareForward()
	defer s.invalidateLengthAwareForward()
	if lossFunc == nil {
		err = errors.New("model: length-aware training loss is nil")
		return metrics, err
	}

	if optimizerRule == nil {
		err = errors.New("model: length-aware training optimizer is nil")
		return metrics, err
	}

	if selector, selectorIndex, lengthValues, err = s.prepareLengthAwareInput(
		input,
		lengths,
		"training",
	); err != nil {
		return metrics, err
	}

	if err = validateLengthAwareTargets(input, targets); err != nil {
		return metrics, err
	}

	metrics, err = s.trainPreparedWithLengths(
		input,
		targets,
		lengthValues,
		selector,
		selectorIndex,
		lossFunc,
		optimizerRule,
	)
	return metrics, err
}

func (s *Sequential) trainPreparedWithLengths(
	input,
	targets *matrix.Matrix,
	lengths []int,
	selector sequenceLengthLayer,
	selectorIndex int,
	lossFunc loss.Loss,
	optimizerRule optimizer.Optimizer,
) (metrics TrainMetrics, err error) {
	var (
		previousTraining bool
		predictions      *matrix.Matrix
		gradient         *matrix.Matrix
		execution        *device.Execution
		owned            bool
	)

	previousTraining = s.Training()
	if err = s.SetTraining(true); err != nil {
		return metrics, err
	}
	defer func() {
		var restoreErr error

		if restoreErr = s.SetTraining(previousTraining); restoreErr != nil && err == nil {
			err = restoreErr
		}
	}()

	if execution, owned, err = s.beginExecution(input); err != nil {
		return metrics, fmt.Errorf("model: begin length-aware training execution: %w", err)
	}
	defer func() {
		var (
			panicValue any
			cleanupErr error
		)

		panicValue = recover()
		if panicValue != nil {
			if owned {
				execution.Abort(errors.New("model: length-aware training panicked"))
			}
			panic(panicValue)
		}
		if !owned {
			return
		}
		if err != nil {
			if cleanupErr = execution.Abort(err); cleanupErr != nil {
				err = errors.Join(
					err,
					fmt.Errorf("model: abort length-aware training execution: %w", cleanupErr),
				)
			}
			return
		}
		if cleanupErr = execution.Finish(); cleanupErr != nil {
			err = fmt.Errorf("model: finish length-aware training execution: %w", cleanupErr)
		}
	}()

	if predictions, err = s.predictWithLengths(
		input,
		lengths,
		selector,
		selectorIndex,
		execution,
	); err != nil {
		return metrics, err
	}

	if execution != nil {
		if err = execution.Bind(targets); err != nil {
			return metrics, fmt.Errorf("model: bind length-aware training targets: %w", err)
		}
	}

	if metrics.Loss, err = lossFunc.Value(predictions, targets); err != nil {
		err = fmt.Errorf("model: length-aware loss value failed: %w", err)
		return metrics, err
	}

	if gradient, err = s.lossGradient(lossFunc, predictions, targets); err != nil {
		err = fmt.Errorf("model: length-aware loss gradient failed: %w", err)
		return metrics, err
	}

	if _, err = s.backwardWithLengths(
		gradient,
		selector,
		selectorIndex,
		execution,
	); err != nil {
		err = fmt.Errorf("model: length-aware backward failed: %w", err)
		return metrics, err
	}

	if execution != nil && !supportsResidentUpdate(optimizerRule) {
		if err = execution.Barrier(device.BoundaryCPUFallback); err != nil {
			err = fmt.Errorf(
				"model: complete length-aware backward execution before optimizer update: %w",
				err,
			)
			return metrics, err
		}
	}

	if err = optimizerRule.Update(s.rebuildParameters()); err != nil {
		err = fmt.Errorf("model: length-aware optimizer update failed: %w", err)
		return metrics, err
	}

	return metrics, nil
}

func supportsResidentUpdate(optimizerRule optimizer.Optimizer) (supported bool) {
	_, supported = optimizerRule.(*optimizer.SGD)
	return supported
}

func (s *Sequential) beginExecution(
	value *matrix.Matrix,
) (execution *device.Execution, owned bool, err error) {
	var (
		runtimeValue *device.Runtime
		available    bool
	)

	if execution, err = device.BoundExecution(value); err != nil {
		return nil, false, err
	}
	if execution != nil {
		return execution, false, nil
	}
	if err = s.validate(); err != nil {
		return nil, false, err
	}
	if !deviceExecutionEligible(s.layers, value) {
		return nil, false, nil
	}
	if runtimeValue, available, err = device.SharedRuntime(); err != nil {
		return nil, false, err
	}
	if !available {
		return nil, false, nil
	}

	if s.execution == nil {
		s.execution = device.NewExecution(runtimeValue)
	} else if err = s.execution.Reset(runtimeValue); err != nil {
		return nil, false, err
	}

	execution = s.execution
	return execution, true, nil
}

func (s *Sequential) lossGradient(
	lossFunc loss.Loss,
	predictions,
	targets *matrix.Matrix,
) (gradient *matrix.Matrix, err error) {
	var (
		destinationLoss loss.DestinationGradient
		rows            int
		cols            int
		ok              bool
	)

	destinationLoss, ok = lossFunc.(loss.DestinationGradient)
	if !ok {
		gradient, err = lossFunc.Gradient(predictions, targets)
		return gradient, err
	}

	rows, cols = predictions.Shape()
	if gradient, _, err = s.gradientPool.Get(rows, cols); err != nil {
		return nil, err
	}

	if err = destinationLoss.GradientInto(predictions, targets, gradient); err != nil {
		return nil, err
	}

	return gradient, nil
}

// Fit trains the model across multiple epochs using mini-batches.
func (s *Sequential) Fit(trainingData *data.Dataset, config FitConfig) (history TrainingHistory, err error) {
	var (
		epoch              int
		metrics            EpochMetrics
		earlyStoppingState earlyStoppingState
		scratch            fitScratch
	)
	s.invalidateLengthAwareForward()
	defer func() {
		var cleanupErr error

		if cleanupErr = scratch.release(); cleanupErr != nil {
			cleanupErr = fmt.Errorf("model: release fit scratch: %w", cleanupErr)
			err = errors.Join(err, cleanupErr)
		}
	}()

	if err = s.validateReady(); err != nil {
		return history, err
	}

	if err = s.validateOrdinaryGraph("fit", "FitWithLengths"); err != nil {
		return history, err
	}

	if err = validateFitDataset("training", trainingData); err != nil {
		return history, err
	}

	if err = config.validate(); err != nil {
		return history, err
	}

	if config.ValidationData != nil {
		if err = validateFitDataset("validation", config.ValidationData); err != nil {
			return history, err
		}
	}

	earlyStoppingState = newEarlyStoppingState(config.EarlyStopping)
	for epoch = 1; epoch <= config.Epochs; epoch++ {
		if err = applyLearningRateSchedule(config, epoch); err != nil {
			return history, err
		}

		if err = s.trainFitEpoch(trainingData, config, epoch, &scratch); err != nil {
			return history, err
		}

		if metrics, err = s.fitEpochMetrics(epoch, trainingData, config, &scratch); err != nil {
			return history, err
		}

		history.record(metrics)

		if config.Callback != nil {
			if err = config.Callback(metrics); err != nil {
				err = fmt.Errorf("model: epoch %d callback failed: %w", epoch, err)
				return history, err
			}
		}

		if earlyStoppingState.observe(metrics) {
			break
		}
	}

	return history, nil
}

// FitWithLengths trains across aligned sequence datasets and logical lengths.
func (s *Sequential) FitWithLengths(
	trainingData *data.SequenceDataset,
	config SequenceFitConfig,
) (history TrainingHistory, err error) {
	var (
		epoch              int
		metrics            EpochMetrics
		selector           sequenceLengthLayer
		selectorIndex      int
		earlyStoppingState earlyStoppingState
		scratch            fitScratch
	)

	s.invalidateLengthAwareForward()
	defer func() {
		var cleanupErr error

		s.invalidateLengthAwareForward()
		if cleanupErr = scratch.release(); cleanupErr != nil {
			cleanupErr = fmt.Errorf("model: release sequence fit scratch: %w", cleanupErr)
			err = errors.Join(err, cleanupErr)
		}
	}()

	if selector, selectorIndex, err = s.lengthAwareGraph("fit"); err != nil {
		return history, err
	}

	if err = validateSequenceFitDataset(
		"training",
		trainingData,
		selector.InputShape().Steps(),
	); err != nil {
		return history, err
	}

	if err = config.validate(); err != nil {
		return history, err
	}

	if config.ValidationData != nil {
		if err = validateSequenceFitDataset(
			"validation",
			config.ValidationData,
			selector.InputShape().Steps(),
		); err != nil {
			return history, err
		}
	}

	earlyStoppingState = newEarlyStoppingState(config.EarlyStopping)
	for epoch = 1; epoch <= config.Epochs; epoch++ {
		if err = applySequenceLearningRateSchedule(config, epoch); err != nil {
			return history, err
		}

		if err = s.trainSequenceFitEpoch(
			trainingData,
			config,
			epoch,
			selector,
			selectorIndex,
			&scratch,
		); err != nil {
			return history, err
		}

		if metrics, err = s.sequenceFitEpochMetrics(
			epoch,
			trainingData,
			config,
			selector,
			selectorIndex,
			&scratch,
		); err != nil {
			return history, err
		}

		history.record(metrics)

		if config.Callback != nil {
			if err = config.Callback(metrics); err != nil {
				err = fmt.Errorf("model: sequence epoch %d callback failed: %w", epoch, err)
				return history, err
			}
		}

		if earlyStoppingState.observe(metrics) {
			break
		}
	}

	return history, nil
}

// Save writes the model using the v1 JSON contract.
//
// The document uses format "neuralnetwork.sequential", version 1, and layer
// types "dense", "activation", "dropout", "batch_normalization", "conv2d",
// "max_pool2d", "flatten", "simple_rnn", "last_step", or
// "gather_last_valid". It stores supported layer configuration, trainable
// parameter values, and batch-normalization running statistics. It does not
// serialize optimizer state, accumulated gradients, forward caches, recurrent
// hidden histories, gathered length snapshots, training history, callbacks,
// learning-rate schedules, or original random source state. Sequence and
// spatial fields are additive, so existing version 1 documents retain their
// encoding and compatibility.
func (s *Sequential) Save(writer io.Writer) (err error) {
	if writer == nil {
		err = errors.New("model: save writer is nil")
		return err
	}

	if err = s.validate(); err != nil {
		return err
	}

	if err = encodeSequential(writer, s); err != nil {
		err = fmt.Errorf("model: save sequential: %w", err)
		return err
	}

	return nil
}

func (s *Sequential) trainFitEpoch(trainingData *data.Dataset, config FitConfig, epoch int, scratch *fitScratch) (err error) {
	var (
		indexes []int
		start   int
		end     int
		inputs  *matrix.Matrix
		targets *matrix.Matrix
	)

	indexes = scratch.rowIndexes(trainingData.SampleCount())
	if config.Shuffle {
		config.Random.Shuffle(len(indexes), func(left, right int) {
			indexes[left], indexes[right] = indexes[right], indexes[left]
		})
	}

	for start = 0; start < len(indexes); start += config.BatchSize {
		end = start + config.BatchSize
		if end > len(indexes) {
			end = len(indexes)
		}

		if inputs, targets, err = scratch.batchMatrices(trainingData, indexes[start:end]); err != nil {
			err = fmt.Errorf("model: epoch %d batch matrix copy failed: %w", epoch, err)
			return err
		}

		if _, err = s.TrainBatch(inputs, targets, config.Loss, config.Optimizer); err != nil {
			err = fmt.Errorf("model: epoch %d train batch failed: %w", epoch, err)
			return err
		}
	}

	return nil
}

func (s *Sequential) trainSequenceFitEpoch(
	trainingData *data.SequenceDataset,
	config SequenceFitConfig,
	epoch int,
	selector sequenceLengthLayer,
	selectorIndex int,
	scratch *fitScratch,
) (err error) {
	var (
		indexes      []int
		lengthValues []int
		start        int
		end          int
		batch        int
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
	)

	indexes = scratch.rowIndexes(trainingData.SampleCount())
	if config.Shuffle {
		config.Random.Shuffle(len(indexes), func(left, right int) {
			indexes[left], indexes[right] = indexes[right], indexes[left]
		})
	}

	for start = 0; start < len(indexes); start += config.BatchSize {
		end = start + config.BatchSize
		if end > len(indexes) {
			end = len(indexes)
		}
		batch++

		if inputs, targets, lengthValues, err = scratch.batchSequenceValues(
			trainingData,
			indexes[start:end],
		); err != nil {
			err = fmt.Errorf(
				"model: sequence epoch %d batch %d copy failed: %w",
				epoch,
				batch,
				err,
			)
			return err
		}

		s.invalidateLengthAwareForward()
		if lengthValues, err = s.prepareLengthValues(
			inputs,
			trainingData.Steps(),
			lengthValues,
			selector,
			"training",
		); err != nil {
			err = fmt.Errorf(
				"model: sequence epoch %d batch %d validation failed: %w",
				epoch,
				batch,
				err,
			)
			return err
		}

		if err = validateLengthAwareTargets(inputs, targets); err != nil {
			err = fmt.Errorf(
				"model: sequence epoch %d batch %d target validation failed: %w",
				epoch,
				batch,
				err,
			)
			return err
		}

		if _, err = s.trainPreparedWithLengths(
			inputs,
			targets,
			lengthValues,
			selector,
			selectorIndex,
			config.Loss,
			config.Optimizer,
		); err != nil {
			err = fmt.Errorf(
				"model: sequence epoch %d batch %d training failed: %w",
				epoch,
				batch,
				err,
			)
			return err
		}
		s.invalidateLengthAwareForward()
	}

	return nil
}

func (s *Sequential) fitEpochMetrics(epoch int, trainingData *data.Dataset, config FitConfig, scratch *fitScratch) (metrics EpochMetrics, err error) {
	var (
		accuracy    float32
		hasAccuracy bool
	)

	metrics.Epoch = epoch
	if metrics.Loss, accuracy, hasAccuracy, err = s.evaluateFitDataset(trainingData, config.Loss, config.Accuracy, &scratch.trainingEvaluation); err != nil {
		err = fmt.Errorf("model: epoch %d training evaluation failed: %w", epoch, err)
		return metrics, err
	}

	if hasAccuracy {
		metrics.Accuracy = accuracy
		metrics.HasAccuracy = true
	}

	if config.ValidationData == nil {
		return metrics, nil
	}

	if metrics.ValidationLoss, accuracy, hasAccuracy, err = s.evaluateFitDataset(config.ValidationData, config.Loss, config.Accuracy, &scratch.validationEvaluation); err != nil {
		err = fmt.Errorf("model: epoch %d validation evaluation failed: %w", epoch, err)
		return metrics, err
	}

	metrics.HasValidationLoss = true
	if hasAccuracy {
		metrics.ValidationAccuracy = accuracy
		metrics.HasValidationAccuracy = true
	}

	return metrics, nil
}

func (s *Sequential) sequenceFitEpochMetrics(
	epoch int,
	trainingData *data.SequenceDataset,
	config SequenceFitConfig,
	selector sequenceLengthLayer,
	selectorIndex int,
	scratch *fitScratch,
) (metrics EpochMetrics, err error) {
	var (
		accuracy    float32
		hasAccuracy bool
	)

	metrics.Epoch = epoch
	if metrics.Loss, accuracy, hasAccuracy, err = s.evaluateSequenceFitDataset(
		trainingData,
		config.Loss,
		config.Accuracy,
		selector,
		selectorIndex,
		&scratch.trainingEvaluation,
		&scratch.trainingEvaluationLengths,
	); err != nil {
		err = fmt.Errorf("model: sequence epoch %d training evaluation failed: %w", epoch, err)
		return metrics, err
	}

	if hasAccuracy {
		metrics.Accuracy = accuracy
		metrics.HasAccuracy = true
	}

	if config.ValidationData == nil {
		return metrics, nil
	}

	if metrics.ValidationLoss, accuracy, hasAccuracy, err = s.evaluateSequenceFitDataset(
		config.ValidationData,
		config.Loss,
		config.Accuracy,
		selector,
		selectorIndex,
		&scratch.validationEvaluation,
		&scratch.validationEvaluationLengths,
	); err != nil {
		err = fmt.Errorf("model: sequence epoch %d validation evaluation failed: %w", epoch, err)
		return metrics, err
	}

	metrics.HasValidationLoss = true
	if hasAccuracy {
		metrics.ValidationAccuracy = accuracy
		metrics.HasValidationAccuracy = true
	}

	return metrics, nil
}

func (s *Sequential) evaluateFitDataset(
	dataset *data.Dataset,
	lossFunc loss.Loss,
	accuracyFunc AccuracyFunc,
	matrices *fitMatrixPair,
) (lossValue, accuracyValue float32, hasAccuracy bool, err error) {
	var (
		previousTraining bool
		inputs           *matrix.Matrix
		targets          *matrix.Matrix
		predictions      *matrix.Matrix
	)

	if inputs, targets, err = matrices.datasetMatrices(dataset); err != nil {
		return 0, 0, false, err
	}

	previousTraining = s.Training()
	if err = s.SetTraining(false); err != nil {
		return 0, 0, false, err
	}
	defer func() {
		var restoreErr error

		if restoreErr = s.SetTraining(previousTraining); restoreErr != nil && err == nil {
			err = restoreErr
		}
	}()

	if predictions, err = s.Predict(inputs); err != nil {
		return 0, 0, false, err
	}

	if lossValue, err = lossFunc.Value(predictions, targets); err != nil {
		return 0, 0, false, err
	}

	if accuracyFunc == nil {
		return lossValue, 0, false, nil
	}

	if accuracyValue, err = accuracyFunc(predictions, targets); err != nil {
		return lossValue, 0, false, err
	}

	return lossValue, accuracyValue, true, nil
}

func (s *Sequential) evaluateSequenceFitDataset(
	dataset *data.SequenceDataset,
	lossFunc loss.Loss,
	accuracyFunc AccuracyFunc,
	selector sequenceLengthLayer,
	selectorIndex int,
	matrices *fitMatrixPair,
	lengthScratch *[]int,
) (lossValue, accuracyValue float32, hasAccuracy bool, err error) {
	var (
		previousTraining bool
		lengthValues     []int
		inputs           *matrix.Matrix
		targets          *matrix.Matrix
		predictions      *matrix.Matrix
	)

	if inputs, targets, lengthValues, err = matrices.sequenceDatasetValues(
		dataset,
		lengthScratch,
	); err != nil {
		return 0, 0, false, err
	}

	s.invalidateLengthAwareForward()
	if lengthValues, err = s.prepareLengthValues(
		inputs,
		dataset.Steps(),
		lengthValues,
		selector,
		"evaluation",
	); err != nil {
		return 0, 0, false, err
	}

	previousTraining = s.Training()
	if err = s.SetTraining(false); err != nil {
		return 0, 0, false, err
	}
	defer func() {
		var restoreErr error

		if restoreErr = s.SetTraining(previousTraining); restoreErr != nil && err == nil {
			err = restoreErr
		}
	}()

	if predictions, err = s.runLengthAwarePrediction(
		inputs,
		lengthValues,
		selector,
		selectorIndex,
	); err != nil {
		return 0, 0, false, err
	}

	if lossValue, err = lossFunc.Value(predictions, targets); err != nil {
		return 0, 0, false, err
	}

	if accuracyFunc == nil {
		return lossValue, 0, false, nil
	}

	if accuracyValue, err = accuracyFunc(predictions, targets); err != nil {
		return lossValue, 0, false, err
	}

	return lossValue, accuracyValue, true, nil
}

func (s *Sequential) prepareLengthAwareInput(
	input *matrix.Matrix,
	lengths *data.SequenceLengths,
	operation string,
) (
	selector sequenceLengthLayer,
	selectorIndex int,
	values []int,
	err error,
) {
	var rows int

	if selector, selectorIndex, err = s.lengthAwareGraph(operation); err != nil {
		return nil, 0, nil, err
	}

	if rows, err = validateLengthAwareInput(input); err != nil {
		return nil, 0, nil, err
	}

	if lengths == nil {
		err = fmt.Errorf("model: %s sequence lengths are nil", operation)
		return nil, 0, nil, err
	}

	if err = lengths.Validate(); err != nil {
		err = fmt.Errorf("model: %s sequence lengths invalid: %w", operation, err)
		return nil, 0, nil, err
	}

	if lengths.SampleCount() != rows {
		err = fmt.Errorf(
			"model: %s sequence length count mismatch: got=%d want=%d",
			operation,
			lengths.SampleCount(),
			rows,
		)
		return nil, 0, nil, err
	}

	if lengths.Steps() != selector.InputShape().Steps() {
		err = fmt.Errorf(
			"model: %s sequence length steps mismatch: got=%d want=%d",
			operation,
			lengths.Steps(),
			selector.InputShape().Steps(),
		)
		return nil, 0, nil, err
	}

	values = s.resizeLengthValues(rows)
	if err = lengths.ValuesInto(values); err != nil {
		err = fmt.Errorf("model: %s copy sequence lengths: %w", operation, err)
		return nil, 0, nil, err
	}

	return selector, selectorIndex, values, nil
}

func (s *Sequential) prepareLengthValues(
	input *matrix.Matrix,
	steps int,
	source []int,
	selector sequenceLengthLayer,
	operation string,
) (values []int, err error) {
	var (
		rows   int
		row    int
		length int
	)

	if rows, err = validateLengthAwareInput(input); err != nil {
		return nil, err
	}

	if steps != selector.InputShape().Steps() {
		err = fmt.Errorf(
			"model: %s sequence length steps mismatch: got=%d want=%d",
			operation,
			steps,
			selector.InputShape().Steps(),
		)
		return nil, err
	}

	if len(source) != rows {
		err = fmt.Errorf(
			"model: %s sequence length count mismatch: got=%d want=%d",
			operation,
			len(source),
			rows,
		)
		return nil, err
	}

	for row, length = range source {
		if length < 1 || length > steps {
			err = fmt.Errorf(
				"model: %s sequence length out of range: row=%d value=%d want=1..%d",
				operation,
				row,
				length,
				steps,
			)
			return nil, err
		}
	}

	values = s.resizeLengthValues(rows)
	copy(values, source)
	return values, nil
}

func (s *Sequential) resizeLengthValues(count int) (values []int) {
	if cap(s.lengthValues) < count {
		s.lengthValues = make([]int, count)
	} else {
		s.lengthValues = s.lengthValues[:count]
	}

	values = s.lengthValues
	return values
}

func (s *Sequential) validateOrdinaryGraph(operation, alternative string) (err error) {
	var (
		index   int
		current layer.Layer
		ok      bool
	)

	if err = s.validateReady(); err != nil {
		return err
	}

	for index, current = range s.layers {
		if _, ok = current.(*layer.GatherLastValid); ok {
			err = fmt.Errorf(
				"model: %s graph contains gather last valid at layer %d; use %s",
				operation,
				index,
				alternative,
			)
			return err
		}
	}

	return nil
}

func (s *Sequential) lengthAwareGraph(
	operation string,
) (selector sequenceLengthLayer, selectorIndex int, err error) {
	var (
		index       int
		current     layer.Layer
		gather      *layer.GatherLastValid
		gatherCount int
		ok          bool
	)

	if err = s.validateReady(); err != nil {
		return nil, 0, err
	}

	for index, current = range s.layers {
		if gather, ok = current.(*layer.GatherLastValid); !ok {
			continue
		}

		gatherCount++
		selector = gather
		selectorIndex = index
	}

	if gatherCount != 1 {
		err = fmt.Errorf(
			"model: %s requires exactly one gather last valid layer: got=%d",
			operation,
			gatherCount,
		)
		return nil, 0, err
	}

	if selector.InputShape().Steps() <= 0 || selector.InputShape().FeatureSize() <= 0 {
		err = fmt.Errorf(
			"model: %s gather last valid layer %d is invalid",
			operation,
			selectorIndex,
		)
		return nil, 0, err
	}

	return selector, selectorIndex, nil
}

func (s *Sequential) validateLengthAwareBackward(
	outputGradient *matrix.Matrix,
) (err error) {
	var (
		rows int
		cols int
	)

	if err = s.validateReady(); err != nil {
		return err
	}

	if !s.lengthAwareForward {
		err = errors.New(
			"model: length-aware backward called before matching PredictWithLengths",
		)
		return err
	}

	if outputGradient == nil {
		err = errors.New("model: length-aware output gradient is nil")
		return err
	}

	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("model: length-aware output gradient matrix invalid: %w", err)
		return err
	}

	rows, cols = outputGradient.Shape()
	if rows != s.lengthAwareForwardRows || cols != s.lengthAwareForwardColumns {
		err = fmt.Errorf(
			"model: length-aware output gradient shape mismatch: got %dx%d, want %dx%d",
			rows,
			cols,
			s.lengthAwareForwardRows,
			s.lengthAwareForwardColumns,
		)
		return err
	}

	return nil
}

func (s *Sequential) invalidateLengthAwareForward() {
	if s == nil {
		return
	}

	s.lengthAwareForward = false
	s.lengthAwareForwardRows = 0
	s.lengthAwareForwardColumns = 0
}

func (s *Sequential) recordLengthAwareForward(rows, columns int) {
	s.lengthAwareForward = true
	s.lengthAwareForwardRows = rows
	s.lengthAwareForwardColumns = columns
}

func (s *Sequential) validate() (err error) {
	if s == nil {
		err = errors.New("model: sequential model is nil")
		return err
	}

	return nil
}

func validateLengthAwareInput(input *matrix.Matrix) (rows int, err error) {
	if input == nil {
		err = errors.New("model: length-aware input is nil")
		return 0, err
	}

	if err = input.Validate(); err != nil {
		err = fmt.Errorf("model: length-aware input matrix invalid: %w", err)
		return 0, err
	}

	rows = input.Rows()
	return rows, nil
}

func validateLengthAwareTargets(input, targets *matrix.Matrix) (err error) {
	if targets == nil {
		err = errors.New("model: length-aware targets are nil")
		return err
	}

	if err = targets.Validate(); err != nil {
		err = fmt.Errorf("model: length-aware target matrix invalid: %w", err)
		return err
	}

	if targets.Rows() != input.Rows() {
		err = fmt.Errorf(
			"model: length-aware target row count mismatch: got=%d want=%d",
			targets.Rows(),
			input.Rows(),
		)
		return err
	}

	return nil
}

func (s *Sequential) validateReady() (err error) {
	if err = s.validate(); err != nil {
		return err
	}

	if len(s.layers) == 0 {
		err = errors.New("model: sequential model has no layers")
		return err
	}

	return nil
}

func validateFitDataset(name string, dataset *data.Dataset) (err error) {
	if dataset == nil {
		err = fmt.Errorf("model: %s dataset is nil", name)
		return err
	}

	if err = dataset.Validate(); err != nil {
		err = fmt.Errorf("model: %s dataset invalid: %w", name, err)
		return err
	}

	return nil
}

func validateSequenceFitDataset(
	name string,
	dataset *data.SequenceDataset,
	steps int,
) (err error) {
	if dataset == nil {
		err = fmt.Errorf("model: %s sequence dataset is nil", name)
		return err
	}

	if err = dataset.Validate(); err != nil {
		err = fmt.Errorf("model: %s sequence dataset invalid: %w", name, err)
		return err
	}

	if dataset.Steps() != steps {
		err = fmt.Errorf(
			"model: %s sequence dataset steps mismatch: got=%d want=%d",
			name,
			dataset.Steps(),
			steps,
		)
		return err
	}

	return nil
}

type fitScratch struct {
	indexes                     []int
	batchLengths                []int
	trainingEvaluationLengths   []int
	validationEvaluationLengths []int
	batch                       fitMatrixPair
	trainingEvaluation          fitMatrixPair
	validationEvaluation        fitMatrixPair
}

type fitMatrixPair struct {
	inputs  scratch.MatrixPool
	targets scratch.MatrixPool
}

func (s *fitScratch) release() (err error) {
	var releaseErr error

	if s == nil {
		return nil
	}
	if releaseErr = s.batch.release(); releaseErr != nil {
		err = errors.Join(err, releaseErr)
	}
	if releaseErr = s.trainingEvaluation.release(); releaseErr != nil {
		err = errors.Join(err, releaseErr)
	}
	if releaseErr = s.validationEvaluation.release(); releaseErr != nil {
		err = errors.Join(err, releaseErr)
	}
	return err
}

func (p *fitMatrixPair) release() (err error) {
	var releaseErr error

	if p == nil {
		return nil
	}
	if releaseErr = p.inputs.Release(); releaseErr != nil {
		err = errors.Join(err, releaseErr)
	}
	if releaseErr = p.targets.Release(); releaseErr != nil {
		err = errors.Join(err, releaseErr)
	}
	return err
}

func (s *fitScratch) rowIndexes(count int) (indexes []int) {
	var index int

	if cap(s.indexes) < count {
		s.indexes = make([]int, count)
	} else {
		s.indexes = s.indexes[:count]
	}

	for index = range s.indexes {
		s.indexes[index] = index
	}

	indexes = s.indexes
	return indexes
}

func (s *fitScratch) batchMatrices(dataset *data.Dataset, indexes []int) (inputs, targets *matrix.Matrix, err error) {
	if inputs, targets, err = s.batch.get(len(indexes), dataset.InputSize(), dataset.TargetSize()); err != nil {
		return nil, nil, err
	}

	if err = dataset.SelectRowsInto(indexes, inputs, targets); err != nil {
		return nil, nil, err
	}

	return inputs, targets, nil
}

func (s *fitScratch) batchSequenceValues(
	dataset *data.SequenceDataset,
	indexes []int,
) (inputs, targets *matrix.Matrix, lengths []int, err error) {
	if inputs, targets, err = s.batch.get(
		len(indexes),
		dataset.InputSize(),
		dataset.TargetSize(),
	); err != nil {
		return nil, nil, nil, err
	}

	s.batchLengths = resizeIntSlice(s.batchLengths, len(indexes))
	if err = dataset.SelectRowsInto(
		indexes,
		inputs,
		targets,
		s.batchLengths,
	); err != nil {
		return nil, nil, nil, err
	}

	lengths = s.batchLengths
	return inputs, targets, lengths, nil
}

func (p *fitMatrixPair) datasetMatrices(dataset *data.Dataset) (inputs, targets *matrix.Matrix, err error) {
	if inputs, targets, err = p.get(dataset.SampleCount(), dataset.InputSize(), dataset.TargetSize()); err != nil {
		return nil, nil, err
	}

	if err = dataset.InputsInto(inputs); err != nil {
		return nil, nil, err
	}

	if err = dataset.TargetsInto(targets); err != nil {
		return nil, nil, err
	}

	return inputs, targets, nil
}

func (p *fitMatrixPair) sequenceDatasetValues(
	dataset *data.SequenceDataset,
	lengthScratch *[]int,
) (inputs, targets *matrix.Matrix, lengths []int, err error) {
	if inputs, targets, err = p.get(
		dataset.SampleCount(),
		dataset.InputSize(),
		dataset.TargetSize(),
	); err != nil {
		return nil, nil, nil, err
	}

	*lengthScratch = resizeIntSlice(*lengthScratch, dataset.SampleCount())
	if err = dataset.InputsInto(inputs); err != nil {
		return nil, nil, nil, err
	}

	if err = dataset.TargetsInto(targets); err != nil {
		return nil, nil, nil, err
	}

	if err = dataset.LengthsInto(*lengthScratch); err != nil {
		return nil, nil, nil, err
	}

	lengths = *lengthScratch
	return inputs, targets, lengths, nil
}

func (p *fitMatrixPair) get(rows, inputSize, targetSize int) (inputs, targets *matrix.Matrix, err error) {
	if inputs, _, err = p.inputs.Get(rows, inputSize); err != nil {
		return nil, nil, err
	}

	if targets, _, err = p.targets.Get(rows, targetSize); err != nil {
		return nil, nil, err
	}

	return inputs, targets, nil
}

func resizeIntSlice(values []int, count int) (out []int) {
	if cap(values) < count {
		values = make([]int, count)
	} else {
		values = values[:count]
	}

	out = values
	return out
}

func applyLearningRateSchedule(config FitConfig, epoch int) (err error) {
	var learningRate float32

	if config.LearningRateSchedule == nil {
		return nil
	}

	if learningRate, err = config.LearningRateSchedule.LearningRate(epoch); err != nil {
		err = fmt.Errorf("model: epoch %d learning rate schedule failed: %w", epoch, err)
		return err
	}

	if err = config.Optimizer.SetLearningRate(learningRate); err != nil {
		err = fmt.Errorf("model: epoch %d learning rate update failed: %w", epoch, err)
		return err
	}

	return nil
}

func applySequenceLearningRateSchedule(config SequenceFitConfig, epoch int) (err error) {
	var learningRate float32

	if config.LearningRateSchedule == nil {
		return nil
	}

	if learningRate, err = config.LearningRateSchedule.LearningRate(epoch); err != nil {
		err = fmt.Errorf("model: sequence epoch %d learning rate schedule failed: %w", epoch, err)
		return err
	}

	if err = config.Optimizer.SetLearningRate(learningRate); err != nil {
		err = fmt.Errorf("model: sequence epoch %d learning rate update failed: %w", epoch, err)
		return err
	}

	return nil
}
