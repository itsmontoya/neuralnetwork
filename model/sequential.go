// Package model provides neural network composition and training orchestration.
package model

import (
	"errors"
	"fmt"
	"io"

	"github.com/itsmontoya/neuralnetwork/data"
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

// LoadSequential restores a Sequential model from a v1 JSON serialization.
//
// Dropout layers are restored with deterministic local random sources.
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
	layers   []layer.Layer
	training bool
}

// Add appends a layer to the model.
func (s *Sequential) Add(next layer.Layer) (err error) {
	var modeLayer trainingModeLayer
	var ok bool

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

	output = input
	for index, current = range s.layers {
		if output, err = current.Forward(output); err != nil {
			err = fmt.Errorf("model: layer %d forward failed: %w", index, err)
			return nil, err
		}
	}

	return output, nil
}

// Backward runs a backward pass through every layer in reverse order.
func (s *Sequential) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
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

	inputGradient = outputGradient
	for index = len(s.layers) - 1; index >= 0; index-- {
		if inputGradient, err = s.layers[index].Backward(inputGradient); err != nil {
			err = fmt.Errorf("model: layer %d backward failed: %w", index, err)
			return nil, err
		}
	}

	return inputGradient, nil
}

// Parameters returns trainable parameters from all layers in layer order.
func (s *Sequential) Parameters() (parameters []*optimizer.Parameter) {
	var (
		current        layer.Layer
		parameterLayer parameterProvider
		ok             bool
	)

	if s == nil {
		return nil
	}

	for _, current = range s.layers {
		parameterLayer, ok = current.(parameterProvider)
		if !ok {
			continue
		}

		parameters = append(parameters, parameterLayer.Parameters()...)
	}

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
	)

	if lossFunc == nil {
		err = errors.New("model: loss is nil")
		return metrics, err
	}

	if optimizerRule == nil {
		err = errors.New("model: optimizer is nil")
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

	if predictions, err = s.Predict(input); err != nil {
		return metrics, err
	}

	if metrics.Loss, err = lossFunc.Value(predictions, targets); err != nil {
		err = fmt.Errorf("model: loss value failed: %w", err)
		return metrics, err
	}

	if gradient, err = lossFunc.Gradient(predictions, targets); err != nil {
		err = fmt.Errorf("model: loss gradient failed: %w", err)
		return metrics, err
	}

	if _, err = s.Backward(gradient); err != nil {
		err = fmt.Errorf("model: backward failed: %w", err)
		return metrics, err
	}

	if err = optimizerRule.Update(s.Parameters()); err != nil {
		err = fmt.Errorf("model: optimizer update failed: %w", err)
		return metrics, err
	}

	return metrics, nil
}

// Fit trains the model across multiple epochs using mini-batches.
func (s *Sequential) Fit(trainingData *data.Dataset, config FitConfig) (history TrainingHistory, err error) {
	var (
		epoch              int
		metrics            EpochMetrics
		earlyStoppingState earlyStoppingState
	)

	if err = s.validateReady(); err != nil {
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

		if err = s.trainFitEpoch(trainingData, config, epoch); err != nil {
			return history, err
		}

		if metrics, err = s.fitEpochMetrics(epoch, trainingData, config); err != nil {
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

// Save writes the model architecture and trainable parameter values as v1 JSON.
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

func (s *Sequential) trainFitEpoch(trainingData *data.Dataset, config FitConfig, epoch int) (err error) {
	var (
		batches []*data.Batch
		batch   *data.Batch
		inputs  *matrix.Matrix
		targets *matrix.Matrix
	)

	if config.Shuffle {
		batches, err = trainingData.Batches(config.BatchSize, config.Random)
	} else {
		batches, err = trainingData.Batches(config.BatchSize, nil)
	}
	if err != nil {
		err = fmt.Errorf("model: epoch %d batching failed: %w", epoch, err)
		return err
	}

	for _, batch = range batches {
		if inputs, err = batch.Inputs(); err != nil {
			err = fmt.Errorf("model: epoch %d batch inputs failed: %w", epoch, err)
			return err
		}

		if targets, err = batch.Targets(); err != nil {
			err = fmt.Errorf("model: epoch %d batch targets failed: %w", epoch, err)
			return err
		}

		if _, err = s.TrainBatch(inputs, targets, config.Loss, config.Optimizer); err != nil {
			err = fmt.Errorf("model: epoch %d train batch failed: %w", epoch, err)
			return err
		}
	}

	return nil
}

func (s *Sequential) fitEpochMetrics(epoch int, trainingData *data.Dataset, config FitConfig) (metrics EpochMetrics, err error) {
	var (
		accuracy    float64
		hasAccuracy bool
	)

	metrics.Epoch = epoch
	if metrics.Loss, accuracy, hasAccuracy, err = s.evaluateFitDataset(trainingData, config.Loss, config.Accuracy); err != nil {
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

	if metrics.ValidationLoss, accuracy, hasAccuracy, err = s.evaluateFitDataset(config.ValidationData, config.Loss, config.Accuracy); err != nil {
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

func (s *Sequential) evaluateFitDataset(dataset *data.Dataset, lossFunc loss.Loss, accuracyFunc AccuracyFunc) (lossValue, accuracyValue float64, hasAccuracy bool, err error) {
	var (
		previousTraining bool
		inputs           *matrix.Matrix
		targets          *matrix.Matrix
		predictions      *matrix.Matrix
	)

	if inputs, err = dataset.Inputs(); err != nil {
		return 0, 0, false, err
	}

	if targets, err = dataset.Targets(); err != nil {
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

func (s *Sequential) validate() (err error) {
	if s == nil {
		err = errors.New("model: sequential model is nil")
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

	if _, err = dataset.Inputs(); err != nil {
		err = fmt.Errorf("model: %s dataset inputs invalid: %w", name, err)
		return err
	}

	if _, err = dataset.Targets(); err != nil {
		err = fmt.Errorf("model: %s dataset targets invalid: %w", name, err)
		return err
	}

	return nil
}

func applyLearningRateSchedule(config FitConfig, epoch int) (err error) {
	var learningRate float64

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
