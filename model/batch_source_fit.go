package model

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"reflect"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

type batchSourceMetrics struct {
	lossSum     float64
	accuracySum float64
	samples     int
	batches     int
	hasAccuracy bool
}

func fitWithBatchSource(
	s *Sequential,
	ctx context.Context,
	trainingSource BatchSource,
	config BatchSourceFitConfig,
) (history TrainingHistory, err error) {
	var (
		epoch              int
		metrics            EpochMetrics
		earlyStoppingState earlyStoppingState
	)

	if ctx == nil {
		err = errors.New("model: batch source fit context is nil")
		return history, err
	}
	if isNilBatchSource(trainingSource) {
		err = errors.New("model: training batch source is nil")
		return history, err
	}
	if err = s.validateReady(); err != nil {
		return history, err
	}
	if err = s.validateOrdinaryGraph("batch source fit", "FitWithLengths"); err != nil {
		return history, err
	}
	if err = config.validate(); err != nil {
		return history, err
	}
	if err = contextError(ctx, "model: batch source fit canceled before training"); err != nil {
		return history, err
	}

	earlyStoppingState = newEarlyStoppingState(config.EarlyStopping)
	for epoch = 1; epoch <= config.Epochs; epoch++ {
		if err = applyBatchSourceLearningRateSchedule(config, epoch); err != nil {
			return history, err
		}
		if err = trainBatchSourceEpoch(s, ctx, trainingSource, config, epoch); err != nil {
			return history, err
		}
		if metrics, err = batchSourceEpochMetrics(s, ctx, trainingSource, config, epoch); err != nil {
			return history, err
		}

		history.record(metrics)
		if config.Callback != nil {
			if err = config.Callback(metrics); err != nil {
				err = fmt.Errorf("model: batch source fit epoch %d callback failed: %w", epoch, err)
				return history, err
			}
		}
		if earlyStoppingState.observe(metrics) {
			break
		}
	}

	return history, nil
}

func trainBatchSourceEpoch(
	s *Sequential,
	ctx context.Context,
	source BatchSource,
	config BatchSourceFitConfig,
	epoch int,
) (err error) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		batch   int
		samples int
		done    bool
	)

	if err = resetBatchSource(ctx, source, epoch, config.Random, "training"); err != nil {
		return err
	}
	for {
		if err = contextError(ctx, fmt.Sprintf("model: batch source fit epoch %d training canceled", epoch)); err != nil {
			return err
		}
		batch++
		if inputs, targets, done, err = source.Next(ctx); err != nil {
			err = fmt.Errorf("model: batch source fit epoch %d training batch %d source failed: %w", epoch, batch, err)
			return err
		}
		if err = contextError(ctx, fmt.Sprintf("model: batch source fit epoch %d training batch %d canceled", epoch, batch)); err != nil {
			return err
		}
		if done {
			if err = validateBatchSourceDone(inputs, targets); err != nil {
				err = fmt.Errorf("model: batch source fit epoch %d training batch %d: %w", epoch, batch, err)
				return err
			}
			batch--
			break
		}
		if err = validateBatchSourceMatrices(inputs, targets); err != nil {
			err = fmt.Errorf("model: batch source fit epoch %d training batch %d invalid: %w", epoch, batch, err)
			return err
		}
		if _, err = s.trainBatch(inputs, targets, config.Loss, config.Optimizer); err != nil {
			err = fmt.Errorf("model: batch source fit epoch %d training batch %d failed: %w", epoch, batch, err)
			return err
		}
		samples += inputs.Rows()
	}

	if batch == 0 {
		err = fmt.Errorf("model: batch source fit epoch %d training source returned no batches", epoch)
		return err
	}
	if err = validateBatchSourceCounts(source, epoch, "training", samples, batch); err != nil {
		return err
	}
	return nil
}

func batchSourceEpochMetrics(
	s *Sequential,
	ctx context.Context,
	trainingSource BatchSource,
	config BatchSourceFitConfig,
	epoch int,
) (metrics EpochMetrics, err error) {
	var (
		trainingMetrics   batchSourceMetrics
		validationMetrics batchSourceMetrics
	)

	metrics.Epoch = epoch
	if trainingMetrics, err = evaluateBatchSource(s, ctx, trainingSource, config, epoch, "training"); err != nil {
		return metrics, err
	}
	metrics.Loss = trainingMetrics.loss()
	if trainingMetrics.hasAccuracy {
		metrics.Accuracy = trainingMetrics.accuracy()
		metrics.HasAccuracy = true
	}

	if isNilBatchSource(config.ValidationSource) {
		return metrics, nil
	}
	if validationMetrics, err = evaluateBatchSource(s, ctx, config.ValidationSource, config, epoch, "validation"); err != nil {
		return metrics, err
	}
	metrics.ValidationLoss = validationMetrics.loss()
	metrics.HasValidationLoss = true
	if validationMetrics.hasAccuracy {
		metrics.ValidationAccuracy = validationMetrics.accuracy()
		metrics.HasValidationAccuracy = true
	}
	return metrics, nil
}

func evaluateBatchSource(
	s *Sequential,
	ctx context.Context,
	source BatchSource,
	config BatchSourceFitConfig,
	epoch int,
	name string,
) (metrics batchSourceMetrics, err error) {
	var (
		previousTraining bool
		inputs           *matrix.Matrix
		targets          *matrix.Matrix
		predictions      *matrix.Matrix
		lossValue        float32
		accuracyValue    float32
		batch            int
		done             bool
	)

	if err = resetBatchSource(ctx, source, epoch, nil, name+" evaluation"); err != nil {
		return metrics, err
	}
	previousTraining = s.Training()
	if err = s.SetTraining(false); err != nil {
		return metrics, err
	}
	defer func() {
		var restoreErr error

		if restoreErr = s.SetTraining(previousTraining); restoreErr != nil {
			err = errors.Join(err, restoreErr)
		}
	}()

	for {
		if err = contextError(ctx, fmt.Sprintf("model: batch source fit epoch %d %s evaluation canceled", epoch, name)); err != nil {
			return metrics, err
		}
		batch++
		if inputs, targets, done, err = source.Next(ctx); err != nil {
			err = fmt.Errorf("model: batch source fit epoch %d %s evaluation batch %d source failed: %w", epoch, name, batch, err)
			return metrics, err
		}
		if err = contextError(ctx, fmt.Sprintf("model: batch source fit epoch %d %s evaluation batch %d canceled", epoch, name, batch)); err != nil {
			return metrics, err
		}
		if done {
			if err = validateBatchSourceDone(inputs, targets); err != nil {
				err = fmt.Errorf("model: batch source fit epoch %d %s evaluation batch %d: %w", epoch, name, batch, err)
				return metrics, err
			}
			batch--
			break
		}
		if err = validateBatchSourceMatrices(inputs, targets); err != nil {
			err = fmt.Errorf("model: batch source fit epoch %d %s evaluation batch %d invalid: %w", epoch, name, batch, err)
			return metrics, err
		}
		if predictions, err = s.predictOwned(inputs); err != nil {
			err = fmt.Errorf("model: batch source fit epoch %d %s evaluation batch %d prediction failed: %w", epoch, name, batch, err)
			return metrics, err
		}
		if lossValue, err = config.Loss.Value(predictions, targets); err != nil {
			err = fmt.Errorf("model: batch source fit epoch %d %s evaluation batch %d loss failed: %w", epoch, name, batch, err)
			return metrics, err
		}
		if config.Accuracy != nil {
			if accuracyValue, err = config.Accuracy(predictions, targets); err != nil {
				err = fmt.Errorf("model: batch source fit epoch %d %s evaluation batch %d accuracy failed: %w", epoch, name, batch, err)
				return metrics, err
			}
			metrics.hasAccuracy = true
		}
		metrics.observe(inputs.Rows(), lossValue, accuracyValue)
	}

	if metrics.batches == 0 {
		err = fmt.Errorf("model: batch source fit epoch %d %s evaluation source returned no batches", epoch, name)
		return metrics, err
	}
	if err = validateBatchSourceCounts(source, epoch, name+" evaluation", metrics.samples, metrics.batches); err != nil {
		return metrics, err
	}
	return metrics, nil
}

func resetBatchSource(
	ctx context.Context,
	source BatchSource,
	epoch int,
	random *rand.Rand,
	name string,
) (err error) {
	if err = contextError(ctx, fmt.Sprintf("model: batch source fit epoch %d %s reset canceled", epoch, name)); err != nil {
		return err
	}
	if err = source.Reset(epoch, random); err != nil {
		err = fmt.Errorf("model: batch source fit epoch %d %s reset failed: %w", epoch, name, err)
		return err
	}
	return nil
}

func validateBatchSourceMatrices(inputs, targets *matrix.Matrix) (err error) {
	if inputs == nil {
		err = errors.New("input matrix is nil")
		return err
	}
	if err = inputs.Validate(); err != nil {
		err = fmt.Errorf("input matrix: %w", err)
		return err
	}
	if targets == nil {
		err = errors.New("target matrix is nil")
		return err
	}
	if err = targets.Validate(); err != nil {
		err = fmt.Errorf("target matrix: %w", err)
		return err
	}
	if targets.Rows() != inputs.Rows() {
		err = fmt.Errorf("target row count mismatch: got=%d want=%d", targets.Rows(), inputs.Rows())
		return err
	}
	return nil
}

func validateBatchSourceDone(inputs, targets *matrix.Matrix) (err error) {
	if inputs != nil || targets != nil {
		err = errors.New("source returned matrices with done")
		return err
	}
	return nil
}

func validateBatchSourceCounts(
	source BatchSource,
	epoch int,
	name string,
	samples,
	batches int,
) (err error) {
	var (
		sized          SizedBatchSource
		expected       int
		known          bool
		implementsSize bool
	)

	if sized, implementsSize = source.(SizedBatchSource); !implementsSize {
		return nil
	}
	if expected, known = sized.SampleCount(); known {
		if expected < 0 {
			err = fmt.Errorf("model: batch source fit epoch %d %s reported negative sample count %d", epoch, name, expected)
			return err
		}
		if samples != expected {
			err = fmt.Errorf("model: batch source fit epoch %d %s sample count mismatch: got=%d want=%d", epoch, name, samples, expected)
			return err
		}
	}
	if expected, known = sized.BatchCount(); known {
		if expected < 0 {
			err = fmt.Errorf("model: batch source fit epoch %d %s reported negative batch count %d", epoch, name, expected)
			return err
		}
		if batches != expected {
			err = fmt.Errorf("model: batch source fit epoch %d %s batch count mismatch: got=%d want=%d", epoch, name, batches, expected)
			return err
		}
	}
	return nil
}

func contextError(ctx context.Context, message string) (err error) {
	if err = ctx.Err(); err != nil {
		err = fmt.Errorf("%s: %w", message, err)
		return err
	}
	return nil
}

func isNilBatchSource(source BatchSource) (nilSource bool) {
	var value reflect.Value

	if source == nil {
		return true
	}
	value = reflect.ValueOf(source)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		nilSource = value.IsNil()
	}
	return nilSource
}

func applyBatchSourceLearningRateSchedule(config BatchSourceFitConfig, epoch int) (err error) {
	var learningRate float32

	if config.LearningRateSchedule == nil {
		return nil
	}
	if learningRate, err = config.LearningRateSchedule.LearningRate(epoch); err != nil {
		err = fmt.Errorf("model: batch source fit epoch %d learning rate schedule failed: %w", epoch, err)
		return err
	}
	if err = config.Optimizer.SetLearningRate(learningRate); err != nil {
		err = fmt.Errorf("model: batch source fit epoch %d learning rate update failed: %w", epoch, err)
		return err
	}
	return nil
}

func (m *batchSourceMetrics) observe(samples int, lossValue, accuracyValue float32) {
	m.lossSum += float64(lossValue) * float64(samples)
	m.accuracySum += float64(accuracyValue) * float64(samples)
	m.samples += samples
	m.batches++
}

func (m batchSourceMetrics) loss() (value float32) {
	value = float32(m.lossSum / float64(m.samples))
	return value
}

func (m batchSourceMetrics) accuracy() (value float32) {
	value = float32(m.accuracySum / float64(m.samples))
	return value
}
