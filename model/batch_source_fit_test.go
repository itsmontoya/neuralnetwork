package model_test

import (
	"context"
	"errors"
	"math/rand"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_Sequential_FitWithBatchSourceMatchesFit(t *testing.T) {
	var (
		copiedNetwork    *model.Sequential
		sourceNetwork    *model.Sequential
		dataset          = mustFitDataset(t)
		trainingSource   *reusingBatchSource
		validationSource *reusingBatchSource
		copiedOptimizer  *optimizer.SGD
		sourceOptimizer  *optimizer.SGD
		copiedSchedule   *recordingSchedule
		sourceSchedule   *recordingSchedule
		copiedHistory    model.TrainingHistory
		sourceHistory    model.TrainingHistory
		copiedCallbacks  []model.EpochMetrics
		sourceCallbacks  []model.EpochMetrics
		copiedParameters []*optimizer.Parameter
		sourceParameters []*optimizer.Parameter
		parameterIndex   int
		err              error
	)

	if copiedNetwork, err = model.NewSequential(mustDense(t)); err != nil {
		t.Fatalf("copied NewSequential returned error: %v", err)
	}
	if sourceNetwork, err = model.NewSequential(mustDense(t)); err != nil {
		t.Fatalf("source NewSequential returned error: %v", err)
	}
	trainingSource = newReusingBatchSource(
		4,
		2,
		1,
		2,
		[]float32{0, 0, 1, 0, 0, 1, 1, 1},
		[]float32{1, 3, -2, 0},
	)
	validationSource = newReusingBatchSource(
		4,
		2,
		1,
		2,
		[]float32{0, 0, 1, 0, 0, 1, 1, 1},
		[]float32{1, 3, -2, 0},
	)
	if copiedOptimizer, err = optimizer.NewSGD(0.1); err != nil {
		t.Fatalf("copied NewSGD returned error: %v", err)
	}
	if sourceOptimizer, err = optimizer.NewSGD(0.1); err != nil {
		t.Fatalf("source NewSGD returned error: %v", err)
	}
	copiedSchedule = &recordingSchedule{rates: []float32{0.1, 0.05, 0.025}}
	sourceSchedule = &recordingSchedule{rates: []float32{0.1, 0.05, 0.025}}

	copiedHistory, err = copiedNetwork.Fit(dataset, model.FitConfig{
		Epochs:               3,
		BatchSize:            2,
		Shuffle:              true,
		Random:               rand.New(rand.NewSource(19)),
		Optimizer:            copiedOptimizer,
		LearningRateSchedule: copiedSchedule,
		Loss:                 loss.MeanSquaredError{},
		ValidationData:       dataset,
		Accuracy: func(predictions, targets *matrix.Matrix) (accuracy float32, err error) {
			return 0.75, nil
		},
		Callback: func(metrics model.EpochMetrics) (err error) {
			copiedCallbacks = append(copiedCallbacks, metrics)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Fit returned error: %v", err)
	}
	sourceHistory, err = sourceNetwork.FitWithBatchSource(
		context.Background(),
		trainingSource,
		model.BatchSourceFitConfig{
			Epochs:               3,
			Random:               rand.New(rand.NewSource(19)),
			Optimizer:            sourceOptimizer,
			LearningRateSchedule: sourceSchedule,
			Loss:                 loss.MeanSquaredError{},
			ValidationSource:     validationSource,
			Accuracy: func(predictions, targets *matrix.Matrix) (accuracy float32, err error) {
				return 0.75, nil
			},
			Callback: func(metrics model.EpochMetrics) (err error) {
				sourceCallbacks = append(sourceCallbacks, metrics)
				return nil
			},
		},
	)
	if err != nil {
		t.Fatalf("FitWithBatchSource returned error: %v", err)
	}

	requireBatchSourceHistories(t, sourceHistory, copiedHistory)
	requireBatchSourceHistories(t, model.TrainingHistory{Epochs: sourceCallbacks}, model.TrainingHistory{Epochs: copiedCallbacks})
	requireInts(t, sourceSchedule.epochs, copiedSchedule.epochs)
	requireInts(t, trainingSource.resetEpochs, []int{1, 1, 2, 2, 3, 3})
	requireBools(t, trainingSource.resetWithRandom, []bool{true, false, true, false, true, false})
	requireInts(t, validationSource.resetEpochs, []int{1, 2, 3})
	requireBools(t, validationSource.resetWithRandom, []bool{false, false, false})
	if !trainingSource.reusedInputs || !trainingSource.reusedTargets {
		t.Fatal("training source did not reuse its batch matrices")
	}

	copiedParameters = copiedNetwork.Parameters()
	sourceParameters = sourceNetwork.Parameters()
	if len(sourceParameters) != len(copiedParameters) {
		t.Fatalf("parameter count = %d, want %d", len(sourceParameters), len(copiedParameters))
	}
	for parameterIndex = range sourceParameters {
		requireMatrixValues(t, sourceParameters[parameterIndex].Values(), mustValues(t, copiedParameters[parameterIndex].Values()))
	}
}

func Test_Sequential_FitWithBatchSourceSupportsEarlyStopping(t *testing.T) {
	var (
		network        *model.Sequential
		source         *reusingBatchSource
		optimizerRule  *recordingOptimizer
		earlyStopping  *model.EarlyStopping
		history        model.TrainingHistory
		callbackEpochs []int
		err            error
	)

	if network, err = model.NewSequential(mustDense(t)); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	source = newReusingBatchSource(4, 2, 1, 2, []float32{0, 0, 1, 0, 0, 1, 1, 1}, []float32{1, 3, -2, 0})
	optimizerRule = &recordingOptimizer{}
	if earlyStopping, err = model.NewEarlyStopping(1, 0); err != nil {
		t.Fatalf("NewEarlyStopping returned error: %v", err)
	}
	history, err = network.FitWithBatchSource(context.Background(), source, model.BatchSourceFitConfig{
		Epochs:        10,
		Optimizer:     optimizerRule,
		EarlyStopping: earlyStopping,
		Loss:          loss.MeanSquaredError{},
		Callback: func(metrics model.EpochMetrics) (err error) {
			callbackEpochs = append(callbackEpochs, metrics.Epoch)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("FitWithBatchSource returned error: %v", err)
	}

	requireEpochCount(t, history, 2)
	requireInts(t, callbackEpochs, []int{1, 2})
	if optimizerRule.updateCalls != 4 {
		t.Fatalf("optimizer updates = %d, want 4", optimizerRule.updateCalls)
	}
}

func Test_Sequential_FitWithBatchSourceReturnsContextualErrors(t *testing.T) {
	var sourceErr error
	var callbackErr error
	sourceErr = errors.New("source failed")
	callbackErr = errors.New("callback failed")

	t.Run("training reset", func(t *testing.T) {
		var source *reusingBatchSource

		source = validReusingBatchSource()
		source.resetErrAt = 1
		source.sourceErr = sourceErr
		requireBatchSourceFitError(t, source, nil, nil, sourceErr, "epoch 1 training reset failed")
	})

	t.Run("training next", func(t *testing.T) {
		var source *reusingBatchSource

		source = validReusingBatchSource()
		source.nextErrAt = 1
		source.sourceErr = sourceErr
		requireBatchSourceFitError(t, source, nil, nil, sourceErr, "epoch 1 training batch 1 source failed")
	})

	t.Run("training evaluation reset", func(t *testing.T) {
		var source *reusingBatchSource

		source = validReusingBatchSource()
		source.resetErrAt = 2
		source.sourceErr = sourceErr
		requireBatchSourceFitError(t, source, nil, nil, sourceErr, "epoch 1 training evaluation reset failed")
	})

	t.Run("validation next", func(t *testing.T) {
		var validation *reusingBatchSource

		validation = validReusingBatchSource()
		validation.nextErrAt = 1
		validation.sourceErr = sourceErr
		requireBatchSourceFitError(t, validReusingBatchSource(), validation, nil, sourceErr, "epoch 1 validation evaluation batch 1 source failed")
	})

	t.Run("callback", func(t *testing.T) {
		requireBatchSourceFitError(t, validReusingBatchSource(), nil, callbackErr, callbackErr, "epoch 1 callback failed")
	})

	t.Run("reported count", func(t *testing.T) {
		var source *reusingBatchSource

		source = validReusingBatchSource()
		source.reportedSamples = 3
		requireBatchSourceFitError(t, source, nil, nil, nil, "sample count mismatch: got=4 want=3")
	})
}

func Test_Sequential_FitWithBatchSourceHonorsCancellation(t *testing.T) {
	var (
		network *model.Sequential
		source  *reusingBatchSource
		ctx     context.Context
		cancel  context.CancelFunc
		history model.TrainingHistory
		err     error
	)

	if network, err = model.NewSequential(mustDense(t)); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	source = validReusingBatchSource()
	source.onNext = cancel
	history, err = network.FitWithBatchSource(ctx, source, model.BatchSourceFitConfig{
		Epochs:    1,
		Optimizer: &recordingOptimizer{},
		Loss:      loss.MeanSquaredError{},
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("FitWithBatchSource error = %v, want context.Canceled", err)
	}
	if !strings.Contains(err.Error(), "epoch 1 training batch 1 source failed") {
		t.Fatalf("FitWithBatchSource error = %q, want epoch and batch context", err)
	}
	if len(history.Epochs) != 0 {
		t.Fatalf("history length = %d, want 0", len(history.Epochs))
	}
}

func Test_Sequential_FitWithBatchSourceValidatesContract(t *testing.T) {
	t.Run("nil context", func(t *testing.T) {
		var (
			network *model.Sequential
			err     error
		)

		if network, err = model.NewSequential(mustDense(t)); err != nil {
			t.Fatalf("NewSequential returned error: %v", err)
		}
		if _, err = network.FitWithBatchSource(nil, validReusingBatchSource(), validBatchSourceFitConfig()); err == nil || !strings.Contains(err.Error(), "context is nil") {
			t.Fatalf("FitWithBatchSource error = %v, want nil context error", err)
		}
	})

	t.Run("nil source", func(t *testing.T) {
		var (
			network *model.Sequential
			err     error
		)

		if network, err = model.NewSequential(mustDense(t)); err != nil {
			t.Fatalf("NewSequential returned error: %v", err)
		}
		if _, err = network.FitWithBatchSource(context.Background(), nil, validBatchSourceFitConfig()); err == nil || !strings.Contains(err.Error(), "source is nil") {
			t.Fatalf("FitWithBatchSource error = %v, want nil source error", err)
		}
	})

	t.Run("done with matrices", func(t *testing.T) {
		var source *reusingBatchSource

		source = validReusingBatchSource()
		source.doneWithMatrices = true
		requireBatchSourceFitError(t, source, nil, nil, nil, "source returned matrices with done")
	})

	t.Run("empty pass", func(t *testing.T) {
		var source *reusingBatchSource

		source = newReusingBatchSource(0, 2, 1, 2, nil, nil)
		requireBatchSourceFitError(t, source, nil, nil, nil, "training source returned no batches")
	})
}

func requireBatchSourceFitError(
	t *testing.T,
	source *reusingBatchSource,
	validation *reusingBatchSource,
	callbackErr error,
	wantCause error,
	wantText string,
) {
	var (
		network *model.Sequential
		history model.TrainingHistory
		config  model.BatchSourceFitConfig
		err     error
	)

	t.Helper()
	if network, err = model.NewSequential(mustDense(t)); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	config = validBatchSourceFitConfig()
	config.ValidationSource = validation
	if callbackErr != nil {
		config.Callback = func(model.EpochMetrics) (err error) {
			return callbackErr
		}
	}
	history, err = network.FitWithBatchSource(context.Background(), source, config)
	if err == nil {
		t.Fatal("FitWithBatchSource error = nil, want error")
	}
	if wantCause != nil && !errors.Is(err, wantCause) {
		t.Fatalf("FitWithBatchSource error = %v, want cause %v", err, wantCause)
	}
	if !strings.Contains(err.Error(), wantText) {
		t.Fatalf("FitWithBatchSource error = %q, want substring %q", err, wantText)
	}
	if callbackErr == nil && len(history.Epochs) != 0 {
		t.Fatalf("history length = %d before callback, want 0", len(history.Epochs))
	}
	if callbackErr != nil && len(history.Epochs) != 1 {
		t.Fatalf("history length = %d after callback error, want 1", len(history.Epochs))
	}
}

func validBatchSourceFitConfig() (config model.BatchSourceFitConfig) {
	config.Epochs = 1
	config.Optimizer = &recordingOptimizer{}
	config.Loss = loss.MeanSquaredError{}
	return config
}

func validReusingBatchSource() (source *reusingBatchSource) {
	source = newReusingBatchSource(4, 2, 1, 2, []float32{0, 0, 1, 0, 0, 1, 1, 1}, []float32{1, 3, -2, 0})
	return source
}

func requireBatchSourceHistories(t *testing.T, got, want model.TrainingHistory) {
	var index int

	t.Helper()
	if len(got.Epochs) != len(want.Epochs) {
		t.Fatalf("history length = %d, want %d", len(got.Epochs), len(want.Epochs))
	}
	for index = range got.Epochs {
		if got.Epochs[index].Epoch != want.Epochs[index].Epoch {
			t.Fatalf("epoch[%d] = %d, want %d", index, got.Epochs[index].Epoch, want.Epochs[index].Epoch)
		}
		requireEpochMetricsClose(t, got.Epochs[index], want.Epochs[index])
	}
}

func requireEpochMetricsClose(t *testing.T, got, want model.EpochMetrics) {
	t.Helper()
	if got.HasAccuracy != want.HasAccuracy || got.HasValidationLoss != want.HasValidationLoss || got.HasValidationAccuracy != want.HasValidationAccuracy {
		t.Fatalf("metric presence = %+v, want %+v", got, want)
	}
	requireFloatClose(t, "loss", got.Loss, want.Loss)
	requireFloatClose(t, "accuracy", got.Accuracy, want.Accuracy)
	requireFloatClose(t, "validation loss", got.ValidationLoss, want.ValidationLoss)
	requireFloatClose(t, "validation accuracy", got.ValidationAccuracy, want.ValidationAccuracy)
}

func requireFloatClose(t *testing.T, name string, got, want float32) {
	var difference float32

	t.Helper()
	difference = got - want
	if difference < 0 {
		difference = -difference
	}
	if difference > 1e-5 {
		t.Fatalf("%s = %g, want %g", name, got, want)
	}
}

func newReusingBatchSource(
	samples,
	inputSize,
	targetSize,
	batchSize int,
	inputValues,
	targetValues []float32,
) (source *reusingBatchSource) {
	var s reusingBatchSource

	s.samples = samples
	s.inputSize = inputSize
	s.targetSize = targetSize
	s.batchSize = batchSize
	s.inputValues = append([]float32(nil), inputValues...)
	s.targetValues = append([]float32(nil), targetValues...)
	s.reportedSamples = samples
	if batchSize > 0 {
		s.reportedBatches = (samples + batchSize - 1) / batchSize
	}
	return &s
}

type reusingBatchSource struct {
	samples          int
	inputSize        int
	targetSize       int
	batchSize        int
	inputValues      []float32
	targetValues     []float32
	order            []int
	next             int
	passNextCalls    int
	resetCalls       int
	resetErrAt       int
	nextErrAt        int
	sourceErr        error
	reportedSamples  int
	reportedBatches  int
	inputs           *matrix.Matrix
	targets          *matrix.Matrix
	lastInputs       *matrix.Matrix
	lastTargets      *matrix.Matrix
	reusedInputs     bool
	reusedTargets    bool
	doneWithMatrices bool
	onNext           func()
	resetEpochs      []int
	resetWithRandom  []bool
}

func (s *reusingBatchSource) Reset(epoch int, random *rand.Rand) (err error) {
	var index int

	s.resetCalls++
	if s.resetErrAt == s.resetCalls {
		return s.sourceErr
	}
	s.resetEpochs = append(s.resetEpochs, epoch)
	s.resetWithRandom = append(s.resetWithRandom, random != nil)
	s.order = resizeSourceOrder(s.order, s.samples)
	for index = range s.order {
		s.order[index] = index
	}
	if random != nil {
		random.Shuffle(len(s.order), func(left, right int) {
			s.order[left], s.order[right] = s.order[right], s.order[left]
		})
	}
	s.next = 0
	s.passNextCalls = 0
	return nil
}

func (s *reusingBatchSource) Next(ctx context.Context) (inputs, targets *matrix.Matrix, done bool, err error) {
	var (
		rows         int
		end          int
		row          int
		sourceRow    int
		inputIndex   int
		targetIndex  int
		batchInputs  []float32
		batchTargets []float32
	)

	s.passNextCalls++
	if s.onNext != nil {
		s.onNext()
		s.onNext = nil
	}
	if err = ctx.Err(); err != nil {
		return nil, nil, false, err
	}
	if s.nextErrAt == s.passNextCalls {
		return nil, nil, false, s.sourceErr
	}
	if s.next >= len(s.order) {
		if s.doneWithMatrices {
			return s.inputs, s.targets, true, nil
		}
		return nil, nil, true, nil
	}

	end = s.next + s.batchSize
	if end > len(s.order) {
		end = len(s.order)
	}
	rows = end - s.next
	batchInputs = make([]float32, rows*s.inputSize)
	batchTargets = make([]float32, rows*s.targetSize)
	for row = 0; row < rows; row++ {
		sourceRow = s.order[s.next+row]
		for inputIndex = 0; inputIndex < s.inputSize; inputIndex++ {
			batchInputs[row*s.inputSize+inputIndex] = s.inputValues[sourceRow*s.inputSize+inputIndex]
		}
		for targetIndex = 0; targetIndex < s.targetSize; targetIndex++ {
			batchTargets[row*s.targetSize+targetIndex] = s.targetValues[sourceRow*s.targetSize+targetIndex]
		}
	}
	if s.inputs == nil || s.inputs.Rows() != rows {
		if s.inputs, err = matrix.FromSlice(rows, s.inputSize, batchInputs); err != nil {
			return nil, nil, false, err
		}
	} else if err = s.inputs.CopyValuesFrom(batchInputs); err != nil {
		return nil, nil, false, err
	}
	if s.targets == nil || s.targets.Rows() != rows {
		if s.targets, err = matrix.FromSlice(rows, s.targetSize, batchTargets); err != nil {
			return nil, nil, false, err
		}
	} else if err = s.targets.CopyValuesFrom(batchTargets); err != nil {
		return nil, nil, false, err
	}
	if s.lastInputs == s.inputs {
		s.reusedInputs = true
	}
	if s.lastTargets == s.targets {
		s.reusedTargets = true
	}
	s.lastInputs = s.inputs
	s.lastTargets = s.targets
	s.next = end
	return s.inputs, s.targets, false, nil
}

func (s *reusingBatchSource) SampleCount() (count int, known bool) {
	return s.reportedSamples, true
}

func (s *reusingBatchSource) BatchCount() (count int, known bool) {
	return s.reportedBatches, true
}

func resizeSourceOrder(order []int, count int) (out []int) {
	if cap(order) < count {
		order = make([]int, count)
	} else {
		order = order[:count]
	}
	return order
}
