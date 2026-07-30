package model

import (
	"bytes"
	"errors"
	"math/rand"
	"strings"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

const viewFitTestTolerance = 1e-6

func Test_Sequential_FitWithViewsMatchesCopiedFit(t *testing.T) {
	type testcase struct {
		name    string
		shuffle bool
		policy  data.ViewPolicy
	}

	var tests []testcase

	tests = []testcase{
		{name: "ordered partial final batch", policy: data.ViewOnly},
		{name: "shuffled copied fallback", shuffle: true, policy: data.ViewOrCopy},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				dataset          *data.Dataset
				copiedModel      *Sequential
				viewModel        *Sequential
				copiedOptimizer  *optimizer.SGD
				viewOptimizer    *optimizer.SGD
				copiedSchedule   *optimizer.StepDecay
				viewSchedule     *optimizer.StepDecay
				copiedLoss       viewFitRecordingLoss
				viewLoss         viewFitRecordingLoss
				copiedCallbacks  []EpochMetrics
				viewCallbacks    []EpochMetrics
				copiedConfig     FitConfig
				viewConfig       ViewFitConfig
				copiedHistory    TrainingHistory
				viewHistory      TrainingHistory
				copiedPrediction *matrix.Matrix
				viewPrediction   *matrix.Matrix
				inputs           *matrix.Matrix
				err              error
			)

			dataset = viewFitOrdinaryDataset(t)
			copiedModel = viewFitOrdinaryModel(t)
			viewModel = viewFitOrdinaryModel(t)
			copiedOptimizer = viewFitSGD(t)
			viewOptimizer = viewFitSGD(t)
			if copiedSchedule, err = optimizer.NewStepDecay(0.03, 0.5, 2); err != nil {
				t.Fatalf("NewStepDecay copied returned error: %v", err)
			}
			if viewSchedule, err = optimizer.NewStepDecay(0.03, 0.5, 2); err != nil {
				t.Fatalf("NewStepDecay view returned error: %v", err)
			}

			copiedConfig.Epochs = 3
			copiedConfig.BatchSize = 2
			copiedConfig.Shuffle = tt.shuffle
			copiedConfig.Random = rand.New(rand.NewSource(41))
			copiedConfig.Optimizer = copiedOptimizer
			copiedConfig.LearningRateSchedule = copiedSchedule
			copiedConfig.Loss = &copiedLoss
			copiedConfig.ValidationData = dataset
			copiedConfig.Accuracy = viewFitAccuracy
			copiedConfig.Callback = func(metrics EpochMetrics) (callbackErr error) {
				copiedCallbacks = append(copiedCallbacks, metrics)
				return nil
			}

			viewConfig.FitConfig = copiedConfig
			viewConfig.FitConfig.Random = rand.New(rand.NewSource(41))
			viewConfig.FitConfig.Optimizer = viewOptimizer
			viewConfig.FitConfig.LearningRateSchedule = viewSchedule
			viewConfig.FitConfig.Loss = &viewLoss
			viewConfig.FitConfig.Callback = func(metrics EpochMetrics) (callbackErr error) {
				viewCallbacks = append(viewCallbacks, metrics)
				return nil
			}
			viewConfig.Policy = tt.policy

			if copiedHistory, err = copiedModel.Fit(dataset, copiedConfig); err != nil {
				t.Fatalf("Fit returned error: %v", err)
			}
			if viewHistory, err = viewModel.FitWithViews(
				dataset,
				viewConfig,
			); err != nil {
				t.Fatalf("FitWithViews returned error: %v", err)
			}

			requireViewFitHistoryEqual(t, viewHistory, copiedHistory)
			requireViewFitEpochsEqual(t, viewCallbacks, copiedCallbacks)
			requireViewFitRecordedTargetsEqual(
				t,
				viewLoss.targets,
				copiedLoss.targets,
			)
			requireViewFitParametersEqual(t, viewModel, copiedModel)
			if inputs, err = dataset.Inputs(); err != nil {
				t.Fatalf("Inputs returned error: %v", err)
			}
			if copiedPrediction, err = copiedModel.Predict(inputs); err != nil {
				t.Fatalf("copied Predict returned error: %v", err)
			}
			if viewPrediction, err = viewModel.Predict(inputs); err != nil {
				t.Fatalf("view Predict returned error: %v", err)
			}
			requireViewFitMatrixEqual(t, viewPrediction, copiedPrediction)
		})
	}
}

func Test_Sequential_FitWithLengthViewsMatchesCopiedFit(t *testing.T) {
	type testcase struct {
		name    string
		shuffle bool
		policy  data.ViewPolicy
	}

	var tests []testcase

	tests = []testcase{
		{name: "ordered mixed lengths and partial batch", policy: data.ViewOnly},
		{name: "shuffled copied fallback and clipping", shuffle: true, policy: data.ViewOrCopy},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				dataset          *data.SequenceDataset
				copiedModel      *Sequential
				viewModel        *Sequential
				copiedOptimizer  optimizer.Optimizer
				viewOptimizer    optimizer.Optimizer
				copiedLoss       viewFitRecordingLoss
				viewLoss         viewFitRecordingLoss
				copiedCallbacks  []EpochMetrics
				viewCallbacks    []EpochMetrics
				copiedConfig     SequenceFitConfig
				viewConfig       SequenceViewFitConfig
				copiedHistory    TrainingHistory
				viewHistory      TrainingHistory
				copiedPrediction *matrix.Matrix
				viewPrediction   *matrix.Matrix
				inputs           *matrix.Matrix
				lengths          *data.SequenceLengths
				err              error
			)

			dataset = viewFitSequenceDataset(t)
			copiedModel = viewFitSequenceModel(t)
			viewModel = viewFitSequenceModel(t)
			copiedOptimizer = viewFitClippedOptimizer(t)
			viewOptimizer = viewFitClippedOptimizer(t)

			copiedConfig.Epochs = 3
			copiedConfig.BatchSize = 2
			copiedConfig.Shuffle = tt.shuffle
			copiedConfig.Random = rand.New(rand.NewSource(43))
			copiedConfig.Optimizer = copiedOptimizer
			copiedConfig.Loss = &copiedLoss
			copiedConfig.ValidationData = dataset
			copiedConfig.Accuracy = viewFitAccuracy
			copiedConfig.Callback = func(metrics EpochMetrics) (callbackErr error) {
				copiedCallbacks = append(copiedCallbacks, metrics)
				return nil
			}

			viewConfig.SequenceFitConfig = copiedConfig
			viewConfig.SequenceFitConfig.Random = rand.New(rand.NewSource(43))
			viewConfig.SequenceFitConfig.Optimizer = viewOptimizer
			viewConfig.SequenceFitConfig.Loss = &viewLoss
			viewConfig.SequenceFitConfig.Callback = func(
				metrics EpochMetrics,
			) (callbackErr error) {
				viewCallbacks = append(viewCallbacks, metrics)
				return nil
			}
			viewConfig.Policy = tt.policy

			if copiedHistory, err = copiedModel.FitWithLengths(
				dataset,
				copiedConfig,
			); err != nil {
				t.Fatalf("FitWithLengths returned error: %v", err)
			}
			if viewHistory, err = viewModel.FitWithLengthViews(
				dataset,
				viewConfig,
			); err != nil {
				t.Fatalf("FitWithLengthViews returned error: %v", err)
			}

			requireViewFitHistoryEqual(t, viewHistory, copiedHistory)
			requireViewFitEpochsEqual(t, viewCallbacks, copiedCallbacks)
			requireViewFitRecordedTargetsEqual(
				t,
				viewLoss.targets,
				copiedLoss.targets,
			)
			requireViewFitParametersEqual(t, viewModel, copiedModel)
			if inputs, err = dataset.Inputs(); err != nil {
				t.Fatalf("Inputs returned error: %v", err)
			}
			if lengths, err = dataset.Lengths(); err != nil {
				t.Fatalf("Lengths returned error: %v", err)
			}
			if copiedPrediction, err = copiedModel.PredictWithLengths(
				inputs,
				lengths,
			); err != nil {
				t.Fatalf("copied PredictWithLengths returned error: %v", err)
			}
			if viewPrediction, err = viewModel.PredictWithLengths(
				inputs,
				lengths,
			); err != nil {
				t.Fatalf("view PredictWithLengths returned error: %v", err)
			}
			requireViewFitMatrixEqual(t, viewPrediction, copiedPrediction)
		})
	}
}

func Test_Sequential_ViewFitsValidateBeforeSideEffects(t *testing.T) {
	t.Run("ordinary strict shuffle", func(t *testing.T) {
		var (
			randomSource  *rand.Rand
			control       *rand.Rand
			schedule      viewFitCountingSchedule
			optimizerRule viewFitCountingOptimizer
			callbacks     int
			network       *Sequential
			dataset       *data.Dataset
			err           error
		)

		randomSource = rand.New(rand.NewSource(47))
		control = rand.New(rand.NewSource(47))
		network = viewFitOrdinaryModel(t)
		dataset = viewFitOrdinaryDataset(t)
		_, err = network.FitWithViews(dataset, ViewFitConfig{
			FitConfig: FitConfig{
				Epochs:               1,
				BatchSize:            2,
				Shuffle:              true,
				Random:               randomSource,
				Optimizer:            &optimizerRule,
				LearningRateSchedule: &schedule,
				Loss:                 loss.MeanSquaredError{},
				Callback: func(EpochMetrics) (callbackErr error) {
					callbacks++
					return nil
				},
			},
			Policy: data.ViewOnly,
		})
		if err == nil || !strings.Contains(err.Error(), "shuffle requires ViewOrCopy") {
			t.Fatalf("FitWithViews error = %v, want strict shuffle error", err)
		}
		if randomSource.Int63() != control.Int63() {
			t.Fatal("strict shuffle consumed random state")
		}
		if schedule.calls != 0 || optimizerRule.updateCalls != 0 || callbacks != 0 {
			t.Fatalf(
				"side effects = schedule:%d optimizer:%d callbacks:%d, want zero",
				schedule.calls,
				optimizerRule.updateCalls,
				callbacks,
			)
		}
	})

	t.Run("length-view strict shuffle", func(t *testing.T) {
		var (
			randomSource  *rand.Rand
			control       *rand.Rand
			schedule      viewFitCountingSchedule
			optimizerRule viewFitCountingOptimizer
			network       *Sequential
			dataset       *data.SequenceDataset
			err           error
		)

		randomSource = rand.New(rand.NewSource(53))
		control = rand.New(rand.NewSource(53))
		network = viewFitSequenceModel(t)
		dataset = viewFitSequenceDataset(t)
		_, err = network.FitWithLengthViews(dataset, SequenceViewFitConfig{
			SequenceFitConfig: SequenceFitConfig{
				Epochs:               1,
				BatchSize:            2,
				Shuffle:              true,
				Random:               randomSource,
				Optimizer:            &optimizerRule,
				LearningRateSchedule: &schedule,
				Loss:                 loss.MeanSquaredError{},
			},
			Policy: data.ViewOnly,
		})
		if err == nil || !strings.Contains(err.Error(), "shuffle requires ViewOrCopy") {
			t.Fatalf(
				"FitWithLengthViews error = %v, want strict shuffle error",
				err,
			)
		}
		if randomSource.Int63() != control.Int63() {
			t.Fatal("strict length-view shuffle consumed random state")
		}
		if schedule.calls != 0 || optimizerRule.updateCalls != 0 {
			t.Fatalf(
				"side effects = schedule:%d optimizer:%d, want zero",
				schedule.calls,
				optimizerRule.updateCalls,
			)
		}
	})

	t.Run("custom layer shape preflight", func(t *testing.T) {
		var (
			custom  viewFitTraversalLayer
			network *Sequential
			config  ViewFitConfig
			err     error
		)

		if network, err = NewSequential(&custom); err != nil {
			t.Fatalf("NewSequential returned error: %v", err)
		}
		config = viewFitValidConfig(t)
		_, err = network.FitWithViews(viewFitOrdinaryDataset(t), config)
		if err == nil ||
			!strings.Contains(err.Error(), "does not expose dimensions") {
			t.Fatalf(
				"FitWithViews error = %v, want custom dimension error",
				err,
			)
		}
		if custom.forwardCalls != 0 {
			t.Fatalf("custom layer traversals = %d, want zero", custom.forwardCalls)
		}
	})
}

func Test_Sequential_ViewFitsRejectInvalidPreflight(t *testing.T) {
	type testcase struct {
		name      string
		run       func() error
		wantError string
	}

	var (
		dataset         *data.Dataset
		sequenceDataset *data.SequenceDataset
		tests           []testcase
	)

	dataset = viewFitOrdinaryDataset(t)
	sequenceDataset = viewFitSequenceDataset(t)
	tests = []testcase{
		{
			name: "nil ordinary model",
			run: func() (err error) {
				var network *Sequential

				_, err = network.FitWithViews(dataset, viewFitValidConfig(t))
				return err
			},
			wantError: "model: view fit graph invalid",
		},
		{
			name: "nil ordinary dataset",
			run: func() (err error) {
				_, err = viewFitOrdinaryModel(t).FitWithViews(
					nil,
					viewFitValidConfig(t),
				)
				return err
			},
			wantError: "view fit training dataset is nil",
		},
		{
			name: "zero ordinary config",
			run: func() (err error) {
				_, err = viewFitOrdinaryModel(t).FitWithViews(
					dataset,
					ViewFitConfig{},
				)
				return err
			},
			wantError: "view fit configuration invalid",
		},
		{
			name: "invalid ordinary policy",
			run: func() (err error) {
				var config ViewFitConfig

				config = viewFitValidConfig(t)
				config.Policy = data.ViewPolicy(9)
				_, err = viewFitOrdinaryModel(t).FitWithViews(dataset, config)
				return err
			},
			wantError: "view fit policy is invalid",
		},
		{
			name: "ordinary target mismatch",
			run: func() (err error) {
				var (
					inputs        *matrix.Matrix
					targets       *matrix.Matrix
					invalid       *data.Dataset
					optimizerRule *optimizer.SGD
					config        ViewFitConfig
				)

				inputs = viewFitMatrix(t, 2, 2, []float32{1, 2, 3, 4})
				targets = viewFitMatrix(t, 2, 2, []float32{1, 2, 3, 4})
				invalid, err = data.NewDataset(inputs, targets)
				if err != nil {
					return err
				}
				optimizerRule = viewFitSGD(t)
				config.FitConfig.Epochs = 1
				config.FitConfig.BatchSize = 1
				config.FitConfig.Optimizer = optimizerRule
				config.FitConfig.Loss = loss.MeanSquaredError{}
				_, err = viewFitOrdinaryModel(t).FitWithViews(invalid, config)
				return err
			},
			wantError: "model output size mismatch",
		},
		{
			name: "nil length-view model",
			run: func() (err error) {
				var network *Sequential

				_, err = network.FitWithLengthViews(
					sequenceDataset,
					viewFitValidSequenceConfig(t),
				)
				return err
			},
			wantError: "model: length-view fit graph invalid",
		},
		{
			name: "nil sequence dataset",
			run: func() (err error) {
				_, err = viewFitSequenceModel(t).FitWithLengthViews(
					nil,
					viewFitValidSequenceConfig(t),
				)
				return err
			},
			wantError: "length-view fit training sequence dataset is nil",
		},
		{
			name: "zero length-view config",
			run: func() (err error) {
				_, err = viewFitSequenceModel(t).FitWithLengthViews(
					sequenceDataset,
					SequenceViewFitConfig{},
				)
				return err
			},
			wantError: "length-view fit configuration invalid",
		},
		{
			name: "invalid length-view policy",
			run: func() (err error) {
				var config SequenceViewFitConfig

				config = viewFitValidSequenceConfig(t)
				config.Policy = data.ViewPolicy(9)
				_, err = viewFitSequenceModel(t).FitWithLengthViews(
					sequenceDataset,
					config,
				)
				return err
			},
			wantError: "length-view fit policy is invalid",
		},
		{
			name: "sequence step mismatch",
			run: func() (err error) {
				var (
					inputs  *matrix.Matrix
					targets *matrix.Matrix
					lengths *data.SequenceLengths
					invalid *data.SequenceDataset
					config  SequenceViewFitConfig
				)

				inputs = viewFitMatrix(t, 2, 2, []float32{1, 2, 3, 4})
				targets = viewFitMatrix(t, 2, 1, []float32{1, 2})
				if lengths, err = data.NewSequenceLengths(
					2,
					[]int{1, 2},
				); err != nil {
					return err
				}
				if invalid, err = data.NewSequenceDataset(
					inputs,
					targets,
					lengths,
				); err != nil {
					return err
				}
				config = viewFitValidSequenceConfig(t)
				_, err = viewFitSequenceModel(t).FitWithLengthViews(
					invalid,
					config,
				)
				return err
			},
			wantError: "sequence dataset steps mismatch",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			err = tt.run()
			if err == nil || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("error = %v, want containing %q", err, tt.wantError)
			}
		})
	}
}

func Test_Sequential_ViewFitsRestoreStateAfterFailures(t *testing.T) {
	t.Run("ordinary loss", func(t *testing.T) {
		var (
			failure       error
			network       *Sequential
			optimizerRule *viewFitCountingOptimizer
			config        ViewFitConfig
			err           error
		)

		failure = errors.New("view loss failed")
		network = viewFitOrdinaryModel(t)
		if err = network.SetTraining(false); err != nil {
			t.Fatalf("SetTraining returned error: %v", err)
		}
		optimizerRule = &viewFitCountingOptimizer{}
		config.FitConfig.Epochs = 1
		config.FitConfig.BatchSize = 2
		config.FitConfig.Optimizer = optimizerRule
		config.FitConfig.Loss = &viewFitErrorLoss{valueErr: failure}
		_, err = network.FitWithViews(viewFitOrdinaryDataset(t), config)
		if !errors.Is(err, failure) {
			t.Fatalf("FitWithViews error = %v, want %v", err, failure)
		}
		if network.Training() {
			t.Fatal("Training = true, want restored false")
		}
		if optimizerRule.updateCalls != 0 {
			t.Fatalf("optimizer calls = %d, want 0", optimizerRule.updateCalls)
		}
	})

	t.Run("ordinary panic", func(t *testing.T) {
		var (
			network *Sequential
			config  ViewFitConfig
			dataset *data.Dataset
			err     error
		)

		network = viewFitOrdinaryModel(t)
		dataset = viewFitOrdinaryDataset(t)
		if err = network.SetTraining(false); err != nil {
			t.Fatalf("SetTraining returned error: %v", err)
		}
		config = viewFitValidConfig(t)
		config.FitConfig.Loss = viewFitPanicLoss{}
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("FitWithViews panic = nil, want panic")
				}
			}()
			_, _ = network.FitWithViews(dataset, config)
		}()
		if network.Training() {
			t.Fatal("Training = true after panic, want restored false")
		}
		if err = dataset.WithView(func(*data.DatasetView) (callbackErr error) {
			return nil
		}); err != nil {
			t.Fatalf("WithView after fit panic returned error: %v", err)
		}
	})

	t.Run("ordinary loss gradient", func(t *testing.T) {
		var (
			failure       error
			network       *Sequential
			optimizerRule viewFitCountingOptimizer
			config        ViewFitConfig
			err           error
		)

		failure = errors.New("view loss gradient failed")
		network = viewFitOrdinaryModel(t)
		if err = network.SetTraining(false); err != nil {
			t.Fatalf("SetTraining returned error: %v", err)
		}
		config.FitConfig.Epochs = 1
		config.FitConfig.BatchSize = 2
		config.FitConfig.Optimizer = &optimizerRule
		config.FitConfig.Loss = &viewFitErrorLoss{gradientErr: failure}
		_, err = network.FitWithViews(viewFitOrdinaryDataset(t), config)
		if !errors.Is(err, failure) {
			t.Fatalf("FitWithViews error = %v, want %v", err, failure)
		}
		if network.Training() {
			t.Fatal("Training = true, want restored false")
		}
		if optimizerRule.updateCalls != 0 {
			t.Fatalf("optimizer calls = %d, want 0", optimizerRule.updateCalls)
		}
	})

	t.Run("length-view optimizer", func(t *testing.T) {
		var (
			failure       error
			network       *Sequential
			optimizerRule viewFitCountingOptimizer
			config        SequenceViewFitConfig
			err           error
		)

		failure = errors.New("view optimizer failed")
		network = viewFitSequenceModel(t)
		if err = network.SetTraining(false); err != nil {
			t.Fatalf("SetTraining returned error: %v", err)
		}
		optimizerRule.updateErr = failure
		config.SequenceFitConfig.Epochs = 1
		config.SequenceFitConfig.BatchSize = 2
		config.SequenceFitConfig.Optimizer = &optimizerRule
		config.SequenceFitConfig.Loss = loss.MeanSquaredError{}
		_, err = network.FitWithLengthViews(
			viewFitSequenceDataset(t),
			config,
		)
		if !errors.Is(err, failure) {
			t.Fatalf("FitWithLengthViews error = %v, want %v", err, failure)
		}
		if network.Training() {
			t.Fatal("Training = true, want restored false")
		}
		if _, err = network.BackwardWithLengths(
			viewFitMatrix(t, 2, 1, []float32{1, 1}),
		); err == nil {
			t.Fatal("BackwardWithLengths error = nil, want cleared association")
		}
	})

	t.Run("length-view evaluation accuracy", func(t *testing.T) {
		var (
			failure error
			network *Sequential
			config  SequenceViewFitConfig
			err     error
		)

		failure = errors.New("view accuracy failed")
		network = viewFitSequenceModel(t)
		if err = network.SetTraining(false); err != nil {
			t.Fatalf("SetTraining returned error: %v", err)
		}
		config = viewFitValidSequenceConfig(t)
		config.SequenceFitConfig.Accuracy = func(
			*matrix.Matrix,
			*matrix.Matrix,
		) (accuracy float32, accuracyErr error) {
			return 0, failure
		}
		_, err = network.FitWithLengthViews(
			viewFitSequenceDataset(t),
			config,
		)
		if !errors.Is(err, failure) {
			t.Fatalf("FitWithLengthViews error = %v, want %v", err, failure)
		}
		if network.Training() {
			t.Fatal("Training = true, want restored false")
		}
		if _, err = network.BackwardWithLengths(
			viewFitMatrix(t, 2, 1, []float32{1, 1}),
		); err == nil {
			t.Fatal("BackwardWithLengths error = nil, want cleared association")
		}
	})

	t.Run("ordinary callback partial history", func(t *testing.T) {
		var (
			failure error
			network *Sequential
			config  ViewFitConfig
			history TrainingHistory
			err     error
		)

		failure = errors.New("view callback failed")
		network = viewFitOrdinaryModel(t)
		if err = network.SetTraining(false); err != nil {
			t.Fatalf("SetTraining returned error: %v", err)
		}
		config = viewFitValidConfig(t)
		config.FitConfig.Epochs = 3
		config.FitConfig.Callback = func(EpochMetrics) (callbackErr error) {
			return failure
		}
		history, err = network.FitWithViews(
			viewFitOrdinaryDataset(t),
			config,
		)
		if !errors.Is(err, failure) {
			t.Fatalf("FitWithViews error = %v, want %v", err, failure)
		}
		if len(history.Epochs) != 1 {
			t.Fatalf("history epochs = %d, want 1", len(history.Epochs))
		}
		if network.Training() {
			t.Fatal("Training = true, want restored false")
		}
	})
}

func Test_Sequential_FitWithViewsPreservesEarlyStopping(t *testing.T) {
	var (
		network       *Sequential
		earlyStopping *EarlyStopping
		optimizerRule viewFitCountingOptimizer
		config        ViewFitConfig
		callbacks     int
		history       TrainingHistory
		err           error
	)

	network = viewFitOrdinaryModel(t)
	if earlyStopping, err = NewEarlyStopping(1, 0); err != nil {
		t.Fatalf("NewEarlyStopping returned error: %v", err)
	}
	config.FitConfig.Epochs = 5
	config.FitConfig.BatchSize = 2
	config.FitConfig.Optimizer = &optimizerRule
	config.FitConfig.EarlyStopping = earlyStopping
	config.FitConfig.Loss = loss.MeanSquaredError{}
	config.FitConfig.Callback = func(EpochMetrics) (callbackErr error) {
		callbacks++
		return nil
	}
	if history, err = network.FitWithViews(
		viewFitOrdinaryDataset(t),
		config,
	); err != nil {
		t.Fatalf("FitWithViews returned error: %v", err)
	}
	if len(history.Epochs) != 2 || callbacks != 2 {
		t.Fatalf(
			"completed epochs = history:%d callbacks:%d, want 2",
			len(history.Epochs),
			callbacks,
		)
	}
}

func Test_Sequential_ViewFitsSupportDistinctConcurrentObjects(t *testing.T) {
	var (
		waitGroup sync.WaitGroup
		errorsOut [2]error
		index     int
	)

	waitGroup.Add(len(errorsOut))
	for index = range errorsOut {
		go func(resultIndex int) {
			defer waitGroup.Done()

			var (
				config ViewFitConfig
				err    error
			)

			config = viewFitValidConfig(t)
			_, err = viewFitOrdinaryModel(t).FitWithViews(
				viewFitOrdinaryDataset(t),
				config,
			)
			errorsOut[resultIndex] = err
		}(index)
	}
	waitGroup.Wait()
	for index = range errorsOut {
		if errorsOut[index] != nil {
			t.Fatalf("fit %d returned error: %v", index, errorsOut[index])
		}
	}
}

func Test_Sequential_ViewFitStateIsNotSerialized(t *testing.T) {
	var (
		network *Sequential
		config  ViewFitConfig
		before  bytes.Buffer
		after   bytes.Buffer
		err     error
	)

	network = viewFitOrdinaryModel(t)
	if err = network.Save(&before); err != nil {
		t.Fatalf("Save before returned error: %v", err)
	}
	config = viewFitValidConfig(t)
	if _, err = network.FitWithViews(viewFitOrdinaryDataset(t), config); err != nil {
		t.Fatalf("FitWithViews returned error: %v", err)
	}
	if err = network.Save(&after); err != nil {
		t.Fatalf("Save after returned error: %v", err)
	}
	if strings.Contains(after.String(), "view") ||
		strings.Contains(after.String(), "dataset") ||
		strings.Contains(after.String(), "callback") ||
		strings.Contains(after.String(), "policy") {
		t.Fatalf("serialized fit runtime state: %s", after.String())
	}
	if !strings.Contains(before.String(), `"version": 1`) ||
		!strings.Contains(after.String(), `"version": 1`) {
		t.Fatal("serialization version changed from 1")
	}
}

type viewFitRecordingLoss struct {
	targets [][]float32
}

func (l *viewFitRecordingLoss) Value(
	predictions,
	targets *matrix.Matrix,
) (value float32, err error) {
	var values []float32

	if values, err = targets.Values(); err != nil {
		return 0, err
	}
	l.targets = append(l.targets, values)
	value, err = (loss.MeanSquaredError{}).Value(predictions, targets)
	return value, err
}

func (l *viewFitRecordingLoss) Gradient(
	predictions,
	targets *matrix.Matrix,
) (gradient *matrix.Matrix, err error) {
	gradient, err = (loss.MeanSquaredError{}).Gradient(predictions, targets)
	return gradient, err
}

type viewFitCountingSchedule struct {
	calls int
}

func (s *viewFitCountingSchedule) LearningRate(
	int,
) (learningRate float32, err error) {
	s.calls++
	return 0.01, nil
}

type viewFitCountingOptimizer struct {
	updateCalls  int
	updateErr    error
	learningRate float32
}

func (o *viewFitCountingOptimizer) Update(
	[]*optimizer.Parameter,
) (err error) {
	o.updateCalls++
	return o.updateErr
}

func (o *viewFitCountingOptimizer) LearningRate() (learningRate float32) {
	return o.learningRate
}

func (o *viewFitCountingOptimizer) SetLearningRate(
	learningRate float32,
) (err error) {
	o.learningRate = learningRate
	return nil
}

type viewFitErrorLoss struct {
	valueErr    error
	gradientErr error
}

type viewFitPanicLoss struct{}

type viewFitTraversalLayer struct {
	forwardCalls int
}

func (l *viewFitTraversalLayer) Forward(
	input *matrix.Matrix,
) (output *matrix.Matrix, err error) {
	l.forwardCalls++
	return input, nil
}

func (*viewFitTraversalLayer) Backward(
	outputGradient *matrix.Matrix,
) (inputGradient *matrix.Matrix, err error) {
	return outputGradient, nil
}

func (viewFitPanicLoss) Value(
	*matrix.Matrix,
	*matrix.Matrix,
) (value float32, err error) {
	panic("view fit loss panic")
}

func (viewFitPanicLoss) Gradient(
	*matrix.Matrix,
	*matrix.Matrix,
) (gradient *matrix.Matrix, err error) {
	panic("view fit loss gradient panic")
}

func (l *viewFitErrorLoss) Value(
	*matrix.Matrix,
	*matrix.Matrix,
) (value float32, err error) {
	return 0, l.valueErr
}

func (l *viewFitErrorLoss) Gradient(
	*matrix.Matrix,
	*matrix.Matrix,
) (gradient *matrix.Matrix, err error) {
	return nil, l.gradientErr
}

func viewFitOrdinaryDataset(tb testing.TB) (dataset *data.Dataset) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		err     error
	)

	tb.Helper()
	inputs = viewFitMatrix(tb, 5, 2, []float32{
		0, 0,
		1, 0,
		0, 1,
		1, 1,
		2, -1,
	})
	targets = viewFitMatrix(tb, 5, 1, []float32{0.5, 1.5, -0.5, 0.25, 2})
	if dataset, err = data.NewDataset(inputs, targets); err != nil {
		tb.Fatalf("NewDataset returned error: %v", err)
	}
	return dataset
}

func viewFitSequenceDataset(tb testing.TB) (dataset *data.SequenceDataset) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		lengths *data.SequenceLengths
		err     error
	)

	tb.Helper()
	inputs = viewFitMatrix(tb, 5, 3, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		2, 4, 6,
		3, 6, 9,
	})
	targets = viewFitMatrix(tb, 5, 1, []float32{1, 6, 8, 2, 9})
	if lengths, err = data.NewSequenceLengths(
		3,
		[]int{1, 3, 2, 1, 3},
	); err != nil {
		tb.Fatalf("NewSequenceLengths returned error: %v", err)
	}
	if dataset, err = data.NewSequenceDataset(
		inputs,
		targets,
		lengths,
	); err != nil {
		tb.Fatalf("NewSequenceDataset returned error: %v", err)
	}
	return dataset
}

func viewFitOrdinaryModel(tb testing.TB) (network *Sequential) {
	var (
		dense *layer.Dense
		err   error
	)

	tb.Helper()
	if dense, err = layer.NewDense(
		2,
		1,
		func(inputSize, outputSize int) (weights *matrix.Matrix, initErr error) {
			weights, initErr = matrix.FromSlice(
				inputSize,
				outputSize,
				[]float32{0.25, -0.5},
			)
			return weights, initErr
		},
	); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = NewSequential(dense); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	return network
}

func viewFitSequenceModel(tb testing.TB) (network *Sequential) {
	var (
		shape  layer.SequenceShape
		gather *layer.GatherLastValid
		dense  *layer.Dense
		err    error
	)

	tb.Helper()
	if shape, err = layer.NewSequenceShape(3, 1); err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if gather, err = layer.NewGatherLastValid(shape); err != nil {
		tb.Fatalf("NewGatherLastValid returned error: %v", err)
	}
	if dense, err = layer.NewDense(
		1,
		1,
		func(inputSize, outputSize int) (weights *matrix.Matrix, initErr error) {
			weights, initErr = matrix.FromSlice(
				inputSize,
				outputSize,
				[]float32{0.5},
			)
			return weights, initErr
		},
	); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = NewSequential(gather, dense); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	return network
}

func viewFitSGD(tb testing.TB) (out *optimizer.SGD) {
	var err error

	tb.Helper()
	if out, err = optimizer.NewSGD(0.03); err != nil {
		tb.Fatalf("NewSGD returned error: %v", err)
	}
	return out
}

func viewFitClippedOptimizer(tb testing.TB) (out optimizer.Optimizer) {
	var (
		base    *optimizer.SGD
		clipped *optimizer.GradientClipping
		err     error
	)

	tb.Helper()
	base = viewFitSGD(tb)
	if clipped, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{MaxNorm: 0.75},
	); err != nil {
		tb.Fatalf("NewGradientClipping returned error: %v", err)
	}
	out = clipped
	return out
}

func viewFitValidConfig(tb testing.TB) (config ViewFitConfig) {
	tb.Helper()
	config.FitConfig.Epochs = 1
	config.FitConfig.BatchSize = 2
	config.FitConfig.Optimizer = viewFitSGD(tb)
	config.FitConfig.Loss = loss.MeanSquaredError{}
	return config
}

func viewFitValidSequenceConfig(
	tb testing.TB,
) (config SequenceViewFitConfig) {
	tb.Helper()
	config.SequenceFitConfig.Epochs = 1
	config.SequenceFitConfig.BatchSize = 2
	config.SequenceFitConfig.Optimizer = viewFitSGD(tb)
	config.SequenceFitConfig.Loss = loss.MeanSquaredError{}
	return config
}

func viewFitAccuracy(
	predictions,
	targets *matrix.Matrix,
) (accuracy float32, err error) {
	var lossValue float32

	if lossValue, err = (loss.MeanSquaredError{}).Value(
		predictions,
		targets,
	); err != nil {
		return 0, err
	}
	accuracy = 1 / (1 + lossValue)
	return accuracy, nil
}

func viewFitMatrix(
	tb testing.TB,
	rows,
	cols int,
	values []float32,
) (out *matrix.Matrix) {
	var err error

	tb.Helper()
	if out, err = matrix.FromSlice(rows, cols, values); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}
	return out
}

func requireViewFitHistoryEqual(
	tb testing.TB,
	got,
	want TrainingHistory,
) {
	tb.Helper()
	requireViewFitEpochsEqual(tb, got.Epochs, want.Epochs)
}

func requireViewFitEpochsEqual(
	tb testing.TB,
	got,
	want []EpochMetrics,
) {
	var index int

	tb.Helper()
	if len(got) != len(want) {
		tb.Fatalf("epoch count = %d, want %d", len(got), len(want))
	}
	for index = range want {
		if got[index].Epoch != want[index].Epoch ||
			got[index].HasValidationLoss != want[index].HasValidationLoss ||
			got[index].HasAccuracy != want[index].HasAccuracy ||
			got[index].HasValidationAccuracy != want[index].HasValidationAccuracy {
			tb.Fatalf("epoch %d metadata = %#v, want %#v", index, got[index], want[index])
		}
		requireViewFitFloatEqual(tb, got[index].Loss, want[index].Loss)
		requireViewFitFloatEqual(
			tb,
			got[index].ValidationLoss,
			want[index].ValidationLoss,
		)
		requireViewFitFloatEqual(tb, got[index].Accuracy, want[index].Accuracy)
		requireViewFitFloatEqual(
			tb,
			got[index].ValidationAccuracy,
			want[index].ValidationAccuracy,
		)
	}
}

func requireViewFitRecordedTargetsEqual(
	tb testing.TB,
	got,
	want [][]float32,
) {
	var (
		call  int
		index int
	)

	tb.Helper()
	if len(got) != len(want) {
		tb.Fatalf("loss value calls = %d, want %d", len(got), len(want))
	}
	for call = range want {
		if len(got[call]) != len(want[call]) {
			tb.Fatalf(
				"loss call %d target count = %d, want %d",
				call,
				len(got[call]),
				len(want[call]),
			)
		}
		for index = range want[call] {
			requireViewFitFloatEqual(tb, got[call][index], want[call][index])
		}
	}
}

func requireViewFitParametersEqual(
	tb testing.TB,
	got,
	want *Sequential,
) {
	var (
		gotParameters  []*optimizer.Parameter
		wantParameters []*optimizer.Parameter
		index          int
	)

	tb.Helper()
	gotParameters = got.Parameters()
	wantParameters = want.Parameters()
	if len(gotParameters) != len(wantParameters) {
		tb.Fatalf(
			"parameter count = %d, want %d",
			len(gotParameters),
			len(wantParameters),
		)
	}
	for index = range wantParameters {
		requireViewFitMatrixEqual(
			tb,
			gotParameters[index].Values(),
			wantParameters[index].Values(),
		)
		requireViewFitMatrixEqual(
			tb,
			gotParameters[index].Gradient(),
			wantParameters[index].Gradient(),
		)
	}
}

func requireViewFitMatrixEqual(
	tb testing.TB,
	got,
	want *matrix.Matrix,
) {
	var (
		gotValues  []float32
		wantValues []float32
		index      int
		err        error
	)

	tb.Helper()
	if got.Rows() != want.Rows() || got.Cols() != want.Cols() {
		tb.Fatalf(
			"matrix shape = %dx%d, want %dx%d",
			got.Rows(),
			got.Cols(),
			want.Rows(),
			want.Cols(),
		)
	}
	if gotValues, err = got.Values(); err != nil {
		tb.Fatalf("got Values returned error: %v", err)
	}
	if wantValues, err = want.Values(); err != nil {
		tb.Fatalf("want Values returned error: %v", err)
	}
	for index = range wantValues {
		requireViewFitFloatEqual(tb, gotValues[index], wantValues[index])
	}
}

func requireViewFitFloatEqual(tb testing.TB, got, want float32) {
	var difference float32

	tb.Helper()
	difference = got - want
	if difference < 0 {
		difference = -difference
	}
	if difference > viewFitTestTolerance {
		tb.Fatalf("value = %g, want %g", got, want)
	}
}
