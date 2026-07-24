package model_test

import (
	"errors"
	"math/rand"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_Sequential_LengthAwarePredictionBackwardAndLifecycle(t *testing.T) {
	var (
		gather              *layer.GatherLastValid
		network             *model.Sequential
		input               *matrix.Matrix
		output              *matrix.Matrix
		outputGradient      *matrix.Matrix
		inputGradient       *matrix.Matrix
		inputGradientValues []float32
		repeatedGradient    *matrix.Matrix
		invalidGradient     *matrix.Matrix
		lengthValues        []int
		lengths             *data.SequenceLengths
		err                 error
	)

	gather = mustGatherLastValid(t, 3, 2)
	if network, err = model.NewSequential(gather); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	lengthValues = []int{1, 3}
	lengths = mustSequenceLengths(t, 3, lengthValues)
	input = mustMatrix(t, 2, 6, []float32{
		1, 2, 91, 92, 93, 94,
		3, 4, 5, 6, 7, 8,
	})
	if output, err = network.PredictWithLengths(input, lengths); err != nil {
		t.Fatalf("PredictWithLengths returned error: %v", err)
	}

	lengthValues[0] = 3
	lengthValues[1] = 1
	requireMatrixValues(t, output, []float32{1, 2, 7, 8})

	outputGradient = mustMatrix(t, 2, 2, []float32{1, -2, 3, -4})
	if inputGradient, err = network.BackwardWithLengths(outputGradient); err != nil {
		t.Fatalf("BackwardWithLengths returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float32{
		1, -2, 0, 0, 0, 0,
		0, 0, 0, 0, 3, -4,
	})
	inputGradientValues = append([]float32(nil), mustValues(t, inputGradient)...)

	if repeatedGradient, err = network.BackwardWithLengths(outputGradient); err != nil {
		t.Fatalf("repeated BackwardWithLengths returned error: %v", err)
	}
	requireMatrixValues(t, repeatedGradient, inputGradientValues)

	invalidGradient = mustMatrix(t, 1, 2, []float32{1, 1})
	if _, err = network.BackwardWithLengths(invalidGradient); err == nil {
		t.Fatal("invalid BackwardWithLengths error = nil, want shape error")
	} else if !strings.Contains(err.Error(), "shape mismatch") {
		t.Fatalf("invalid BackwardWithLengths error = %q, want shape context", err)
	}

	if _, err = network.BackwardWithLengths(outputGradient); err == nil {
		t.Fatal("BackwardWithLengths after failure error = nil, want invalidated association")
	} else if !strings.Contains(err.Error(), "before matching PredictWithLengths") {
		t.Fatalf("BackwardWithLengths after failure error = %q, want association context", err)
	}

	if _, err = network.PredictWithLengths(input, lengths); err != nil {
		t.Fatalf("second PredictWithLengths returned error: %v", err)
	}
	if _, err = network.Predict(input); err == nil {
		t.Fatal("ordinary Predict error = nil, want length-aware direction")
	} else if !strings.Contains(err.Error(), "PredictWithLengths") {
		t.Fatalf("ordinary Predict error = %q, want PredictWithLengths direction", err)
	}

	if _, err = network.BackwardWithLengths(outputGradient); err == nil {
		t.Fatal("BackwardWithLengths after ordinary Predict error = nil, want invalidated association")
	}

	if _, err = network.PredictWithLengths(input, lengths); err != nil {
		t.Fatalf("third PredictWithLengths returned error: %v", err)
	}
	if _, err = network.PredictWithLengths(input, mustSequenceLengths(t, 2, []int{1, 2})); err == nil {
		t.Fatal("mismatched PredictWithLengths error = nil, want steps error")
	}
	if _, err = network.BackwardWithLengths(outputGradient); err == nil {
		t.Fatal("BackwardWithLengths after failed prediction error = nil, want invalidated association")
	}
}

func Test_Sequential_LengthAwareOperationsRejectInvalidAssociationsBeforeUpdate(t *testing.T) {
	type testcase struct {
		name      string
		newModel  func(testing.TB) *model.Sequential
		input     *matrix.Matrix
		lengths   *data.SequenceLengths
		wantError string
	}

	var (
		input        *matrix.Matrix
		validLengths *data.SequenceLengths
		tests        []testcase
		tt           testcase
	)

	input = mustMatrix(t, 2, 3, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	validLengths = mustSequenceLengths(t, 3, []int{1, 3})
	tests = []testcase{
		{
			name: "missing selector",
			newModel: func(tb testing.TB) (network *model.Sequential) {
				var err error

				network, err = model.NewSequential(&recordingLayer{})
				if err != nil {
					tb.Fatalf("NewSequential returned error: %v", err)
				}
				return network
			},
			input:     input,
			lengths:   validLengths,
			wantError: "exactly one gather last valid layer: got=0",
		},
		{
			name: "multiple selectors",
			newModel: func(tb testing.TB) (network *model.Sequential) {
				var err error

				network, err = model.NewSequential(
					mustGatherLastValid(tb, 3, 1),
					mustGatherLastValid(tb, 3, 1),
				)
				if err != nil {
					tb.Fatalf("NewSequential returned error: %v", err)
				}
				return network
			},
			input:     input,
			lengths:   validLengths,
			wantError: "exactly one gather last valid layer: got=2",
		},
		{
			name: "length count mismatch",
			newModel: func(tb testing.TB) (network *model.Sequential) {
				var err error

				network, err = model.NewSequential(mustGatherLastValid(tb, 3, 1))
				if err != nil {
					tb.Fatalf("NewSequential returned error: %v", err)
				}
				return network
			},
			input:     input,
			lengths:   mustSequenceLengths(t, 3, []int{2}),
			wantError: "length count mismatch",
		},
		{
			name: "length steps mismatch",
			newModel: func(tb testing.TB) (network *model.Sequential) {
				var err error

				network, err = model.NewSequential(mustGatherLastValid(tb, 3, 1))
				if err != nil {
					tb.Fatalf("NewSequential returned error: %v", err)
				}
				return network
			},
			input:     input,
			lengths:   mustSequenceLengths(t, 2, []int{1, 2}),
			wantError: "length steps mismatch",
		},
	}

	for _, tt = range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				network       *model.Sequential
				optimizerRule *recordingOptimizer
				targets       *matrix.Matrix
				err           error
			)

			network = tt.newModel(t)
			optimizerRule = &recordingOptimizer{}
			targets = mustMatrix(t, tt.input.Rows(), 1, make([]float32, tt.input.Rows()))
			if _, err = network.TrainBatchWithLengths(
				tt.input,
				targets,
				tt.lengths,
				loss.MeanSquaredError{},
				optimizerRule,
			); err == nil {
				t.Fatal("TrainBatchWithLengths error = nil, want validation error")
			} else if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf(
					"TrainBatchWithLengths error = %q, want %q",
					err,
					tt.wantError,
				)
			}

			if optimizerRule.updateCalls != 0 {
				t.Fatalf("optimizer update calls = %d, want 0", optimizerRule.updateCalls)
			}
		})
	}
}

func Test_Sequential_StackedLengthAwareGradientsAndTrainingMatchValidPrefixes(t *testing.T) {
	var (
		mixed                  *model.Sequential
		training               *model.Sequential
		references             []*model.Sequential
		input                  *matrix.Matrix
		targets                *matrix.Matrix
		lengths                *data.SequenceLengths
		predictions            *matrix.Matrix
		trainingPredictions    *matrix.Matrix
		outputGradient         *matrix.Matrix
		inputGradient          *matrix.Matrix
		referenceInput         *matrix.Matrix
		referencePrediction    *matrix.Matrix
		referenceInputGradient *matrix.Matrix
		referenceGradient      *matrix.Matrix
		sgd                    *optimizer.SGD
		mixedParameters        []*optimizer.Parameter
		trainingParameters     []*optimizer.Parameter
		referenceParameters    []*optimizer.Parameter
		expectedGradients      [][]float32
		expectedUpdates        [][]float32
		predictionValues       []float32
		targetValues           []float32
		inputValues            []float32
		referenceValues        []float32
		parameterValues        []float32
		parameterGradient      []float32
		lengthValues           []int
		row                    int
		parameterIndex         int
		valueIndex             int
		learningRate           float32
		err                    error
	)

	lengthValues = []int{1, 3}
	inputValues = []float32{
		0.25, 90, 91,
		-0.5, 0.75, 1.25,
	}
	input = mustMatrix(t, 2, 3, inputValues)
	lengths = mustSequenceLengths(t, 3, lengthValues)
	mixed = mustStackedLengthAwareModel(t, 3)
	training = mustStackedLengthAwareModel(t, 3)
	references = []*model.Sequential{
		mustStackedFixedLengthModel(t, 1),
		mustStackedFixedLengthModel(t, 3),
	}

	if predictions, err = mixed.PredictWithLengths(input, lengths); err != nil {
		t.Fatalf("PredictWithLengths returned error: %v", err)
	}
	predictionValues = mustValues(t, predictions)
	outputGradient = mustMatrix(t, 2, 1, []float32{0.7, -0.4})
	if inputGradient, err = mixed.BackwardWithLengths(outputGradient); err != nil {
		t.Fatalf("BackwardWithLengths returned error: %v", err)
	}

	mixedParameters = mixed.Parameters()
	expectedGradients = make([][]float32, len(mixedParameters))
	for parameterIndex = range expectedGradients {
		expectedGradients[parameterIndex] = make(
			[]float32,
			len(mustValues(t, mixedParameters[parameterIndex].Gradient())),
		)
	}

	for row = range references {
		referenceValues = append(
			referenceValues[:0],
			inputValues[row*3:row*3+lengthValues[row]]...,
		)
		referenceInput = mustMatrix(t, 1, lengthValues[row], referenceValues)
		if referencePrediction, err = references[row].Predict(referenceInput); err != nil {
			t.Fatalf("reference row %d Predict returned error: %v", row, err)
		}
		testutil.RequireAlmostEqual(
			t,
			predictionValues[row],
			mustValues(t, referencePrediction)[0],
			epsilon,
		)

		referenceGradient = mustMatrix(t, 1, 1, []float32{mustValues(t, outputGradient)[row]})
		if referenceInputGradient, err = references[row].Backward(referenceGradient); err != nil {
			t.Fatalf("reference row %d Backward returned error: %v", row, err)
		}
		testutil.RequireSliceAlmostEqual(
			t,
			mustValues(t, referenceInputGradient),
			mustValues(t, inputGradient)[row*3:row*3+lengthValues[row]],
			epsilon,
		)

		referenceParameters = references[row].Parameters()
		for parameterIndex = range referenceParameters {
			parameterGradient = mustValues(t, referenceParameters[parameterIndex].Gradient())
			for valueIndex = range parameterGradient {
				expectedGradients[parameterIndex][valueIndex] += parameterGradient[valueIndex]
			}
		}
	}

	testutil.RequireSliceAlmostEqual(
		t,
		mustValues(t, inputGradient)[1:3],
		[]float32{0, 0},
		epsilon,
	)
	for parameterIndex = range mixedParameters {
		testutil.RequireSliceAlmostEqual(
			t,
			mustValues(t, mixedParameters[parameterIndex].Gradient()),
			expectedGradients[parameterIndex],
			epsilon,
		)
	}

	if trainingPredictions, err = training.PredictWithLengths(input, lengths); err != nil {
		t.Fatalf("training PredictWithLengths returned error: %v", err)
	}
	targetValues = []float32{0.15, -0.2}
	targets = mustMatrix(t, 2, 1, targetValues)
	outputGradient = mustMatrix(t, 2, 1, []float32{
		mustValues(t, trainingPredictions)[0] - targetValues[0],
		mustValues(t, trainingPredictions)[1] - targetValues[1],
	})

	for row = range references {
		references[row] = mustStackedFixedLengthModel(t, lengthValues[row])
		referenceValues = append(
			referenceValues[:0],
			inputValues[row*3:row*3+lengthValues[row]]...,
		)
		referenceInput = mustMatrix(t, 1, lengthValues[row], referenceValues)
		if _, err = references[row].Predict(referenceInput); err != nil {
			t.Fatalf("training reference row %d Predict returned error: %v", row, err)
		}
		referenceGradient = mustMatrix(t, 1, 1, []float32{
			mustValues(t, outputGradient)[row],
		})
		if _, err = references[row].Backward(referenceGradient); err != nil {
			t.Fatalf("training reference row %d Backward returned error: %v", row, err)
		}
	}

	trainingParameters = training.Parameters()
	expectedUpdates = make([][]float32, len(trainingParameters))
	learningRate = 0.05
	for parameterIndex = range trainingParameters {
		parameterValues = mustValues(t, trainingParameters[parameterIndex].Values())
		expectedUpdates[parameterIndex] = append([]float32(nil), parameterValues...)
		for row = range references {
			parameterGradient = mustValues(
				t,
				references[row].Parameters()[parameterIndex].Gradient(),
			)
			for valueIndex = range parameterGradient {
				expectedUpdates[parameterIndex][valueIndex] -= learningRate * parameterGradient[valueIndex]
			}
		}
	}

	if sgd, err = optimizer.NewSGD(learningRate); err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}
	if _, err = training.TrainBatchWithLengths(
		input,
		targets,
		lengths,
		loss.MeanSquaredError{},
		sgd,
	); err != nil {
		t.Fatalf("TrainBatchWithLengths returned error: %v", err)
	}

	for parameterIndex = range trainingParameters {
		testutil.RequireSliceAlmostEqual(
			t,
			mustValues(t, trainingParameters[parameterIndex].Values()),
			expectedUpdates[parameterIndex],
			epsilon,
		)
	}

	if _, err = training.BackwardWithLengths(outputGradient); err == nil {
		t.Fatal("BackwardWithLengths after TrainBatchWithLengths error = nil, want cleared association")
	}
}

func Test_Sequential_FitWithLengthsAlignsBatchesEvaluationsAndControls(t *testing.T) {
	var (
		network        *model.Sequential
		trainingData   *data.SequenceDataset
		validationData *data.SequenceDataset
		optimizerRule  *recordingOptimizer
		lossFunc       *recordingSequenceLoss
		earlyStopping  *model.EarlyStopping
		history        model.TrainingHistory
		callbackEpochs []int
		wantBatches    [][]float32
		call           int
		err            error
	)

	if network, err = model.NewSequential(mustGatherLastValid(t, 3, 1)); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	if err = network.SetTraining(false); err != nil {
		t.Fatalf("SetTraining returned error: %v", err)
	}

	trainingData = mustAlignedSequenceDataset(
		t,
		[]float32{1, 2, 3, 4, 5},
		[]int{1, 2, 3, 1, 2},
	)
	validationData = mustAlignedSequenceDataset(
		t,
		[]float32{6, 7},
		[]int{3, 1},
	)
	optimizerRule = &recordingOptimizer{}
	lossFunc = &recordingSequenceLoss{}
	if earlyStopping, err = model.NewEarlyStopping(1, 0); err != nil {
		t.Fatalf("NewEarlyStopping returned error: %v", err)
	}

	history, err = network.FitWithLengths(trainingData, model.SequenceFitConfig{
		Epochs:         10,
		BatchSize:      2,
		Shuffle:        true,
		Random:         rand.New(rand.NewSource(37)),
		Optimizer:      optimizerRule,
		EarlyStopping:  earlyStopping,
		Loss:           lossFunc,
		ValidationData: validationData,
		Accuracy: func(predictions, targets *matrix.Matrix) (accuracy float32, err error) {
			if !equalFloat32Values(mustValues(t, predictions), mustValues(t, targets)) {
				t.Fatal("accuracy received misaligned predictions and targets")
			}
			return 0.75, nil
		},
		Callback: func(metrics model.EpochMetrics) (err error) {
			callbackEpochs = append(callbackEpochs, metrics.Epoch)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("FitWithLengths returned error: %v", err)
	}

	requireEpochCount(t, history, 2)
	requireFitMetrics(t, history)
	requireInts(t, callbackEpochs, []int{1, 2})
	if network.Training() {
		t.Fatal("Training = true, want restored false")
	}
	if optimizerRule.updateCalls != 6 {
		t.Fatalf("optimizer update calls = %d, want 6", optimizerRule.updateCalls)
	}

	wantBatches = expectedShuffleBatches(5, 2, 2, 37)
	if len(lossFunc.predictions) != 10 {
		t.Fatalf("recorded loss calls = %d, want 10", len(lossFunc.predictions))
	}

	for call = range lossFunc.predictions {
		testutil.RequireSliceAlmostEqual(
			t,
			lossFunc.predictions[call],
			lossFunc.targets[call],
			epsilon,
		)
	}

	for call = 0; call < 3; call++ {
		testutil.RequireSliceAlmostEqual(
			t,
			lossFunc.predictions[call],
			wantBatches[call],
			epsilon,
		)
	}
	testutil.RequireSliceAlmostEqual(
		t,
		lossFunc.predictions[3],
		[]float32{1, 2, 3, 4, 5},
		epsilon,
	)
	testutil.RequireSliceAlmostEqual(
		t,
		lossFunc.predictions[4],
		[]float32{6, 7},
		epsilon,
	)
	for call = 0; call < 3; call++ {
		testutil.RequireSliceAlmostEqual(
			t,
			lossFunc.predictions[5+call],
			wantBatches[3+call],
			epsilon,
		)
	}

	if _, err = network.BackwardWithLengths(mustMatrix(t, 2, 1, []float32{1, 1})); err == nil {
		t.Fatal("BackwardWithLengths after FitWithLengths error = nil, want cleared association")
	}
}

func Test_Sequential_FitWithLengthsIsDeterministicWithFixedSeed(t *testing.T) {
	var (
		firstHistory      model.TrainingHistory
		secondHistory     model.TrainingHistory
		firstParameters   [][]float32
		secondParameters  [][]float32
		firstPredictions  *matrix.Matrix
		secondPredictions *matrix.Matrix
		first             *model.Sequential
		second            *model.Sequential
		dataset           *data.SequenceDataset
		lengths           *data.SequenceLengths
		inputs            *matrix.Matrix
		parameterIndex    int
		err               error
	)

	dataset = mustSequenceTrainingDataset(t)
	first, firstHistory = fitLengthAwareModel(t, dataset, 91)
	second, secondHistory = fitLengthAwareModel(t, dataset, 91)
	requireHistories(t, firstHistory, secondHistory)

	firstParameters = parameterValueCopies(t, first.Parameters())
	secondParameters = parameterValueCopies(t, second.Parameters())
	for parameterIndex = range firstParameters {
		testutil.RequireSliceAlmostEqual(
			t,
			firstParameters[parameterIndex],
			secondParameters[parameterIndex],
			epsilon,
		)
	}

	if inputs, err = dataset.Inputs(); err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}
	if lengths, err = dataset.Lengths(); err != nil {
		t.Fatalf("Lengths returned error: %v", err)
	}
	if firstPredictions, err = first.PredictWithLengths(inputs, lengths); err != nil {
		t.Fatalf("first PredictWithLengths returned error: %v", err)
	}
	if secondPredictions, err = second.PredictWithLengths(inputs, lengths); err != nil {
		t.Fatalf("second PredictWithLengths returned error: %v", err)
	}
	requireMatrixValues(t, firstPredictions, mustValues(t, secondPredictions))
}

func Test_Sequential_FitWithLengthsRejectsInvalidConfigurationAndDatasetSteps(t *testing.T) {
	type testcase struct {
		name      string
		configure func(*model.SequenceFitConfig)
		wantError string
	}

	var (
		tests []testcase
		tt    testcase
	)

	tests = []testcase{
		{
			name: "invalid epochs",
			configure: func(config *model.SequenceFitConfig) {
				config.Epochs = 0
			},
			wantError: "epochs must be positive",
		},
		{
			name: "invalid batch size",
			configure: func(config *model.SequenceFitConfig) {
				config.BatchSize = 0
			},
			wantError: "batch size must be positive",
		},
		{
			name: "nil optimizer",
			configure: func(config *model.SequenceFitConfig) {
				config.Optimizer = nil
			},
			wantError: "optimizer is nil",
		},
		{
			name: "nil loss",
			configure: func(config *model.SequenceFitConfig) {
				config.Loss = nil
			},
			wantError: "loss is nil",
		},
		{
			name: "shuffle without random",
			configure: func(config *model.SequenceFitConfig) {
				config.Shuffle = true
				config.Random = nil
			},
			wantError: "random source is nil",
		},
	}

	for _, tt = range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				network       *model.Sequential
				dataset       *data.SequenceDataset
				optimizerRule *recordingOptimizer
				config        model.SequenceFitConfig
				history       model.TrainingHistory
				err           error
			)

			if network, err = model.NewSequential(mustGatherLastValid(t, 3, 1)); err != nil {
				t.Fatalf("NewSequential returned error: %v", err)
			}
			dataset = mustAlignedSequenceDataset(t, []float32{1, 2}, []int{1, 3})
			optimizerRule = &recordingOptimizer{}
			config = model.SequenceFitConfig{
				Epochs:    1,
				BatchSize: 2,
				Optimizer: optimizerRule,
				Loss:      loss.MeanSquaredError{},
			}
			tt.configure(&config)
			if history, err = network.FitWithLengths(dataset, config); err == nil {
				t.Fatal("FitWithLengths error = nil, want configuration error")
			} else if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("FitWithLengths error = %q, want %q", err, tt.wantError)
			}
			requireEpochCount(t, history, 0)
			if optimizerRule.updateCalls != 0 {
				t.Fatalf("optimizer update calls = %d, want 0", optimizerRule.updateCalls)
			}
		})
	}

	t.Run("training steps mismatch", func(t *testing.T) {
		var (
			network       *model.Sequential
			dataset       *data.SequenceDataset
			optimizerRule *recordingOptimizer
			err           error
		)

		if network, err = model.NewSequential(mustGatherLastValid(t, 2, 1)); err != nil {
			t.Fatalf("NewSequential returned error: %v", err)
		}
		dataset = mustAlignedSequenceDataset(t, []float32{1, 2}, []int{1, 3})
		optimizerRule = &recordingOptimizer{}
		if _, err = network.FitWithLengths(dataset, model.SequenceFitConfig{
			Epochs:    1,
			BatchSize: 2,
			Optimizer: optimizerRule,
			Loss:      loss.MeanSquaredError{},
		}); err == nil {
			t.Fatal("FitWithLengths error = nil, want dataset steps error")
		} else if !strings.Contains(err.Error(), "training sequence dataset steps mismatch") {
			t.Fatalf("FitWithLengths error = %q, want training steps context", err)
		}
		if optimizerRule.updateCalls != 0 {
			t.Fatalf("optimizer update calls = %d, want 0", optimizerRule.updateCalls)
		}
	})

	t.Run("validation steps mismatch", func(t *testing.T) {
		var (
			network       *model.Sequential
			trainingData  *data.SequenceDataset
			validation    *data.SequenceDataset
			validationRaw *data.SequenceLengths
			validationIn  *matrix.Matrix
			validationOut *matrix.Matrix
			optimizerRule *recordingOptimizer
			err           error
		)

		if network, err = model.NewSequential(mustGatherLastValid(t, 3, 1)); err != nil {
			t.Fatalf("NewSequential returned error: %v", err)
		}
		trainingData = mustAlignedSequenceDataset(t, []float32{1, 2}, []int{1, 3})
		validationIn = mustMatrix(t, 1, 2, []float32{1, 2})
		validationOut = mustMatrix(t, 1, 1, []float32{1})
		validationRaw = mustSequenceLengths(t, 2, []int{1})
		if validation, err = data.NewSequenceDataset(
			validationIn,
			validationOut,
			validationRaw,
		); err != nil {
			t.Fatalf("NewSequenceDataset returned error: %v", err)
		}
		optimizerRule = &recordingOptimizer{}
		if _, err = network.FitWithLengths(trainingData, model.SequenceFitConfig{
			Epochs:         1,
			BatchSize:      2,
			Optimizer:      optimizerRule,
			Loss:           loss.MeanSquaredError{},
			ValidationData: validation,
		}); err == nil {
			t.Fatal("FitWithLengths error = nil, want validation steps error")
		} else if !strings.Contains(err.Error(), "validation sequence dataset steps mismatch") {
			t.Fatalf("FitWithLengths error = %q, want validation steps context", err)
		}
		if optimizerRule.updateCalls != 0 {
			t.Fatalf("optimizer update calls = %d, want 0", optimizerRule.updateCalls)
		}
	})
}

func Test_Sequential_LengthAwareTrainingRestoresModeAndClearsStateAfterLossError(t *testing.T) {
	var (
		lossErr       error
		network       *model.Sequential
		input         *matrix.Matrix
		targets       *matrix.Matrix
		lengths       *data.SequenceLengths
		optimizerRule *recordingOptimizer
		err           error
	)

	lossErr = errors.New("sequence loss failed")
	if network, err = model.NewSequential(mustGatherLastValid(t, 2, 1)); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	if err = network.SetTraining(false); err != nil {
		t.Fatalf("SetTraining returned error: %v", err)
	}
	input = mustMatrix(t, 1, 2, []float32{1, 2})
	targets = mustMatrix(t, 1, 1, []float32{1})
	lengths = mustSequenceLengths(t, 2, []int{1})
	optimizerRule = &recordingOptimizer{}
	if _, err = network.TrainBatchWithLengths(
		input,
		targets,
		lengths,
		&errorLoss{valueErr: lossErr},
		optimizerRule,
	); !errors.Is(err, lossErr) {
		t.Fatalf("TrainBatchWithLengths error = %v, want %v", err, lossErr)
	}
	if network.Training() {
		t.Fatal("Training = true, want restored false")
	}
	if optimizerRule.updateCalls != 0 {
		t.Fatalf("optimizer update calls = %d, want 0", optimizerRule.updateCalls)
	}
	if _, err = network.BackwardWithLengths(mustMatrix(t, 1, 1, []float32{1})); err == nil {
		t.Fatal("BackwardWithLengths after training error = nil, want cleared association")
	}
}

func Test_Sequential_FitWithLengthsAppliesLearningRateScheduleBeforeCallbacks(t *testing.T) {
	var (
		network       *model.Sequential
		dataset       *data.SequenceDataset
		optimizerRule *recordingOptimizer
		schedule      *recordingSchedule
		callbackRates []float32
		history       model.TrainingHistory
		err           error
	)

	if network, err = model.NewSequential(mustGatherLastValid(t, 3, 1)); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	dataset = mustAlignedSequenceDataset(t, []float32{1, 2}, []int{1, 3})
	optimizerRule = &recordingOptimizer{}
	schedule = &recordingSchedule{rates: []float32{0.2, 0.1, 0.05}}
	if history, err = network.FitWithLengths(dataset, model.SequenceFitConfig{
		Epochs:               3,
		BatchSize:            2,
		Optimizer:            optimizerRule,
		LearningRateSchedule: schedule,
		Loss:                 loss.MeanSquaredError{},
		Callback: func(metrics model.EpochMetrics) (err error) {
			callbackRates = append(callbackRates, optimizerRule.LearningRate())
			return nil
		},
	}); err != nil {
		t.Fatalf("FitWithLengths returned error: %v", err)
	}

	requireEpochCount(t, history, 3)
	requireInts(t, schedule.epochs, []int{1, 2, 3})
	testutil.RequireSliceAlmostEqual(
		t,
		optimizerRule.setRates,
		[]float32{0.2, 0.1, 0.05},
		epsilon,
	)
	testutil.RequireSliceAlmostEqual(
		t,
		callbackRates,
		[]float32{0.2, 0.1, 0.05},
		epsilon,
	)
}

type recordingSequenceLoss struct {
	predictions [][]float32
	targets     [][]float32
}

func (r *recordingSequenceLoss) Value(
	predictions,
	targets *matrix.Matrix,
) (value float32, err error) {
	r.predictions = append(r.predictions, append([]float32(nil), mustValuesForLoss(predictions)...))
	r.targets = append(r.targets, append([]float32(nil), mustValuesForLoss(targets)...))
	value, err = (loss.MeanSquaredError{}).Value(predictions, targets)
	return value, err
}

func (r *recordingSequenceLoss) Gradient(
	predictions,
	targets *matrix.Matrix,
) (gradient *matrix.Matrix, err error) {
	gradient, err = (loss.MeanSquaredError{}).Gradient(predictions, targets)
	return gradient, err
}

func mustValuesForLoss(value *matrix.Matrix) (values []float32) {
	var err error

	values, err = value.Values()
	if err != nil {
		panic(err)
	}
	return values
}

func mustGatherLastValid(
	tb testing.TB,
	steps,
	featureSize int,
) (gather *layer.GatherLastValid) {
	var (
		shape layer.SequenceShape
		err   error
	)

	tb.Helper()
	if shape, err = layer.NewSequenceShape(steps, featureSize); err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if gather, err = layer.NewGatherLastValid(shape); err != nil {
		tb.Fatalf("NewGatherLastValid returned error: %v", err)
	}
	return gather
}

func mustSequenceLengths(
	tb testing.TB,
	steps int,
	values []int,
) (lengths *data.SequenceLengths) {
	var err error

	tb.Helper()
	if lengths, err = data.NewSequenceLengths(steps, values); err != nil {
		tb.Fatalf("NewSequenceLengths returned error: %v", err)
	}
	return lengths
}

func mustStackedLengthAwareModel(tb testing.TB, steps int) (network *model.Sequential) {
	var (
		firstShape   layer.SequenceShape
		firstConfig  layer.SimpleRNNConfig
		first        *layer.SimpleRNN
		secondConfig layer.SimpleRNNConfig
		second       *layer.SimpleRNN
		gather       *layer.GatherLastValid
		output       *layer.Dense
		err          error
	)

	tb.Helper()
	if firstShape, err = layer.NewSequenceShape(steps, 1); err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if firstConfig, err = layer.NewSimpleRNNConfig(firstShape, 2); err != nil {
		tb.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	if first, err = layer.NewSimpleRNN(
		firstConfig,
		fixedWeights(tb, []float32{0.4, -0.3}),
		fixedWeights(tb, []float32{0.2, -0.1, 0.05, 0.25}),
	); err != nil {
		tb.Fatalf("first NewSimpleRNN returned error: %v", err)
	}
	if secondConfig, err = layer.NewSimpleRNNConfig(first.OutputShape(), 1); err != nil {
		tb.Fatalf("second NewSimpleRNNConfig returned error: %v", err)
	}
	if second, err = layer.NewSimpleRNN(
		secondConfig,
		fixedWeights(tb, []float32{0.35, -0.2}),
		fixedWeights(tb, []float32{0.15}),
	); err != nil {
		tb.Fatalf("second NewSimpleRNN returned error: %v", err)
	}
	if gather, err = layer.NewGatherLastValid(second.OutputShape()); err != nil {
		tb.Fatalf("NewGatherLastValid returned error: %v", err)
	}
	if output, err = layer.NewDense(1, 1, fixedWeights(tb, []float32{0.6})); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = model.NewSequential(first, second, gather, output); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	return network
}

func mustStackedFixedLengthModel(tb testing.TB, steps int) (network *model.Sequential) {
	var (
		firstShape   layer.SequenceShape
		firstConfig  layer.SimpleRNNConfig
		first        *layer.SimpleRNN
		secondConfig layer.SimpleRNNConfig
		second       *layer.SimpleRNN
		last         *layer.LastStep
		output       *layer.Dense
		err          error
	)

	tb.Helper()
	if firstShape, err = layer.NewSequenceShape(steps, 1); err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if firstConfig, err = layer.NewSimpleRNNConfig(firstShape, 2); err != nil {
		tb.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	if first, err = layer.NewSimpleRNN(
		firstConfig,
		fixedWeights(tb, []float32{0.4, -0.3}),
		fixedWeights(tb, []float32{0.2, -0.1, 0.05, 0.25}),
	); err != nil {
		tb.Fatalf("first NewSimpleRNN returned error: %v", err)
	}
	if secondConfig, err = layer.NewSimpleRNNConfig(first.OutputShape(), 1); err != nil {
		tb.Fatalf("second NewSimpleRNNConfig returned error: %v", err)
	}
	if second, err = layer.NewSimpleRNN(
		secondConfig,
		fixedWeights(tb, []float32{0.35, -0.2}),
		fixedWeights(tb, []float32{0.15}),
	); err != nil {
		tb.Fatalf("second NewSimpleRNN returned error: %v", err)
	}
	if last, err = layer.NewLastStep(second.OutputShape()); err != nil {
		tb.Fatalf("NewLastStep returned error: %v", err)
	}
	if output, err = layer.NewDense(1, 1, fixedWeights(tb, []float32{0.6})); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = model.NewSequential(first, second, last, output); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	return network
}

func fixedWeights(tb testing.TB, values []float32) (initializer layer.WeightInitializer) {
	tb.Helper()
	initializer = func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
		if inputSize*outputSize != len(values) {
			tb.Fatalf(
				"initializer shape %dx%d has %d values, want %d",
				inputSize,
				outputSize,
				inputSize*outputSize,
				len(values),
			)
		}
		weights, err = matrix.FromSlice(inputSize, outputSize, values)
		return weights, err
	}
	return initializer
}

func mustAlignedSequenceDataset(
	tb testing.TB,
	identifiers []float32,
	lengthValues []int,
) (dataset *data.SequenceDataset) {
	var (
		inputValues []float32
		inputs      *matrix.Matrix
		targets     *matrix.Matrix
		lengths     *data.SequenceLengths
		row         int
		step        int
		err         error
	)

	tb.Helper()
	inputValues = make([]float32, len(identifiers)*3)
	for row = range identifiers {
		for step = 0; step < 3; step++ {
			inputValues[row*3+step] = 100 + identifiers[row] + float32(step)
		}
		inputValues[row*3+lengthValues[row]-1] = identifiers[row]
	}

	inputs = mustMatrix(tb, len(identifiers), 3, inputValues)
	targets = mustMatrix(tb, len(identifiers), 1, identifiers)
	lengths = mustSequenceLengths(tb, 3, lengthValues)
	if dataset, err = data.NewSequenceDataset(inputs, targets, lengths); err != nil {
		tb.Fatalf("NewSequenceDataset returned error: %v", err)
	}
	return dataset
}

func mustSequenceTrainingDataset(tb testing.TB) (dataset *data.SequenceDataset) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		lengths *data.SequenceLengths
		err     error
	)

	tb.Helper()
	inputs = mustMatrix(tb, 5, 3, []float32{
		1, 20, 30,
		2, 3, 40,
		-1, 0, 2,
		0.5, 50, 60,
		-0.5, 1.5, 70,
	})
	targets = mustMatrix(tb, 5, 1, []float32{0.5, 1.5, -0.25, 0.75, 1})
	lengths = mustSequenceLengths(tb, 3, []int{1, 2, 3, 1, 2})
	if dataset, err = data.NewSequenceDataset(inputs, targets, lengths); err != nil {
		tb.Fatalf("NewSequenceDataset returned error: %v", err)
	}
	return dataset
}

func fitLengthAwareModel(
	tb testing.TB,
	dataset *data.SequenceDataset,
	seed int64,
) (network *model.Sequential, history model.TrainingHistory) {
	var (
		gather *layer.GatherLastValid
		output *layer.Dense
		sgd    *optimizer.SGD
		err    error
	)

	tb.Helper()
	gather = mustGatherLastValid(tb, 3, 1)
	if output, err = layer.NewDense(1, 1, fixedWeights(tb, []float32{0.2})); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = model.NewSequential(gather, output); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	if sgd, err = optimizer.NewSGD(0.03); err != nil {
		tb.Fatalf("NewSGD returned error: %v", err)
	}
	if history, err = network.FitWithLengths(dataset, model.SequenceFitConfig{
		Epochs:    4,
		BatchSize: 2,
		Shuffle:   true,
		Random:    rand.New(rand.NewSource(seed)),
		Optimizer: sgd,
		Loss:      loss.MeanSquaredError{},
	}); err != nil {
		tb.Fatalf("FitWithLengths returned error: %v", err)
	}
	return network, history
}

func parameterValueCopies(
	tb testing.TB,
	parameters []*optimizer.Parameter,
) (values [][]float32) {
	var (
		parameter      *optimizer.Parameter
		parameterValue []float32
		parameterIndex int
	)

	tb.Helper()
	values = make([][]float32, len(parameters))
	for parameterIndex, parameter = range parameters {
		parameterValue = mustValues(tb, parameter.Values())
		values[parameterIndex] = append([]float32(nil), parameterValue...)
	}
	return values
}
