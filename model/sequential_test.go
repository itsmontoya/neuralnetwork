package model_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

const epsilon = 1e-12

func Test_NewSequential_ConstructsTrainingModel(t *testing.T) {
	var (
		network *model.Sequential
		err     error
	)

	network, err = model.NewSequential()
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	if !network.Training() {
		t.Fatal("Training = false, want true")
	}
}

func Test_Sequential_AddRejectsNilLayer(t *testing.T) {
	var (
		network *model.Sequential
		err     error
	)

	network, err = model.NewSequential()
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	err = network.Add(nil)
	if err == nil {
		t.Fatal("Add error = nil, want error")
	}
}

func Test_Sequential_PredictCallsLayersInOrder(t *testing.T) {
	var (
		calls   []string
		network *model.Sequential
		input   *matrix.Matrix
		output  *matrix.Matrix
		err     error
	)

	input = mustMatrix(t, 1, 1, []float64{1})
	network, err = model.NewSequential(
		&recordingLayer{name: "first", calls: &calls, forwardDelta: 2},
		&recordingLayer{name: "second", calls: &calls, forwardDelta: 3},
	)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	output, err = network.Predict(input)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	requireStrings(t, calls, []string{"forward first", "forward second"})
	requireMatrixValues(t, output, []float64{6})
}

func Test_Sequential_BackwardCallsLayersInReverseOrder(t *testing.T) {
	var (
		calls          []string
		network        *model.Sequential
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	outputGradient = mustMatrix(t, 1, 1, []float64{1})
	network, err = model.NewSequential(
		&recordingLayer{name: "first", calls: &calls, backwardDelta: 10},
		&recordingLayer{name: "second", calls: &calls, backwardDelta: 20},
	)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	inputGradient, err = network.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	requireStrings(t, calls, []string{"backward second", "backward first"})
	requireMatrixValues(t, inputGradient, []float64{31})
}

func Test_Sequential_ParametersCollectsTrainableLayersInOrder(t *testing.T) {
	var (
		parameterOne   *optimizer.Parameter
		parameterTwo   *optimizer.Parameter
		parameterThree *optimizer.Parameter
		parameters     []*optimizer.Parameter
		network        *model.Sequential
		err            error
	)

	parameterOne = mustParameter(t, []float64{1})
	parameterTwo = mustParameter(t, []float64{2})
	parameterThree = mustParameter(t, []float64{3})

	network, err = model.NewSequential(
		&parameterLayer{parameters: []*optimizer.Parameter{parameterOne, parameterTwo}},
		&recordingLayer{},
		&parameterLayer{parameters: []*optimizer.Parameter{parameterThree}},
	)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	parameters = network.Parameters()
	if len(parameters) != 3 {
		t.Fatalf("Parameters length = %d, want 3", len(parameters))
	}

	if parameters[0] != parameterOne {
		t.Fatal("Parameters[0] did not match first parameter")
	}

	if parameters[1] != parameterTwo {
		t.Fatal("Parameters[1] did not match second parameter")
	}

	if parameters[2] != parameterThree {
		t.Fatal("Parameters[2] did not match third parameter")
	}
}

func Test_Sequential_SetTrainingPropagatesMode(t *testing.T) {
	var (
		first   *modeLayer
		second  *modeLayer
		network *model.Sequential
		err     error
	)

	first = &modeLayer{}
	second = &modeLayer{}

	network, err = model.NewSequential(first, second)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	requireBools(t, first.modes, []bool{true})
	requireBools(t, second.modes, []bool{true})

	err = network.SetTraining(false)
	if err != nil {
		t.Fatalf("SetTraining returned error: %v", err)
	}

	if network.Training() {
		t.Fatal("Training = true, want false")
	}

	requireBools(t, first.modes, []bool{true, false})
	requireBools(t, second.modes, []bool{true, false})
}

func Test_Sequential_TrainBatchUpdatesParameters(t *testing.T) {
	var (
		dense   *layer.Dense
		network *model.Sequential
		input   *matrix.Matrix
		targets *matrix.Matrix
		sgd     *optimizer.SGD
		metrics model.TrainMetrics
		err     error
	)

	dense = mustDense(t)
	network, err = model.NewSequential(dense)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	input = mustMatrix(t, 1, 2, []float64{1, 2})
	targets = mustMatrix(t, 1, 1, []float64{0})
	sgd, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	metrics, err = network.TrainBatch(input, targets, loss.MeanSquaredError{}, sgd)
	if err != nil {
		t.Fatalf("TrainBatch returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, metrics.Loss, 0.25, epsilon)
	requireMatrixValues(t, dense.Weights().Values(), []float64{1.1, -0.8})
	requireMatrixValues(t, dense.Biases().Values(), []float64{0.6})
	requireMatrixValues(t, dense.Weights().Gradient(), []float64{0, 0})
	requireMatrixValues(t, dense.Biases().Gradient(), []float64{0})
}

func Test_Sequential_FitDecreasesLossAndRecordsHistory(t *testing.T) {
	var (
		dense          *layer.Dense
		network        *model.Sequential
		dataset        *data.Dataset
		sgd            *optimizer.SGD
		history        model.TrainingHistory
		callbackEpochs []int
		err            error
	)

	dense = mustDense(t)
	network, err = model.NewSequential(dense)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	dataset = mustFitDataset(t)
	sgd, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	history, err = network.Fit(dataset, model.FitConfig{
		Epochs:         30,
		BatchSize:      4,
		Optimizer:      sgd,
		Loss:           loss.MeanSquaredError{},
		ValidationData: dataset,
		Accuracy: func(predictions, targets *matrix.Matrix) (accuracy float64, err error) {
			return 0.75, nil
		},
		Callback: func(metrics model.EpochMetrics) (err error) {
			callbackEpochs = append(callbackEpochs, metrics.Epoch)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Fit returned error: %v", err)
	}

	requireEpochCount(t, history, 30)
	if history.Epochs[len(history.Epochs)-1].Loss >= history.Epochs[0].Loss {
		t.Fatalf("last loss = %g, want less than first loss %g", history.Epochs[len(history.Epochs)-1].Loss, history.Epochs[0].Loss)
	}

	requireInts(t, callbackEpochs, sequence(30))
	requireFitMetrics(t, history)
}

func Test_Sequential_FitIsReproducibleWithFixedSeed(t *testing.T) {
	var (
		firstHistory      model.TrainingHistory
		secondHistory     model.TrainingHistory
		firstPredictions  *matrix.Matrix
		secondPredictions *matrix.Matrix
	)

	firstHistory, firstPredictions = fitSeededModel(t, 42)
	secondHistory, secondPredictions = fitSeededModel(t, 42)

	requireHistories(t, firstHistory, secondHistory)
	requireMatrixValues(t, firstPredictions, mustValues(t, secondPredictions))
}

func Test_Sequential_FitAppliesLearningRateScheduleBeforeEachEpoch(t *testing.T) {
	var (
		dense       *layer.Dense
		network     *model.Sequential
		dataset     *data.Dataset
		sgd         *optimizer.SGD
		schedule    *optimizer.StepDecay
		history     model.TrainingHistory
		epochRates  []float64
		callbackErr error
		err         error
	)

	dense = mustDense(t)
	network, err = model.NewSequential(dense)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	dataset = mustFitDataset(t)
	sgd, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	schedule, err = optimizer.NewStepDecay(0.2, 0.5, 1)
	if err != nil {
		t.Fatalf("NewStepDecay returned error: %v", err)
	}

	history, err = network.Fit(dataset, model.FitConfig{
		Epochs:               3,
		BatchSize:            4,
		Optimizer:            sgd,
		LearningRateSchedule: schedule,
		Loss:                 loss.MeanSquaredError{},
		Callback: func(metrics model.EpochMetrics) (err error) {
			epochRates = append(epochRates, sgd.LearningRate())
			return callbackErr
		},
	})
	if err != nil {
		t.Fatalf("Fit returned error: %v", err)
	}

	requireEpochCount(t, history, 3)
	testutil.RequireSliceAlmostEqual(t, epochRates, []float64{0.2, 0.1, 0.05}, epsilon)
}

func Test_Sequential_FitStopsEarlyOnTrainingLoss(t *testing.T) {
	var (
		network       *model.Sequential
		dataset       *data.Dataset
		sgd           *optimizer.SGD
		earlyStopping *model.EarlyStopping
		history       model.TrainingHistory
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		err           error
	)

	network, err = model.NewSequential(&recordingLayer{})
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	inputs = mustMatrix(t, 2, 2, []float64{
		1, 2,
		3, 4,
	})
	targets = mustMatrix(t, 2, 2, []float64{
		1, 2,
		3, 4,
	})
	dataset, err = data.NewDataset(inputs, targets)
	if err != nil {
		t.Fatalf("NewDataset returned error: %v", err)
	}

	sgd, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	earlyStopping, err = model.NewEarlyStopping(2, 0)
	if err != nil {
		t.Fatalf("NewEarlyStopping returned error: %v", err)
	}

	history, err = network.Fit(dataset, model.FitConfig{
		Epochs:        10,
		BatchSize:     2,
		Optimizer:     sgd,
		Loss:          loss.MeanSquaredError{},
		EarlyStopping: earlyStopping,
	})
	if err != nil {
		t.Fatalf("Fit returned error: %v", err)
	}

	requireEpochCount(t, history, 3)
}

func Test_Sequential_FitEarlyStoppingUsesValidationLossWhenAvailable(t *testing.T) {
	var (
		network       *model.Sequential
		trainingData  *data.Dataset
		validation    *data.Dataset
		sgd           *optimizer.SGD
		earlyStopping *model.EarlyStopping
		history       model.TrainingHistory
		lossFunc      *sequenceLoss
		err           error
	)

	network, err = model.NewSequential(&recordingLayer{})
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	trainingData = mustSequenceDataset(t)
	validation = mustSequenceDataset(t)

	sgd, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	earlyStopping, err = model.NewEarlyStopping(1, 0)
	if err != nil {
		t.Fatalf("NewEarlyStopping returned error: %v", err)
	}

	lossFunc = &sequenceLoss{
		values: []float64{
			0, 10, 1,
			0, 9, 1,
			0, 8, 1,
		},
	}
	history, err = network.Fit(trainingData, model.FitConfig{
		Epochs:         5,
		BatchSize:      2,
		Optimizer:      sgd,
		Loss:           lossFunc,
		ValidationData: validation,
		EarlyStopping:  earlyStopping,
	})
	if err != nil {
		t.Fatalf("Fit returned error: %v", err)
	}

	requireEpochCount(t, history, 2)
	testutil.RequireAlmostEqual(t, history.Epochs[0].Loss, 10, epsilon)
	testutil.RequireAlmostEqual(t, history.Epochs[1].Loss, 9, epsilon)
	testutil.RequireAlmostEqual(t, history.Epochs[0].ValidationLoss, 1, epsilon)
	testutil.RequireAlmostEqual(t, history.Epochs[1].ValidationLoss, 1, epsilon)
}

func Test_Sequential_TrainBatchUsesTrainingModeAndRestoresPreviousMode(t *testing.T) {
	var (
		dropout *layer.Dropout
		network *model.Sequential
		input   *matrix.Matrix
		targets *matrix.Matrix
		sgd     *optimizer.SGD
		metrics model.TrainMetrics
		err     error
	)

	dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	network, err = model.NewSequential(dropout)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	err = network.SetTraining(false)
	if err != nil {
		t.Fatalf("SetTraining returned error: %v", err)
	}

	input = mustMatrix(t, 1, 1, []float64{1})
	targets = mustMatrix(t, 1, 1, []float64{1})
	sgd, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	metrics, err = network.TrainBatch(input, targets, loss.MeanSquaredError{}, sgd)
	if err != nil {
		t.Fatalf("TrainBatch returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, metrics.Loss, 1, epsilon)
	if network.Training() {
		t.Fatal("Training = true, want restored false")
	}
}

func Test_Sequential_FitEvaluatesWithTrainingDisabled(t *testing.T) {
	var (
		dropout *layer.Dropout
		network *model.Sequential
		dataset *data.Dataset
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		sgd     *optimizer.SGD
		history model.TrainingHistory
		err     error
	)

	dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(2)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	network, err = model.NewSequential(dropout)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	inputs = mustMatrix(t, 2, 2, []float64{
		1, 2,
		3, 4,
	})
	targets = mustMatrix(t, 2, 2, []float64{
		1, 2,
		3, 4,
	})
	dataset, err = data.NewDataset(inputs, targets)
	if err != nil {
		t.Fatalf("NewDataset returned error: %v", err)
	}

	sgd, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	history, err = network.Fit(dataset, model.FitConfig{
		Epochs:    1,
		BatchSize: 1,
		Optimizer: sgd,
		Loss:      loss.MeanSquaredError{},
	})
	if err != nil {
		t.Fatalf("Fit returned error: %v", err)
	}

	requireEpochCount(t, history, 1)
	testutil.RequireAlmostEqual(t, history.Epochs[0].Loss, 0, epsilon)
}

func mustDense(tb testing.TB) (dense *layer.Dense) {
	var (
		biases *matrix.Matrix
		err    error
	)

	tb.Helper()

	dense, err = layer.NewDense(2, 1, func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
		weights, err = matrix.FromSlice(inputSize, outputSize, []float64{1, -1})
		return weights, err
	})
	if err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}

	biases = mustMatrix(tb, 1, 1, []float64{0.5})
	err = dense.Biases().Values().CopyFrom(biases)
	if err != nil {
		tb.Fatalf("CopyFrom returned error: %v", err)
	}

	return dense
}

func mustFitDataset(tb testing.TB) (dataset *data.Dataset) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		err     error
	)

	tb.Helper()

	inputs = mustMatrix(tb, 4, 2, []float64{
		0, 0,
		1, 0,
		0, 1,
		1, 1,
	})
	targets = mustMatrix(tb, 4, 1, []float64{
		1,
		3,
		-2,
		0,
	})

	dataset, err = data.NewDataset(inputs, targets)
	if err != nil {
		tb.Fatalf("NewDataset returned error: %v", err)
	}

	return dataset
}

func mustSequenceDataset(tb testing.TB) (dataset *data.Dataset) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		err     error
	)

	tb.Helper()

	inputs = mustMatrix(tb, 2, 1, []float64{1, 2})
	targets = mustMatrix(tb, 2, 1, []float64{1, 2})
	dataset, err = data.NewDataset(inputs, targets)
	if err != nil {
		tb.Fatalf("NewDataset returned error: %v", err)
	}

	return dataset
}

func fitSeededModel(tb testing.TB, seed int64) (history model.TrainingHistory, predictions *matrix.Matrix) {
	var (
		dense   *layer.Dense
		network *model.Sequential
		dataset *data.Dataset
		inputs  *matrix.Matrix
		sgd     *optimizer.SGD
		random  *rand.Rand
		err     error
	)

	tb.Helper()

	dense = mustDense(tb)
	network, err = model.NewSequential(dense)
	if err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}

	dataset = mustFitDataset(tb)
	sgd, err = optimizer.NewSGD(0.1)
	if err != nil {
		tb.Fatalf("NewSGD returned error: %v", err)
	}

	random = rand.New(rand.NewSource(seed))
	history, err = network.Fit(dataset, model.FitConfig{
		Epochs:    20,
		BatchSize: 2,
		Shuffle:   true,
		Random:    random,
		Optimizer: sgd,
		Loss:      loss.MeanSquaredError{},
	})
	if err != nil {
		tb.Fatalf("Fit returned error: %v", err)
	}

	inputs, err = dataset.Inputs()
	if err != nil {
		tb.Fatalf("Inputs returned error: %v", err)
	}

	predictions, err = network.Predict(inputs)
	if err != nil {
		tb.Fatalf("Predict returned error: %v", err)
	}

	return history, predictions
}

func mustMatrix(tb testing.TB, rows, cols int, values []float64) (m *matrix.Matrix) {
	var err error

	tb.Helper()

	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}

func mustParameter(tb testing.TB, values []float64) (parameter *optimizer.Parameter) {
	var (
		valueMatrix *matrix.Matrix
		err         error
	)

	tb.Helper()

	valueMatrix = mustMatrix(tb, 1, len(values), values)
	parameter, err = optimizer.NewParameter(valueMatrix)
	if err != nil {
		tb.Fatalf("NewParameter returned error: %v", err)
	}

	return parameter
}

func mustValues(tb testing.TB, m *matrix.Matrix) (values []float64) {
	var err error

	tb.Helper()

	values, err = m.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	return values
}

func requireMatrixValues(tb testing.TB, got *matrix.Matrix, want []float64) {
	var (
		values []float64
		err    error
	)

	tb.Helper()

	values, err = got.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(tb, values, want, epsilon)
}

func requireEpochCount(tb testing.TB, history model.TrainingHistory, want int) {
	tb.Helper()

	if len(history.Epochs) != want {
		tb.Fatalf("history epoch count = %d, want %d", len(history.Epochs), want)
	}
}

func requireFitMetrics(tb testing.TB, history model.TrainingHistory) {
	var metrics model.EpochMetrics

	tb.Helper()

	for _, metrics = range history.Epochs {
		if !metrics.HasValidationLoss {
			tb.Fatalf("epoch %d missing validation loss", metrics.Epoch)
		}

		if !metrics.HasAccuracy {
			tb.Fatalf("epoch %d missing accuracy", metrics.Epoch)
		}

		if !metrics.HasValidationAccuracy {
			tb.Fatalf("epoch %d missing validation accuracy", metrics.Epoch)
		}

		testutil.RequireAlmostEqual(tb, metrics.Accuracy, 0.75, epsilon)
		testutil.RequireAlmostEqual(tb, metrics.ValidationAccuracy, 0.75, epsilon)
	}
}

func requireHistories(tb testing.TB, got, want model.TrainingHistory) {
	var index int

	tb.Helper()

	if len(got.Epochs) != len(want.Epochs) {
		tb.Fatalf("history lengths differ: got %d, want %d", len(got.Epochs), len(want.Epochs))
	}

	for index = range got.Epochs {
		if got.Epochs[index].Epoch != want.Epochs[index].Epoch {
			tb.Fatalf("epoch differs at index %d: got %d, want %d", index, got.Epochs[index].Epoch, want.Epochs[index].Epoch)
		}

		testutil.RequireAlmostEqual(tb, got.Epochs[index].Loss, want.Epochs[index].Loss, epsilon)
	}
}

func requireInts(tb testing.TB, got, want []int) {
	var index int

	tb.Helper()

	if len(got) != len(want) {
		tb.Fatalf("int slice lengths differ: got %d, want %d", len(got), len(want))
	}

	for index = range got {
		if got[index] == want[index] {
			continue
		}

		tb.Fatalf("int slice differs at index %d: got %d, want %d", index, got[index], want[index])
	}
}

func sequence(count int) (values []int) {
	var index int

	values = make([]int, count)
	for index = range values {
		values[index] = index + 1
	}

	return values
}

func requireStrings(tb testing.TB, got, want []string) {
	var index int

	tb.Helper()

	if len(got) != len(want) {
		tb.Fatalf("string slice lengths differ: got %d, want %d", len(got), len(want))
	}

	for index = range got {
		if got[index] == want[index] {
			continue
		}

		tb.Fatalf("string slice differs at index %d: got %q, want %q", index, got[index], want[index])
	}
}

func requireBools(tb testing.TB, got, want []bool) {
	var index int

	tb.Helper()

	if len(got) != len(want) {
		tb.Fatalf("bool slice lengths differ: got %d, want %d", len(got), len(want))
	}

	for index = range got {
		if got[index] == want[index] {
			continue
		}

		tb.Fatalf("bool slice differs at index %d: got %t, want %t", index, got[index], want[index])
	}
}

type recordingLayer struct {
	name          string
	calls         *[]string
	forwardDelta  float64
	backwardDelta float64
}

func (r *recordingLayer) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	if r.calls != nil {
		*r.calls = append(*r.calls, "forward "+r.name)
	}

	output, err = input.AddScalar(r.forwardDelta)
	return output, err
}

func (r *recordingLayer) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	if r.calls != nil {
		*r.calls = append(*r.calls, "backward "+r.name)
	}

	inputGradient, err = outputGradient.AddScalar(r.backwardDelta)
	return inputGradient, err
}

type parameterLayer struct {
	recordingLayer
	parameters []*optimizer.Parameter
}

func (p *parameterLayer) Parameters() (parameters []*optimizer.Parameter) {
	parameters = p.parameters
	return parameters
}

type modeLayer struct {
	recordingLayer
	modes []bool
}

func (m *modeLayer) SetTraining(training bool) {
	m.modes = append(m.modes, training)
}

type sequenceLoss struct {
	values []float64
	index  int
}

func (s *sequenceLoss) Value(predictions, targets *matrix.Matrix) (value float64, err error) {
	if s.index >= len(s.values) {
		err = fmt.Errorf("sequence loss exhausted at index %d", s.index)
		return 0, err
	}

	value = s.values[s.index]
	s.index++
	return value, nil
}

func (s *sequenceLoss) Gradient(predictions, targets *matrix.Matrix) (gradient *matrix.Matrix, err error) {
	var (
		rows int
		cols int
	)

	rows, cols = predictions.Shape()
	gradient, err = matrix.New(rows, cols)
	return gradient, err
}
