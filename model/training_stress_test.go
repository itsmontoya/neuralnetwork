package model_test

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_Sequential_ComprehensiveTrainingIsDeterministicAndDoesNotReuseStaleScratch(t *testing.T) {
	var (
		firstHistory         model.TrainingHistory
		secondHistory        model.TrainingHistory
		firstPredictions     []float32
		secondPredictions    []float32
		firstCallbackEpochs  []int
		secondCallbackEpochs []int
	)

	firstHistory, firstPredictions, firstCallbackEpochs = fitComprehensiveSeededModel(t, 73)
	secondHistory, secondPredictions, secondCallbackEpochs = fitComprehensiveSeededModel(t, 73)

	if !reflect.DeepEqual(firstHistory, secondHistory) {
		t.Fatal("fixed-seed comprehensive training histories differ")
	}

	testutil.RequireSliceAlmostEqual(t, firstPredictions, secondPredictions, epsilon)
	requireInts(t, firstCallbackEpochs, secondCallbackEpochs)
}

func fitComprehensiveSeededModel(
	tb testing.TB,
	seed int64,
) (history model.TrainingHistory, predictions []float32, callbackEpochs []int) {
	var (
		weightRandom     *rand.Rand
		hidden           *layer.Dense
		batchNorm        *layer.BatchNormalization
		hiddenActivation *layer.Activation
		dropout          *layer.Dropout
		output           *layer.Dense
		outputActivation *layer.Activation
		network          *model.Sequential
		trainingData     *data.Dataset
		validationData   *data.Dataset
		trainingInputs   *matrix.Matrix
		validationInputs *matrix.Matrix
		firstPrediction  *matrix.Matrix
		secondPrediction *matrix.Matrix
		firstValues      []float32
		secondValues     []float32
		base             *optimizer.Adam
		l1               *optimizer.L1
		l2               *optimizer.L2WeightDecay
		optimizerRule    *optimizer.Regularized
		earlyStopping    *model.EarlyStopping
		metrics          model.EpochMetrics
		err              error
	)

	tb.Helper()

	weightRandom = rand.New(rand.NewSource(seed))
	if hidden, err = layer.NewDense(3, 6, layer.XavierUniformWeights(weightRandom)); err != nil {
		tb.Fatalf("NewDense hidden returned error: %v", err)
	}
	if batchNorm, err = layer.NewBatchNormalization(6); err != nil {
		tb.Fatalf("NewBatchNormalization returned error: %v", err)
	}
	if hiddenActivation, err = layer.NewActivation(activation.ReLU{}); err != nil {
		tb.Fatalf("NewActivation hidden returned error: %v", err)
	}
	if dropout, err = layer.NewDropout(0.25, rand.New(rand.NewSource(seed+1))); err != nil {
		tb.Fatalf("NewDropout returned error: %v", err)
	}
	if output, err = layer.NewDense(6, 1, layer.XavierUniformWeights(weightRandom)); err != nil {
		tb.Fatalf("NewDense output returned error: %v", err)
	}
	if outputActivation, err = layer.NewActivation(activation.Sigmoid{}); err != nil {
		tb.Fatalf("NewActivation output returned error: %v", err)
	}
	if network, err = model.NewSequential(
		hidden,
		batchNorm,
		hiddenActivation,
		dropout,
		output,
		outputActivation,
	); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}

	trainingData = comprehensiveDataset(tb, 17, 0)
	validationData = comprehensiveDataset(tb, 7, 3)
	if base, err = optimizer.NewAdam(0.005); err != nil {
		tb.Fatalf("NewAdam returned error: %v", err)
	}
	if l1, err = optimizer.NewL1(0.0001); err != nil {
		tb.Fatalf("NewL1 returned error: %v", err)
	}
	if l2, err = optimizer.NewL2WeightDecay(0.0001); err != nil {
		tb.Fatalf("NewL2WeightDecay returned error: %v", err)
	}
	if optimizerRule, err = optimizer.NewRegularized(base, l1, l2); err != nil {
		tb.Fatalf("NewRegularized returned error: %v", err)
	}
	if earlyStopping, err = model.NewEarlyStopping(20, math.MaxFloat32); err != nil {
		tb.Fatalf("NewEarlyStopping returned error: %v", err)
	}

	history, err = network.Fit(trainingData, model.FitConfig{
		Epochs:         200,
		BatchSize:      4,
		Shuffle:        true,
		Random:         rand.New(rand.NewSource(seed + 2)),
		Optimizer:      optimizerRule,
		EarlyStopping:  earlyStopping,
		Loss:           loss.BinaryCrossEntropy{},
		ValidationData: validationData,
		Accuracy:       metric.BinaryAccuracy{}.Value,
		Callback: func(callbackMetrics model.EpochMetrics) (err error) {
			callbackEpochs = append(callbackEpochs, callbackMetrics.Epoch)
			return nil
		},
	})
	if err != nil {
		tb.Fatalf("Fit returned error: %v", err)
	}

	requireEpochCount(tb, history, 21)
	requireInts(tb, callbackEpochs, sequence(21))
	for _, metrics = range history.Epochs {
		if !metrics.HasValidationLoss || !metrics.HasAccuracy || !metrics.HasValidationAccuracy {
			tb.Fatalf("epoch %d is missing comprehensive metrics", metrics.Epoch)
		}
	}

	if err = network.SetTraining(false); err != nil {
		tb.Fatalf("SetTraining returned error: %v", err)
	}
	if trainingInputs, err = trainingData.Inputs(); err != nil {
		tb.Fatalf("training Inputs returned error: %v", err)
	}
	if validationInputs, err = validationData.Inputs(); err != nil {
		tb.Fatalf("validation Inputs returned error: %v", err)
	}
	if firstPrediction, err = network.Predict(trainingInputs); err != nil {
		tb.Fatalf("first training Predict returned error: %v", err)
	}
	if firstValues, err = firstPrediction.Values(); err != nil {
		tb.Fatalf("first prediction Values returned error: %v", err)
	}
	if _, err = network.Predict(validationInputs); err != nil {
		tb.Fatalf("validation Predict returned error: %v", err)
	}
	if secondPrediction, err = network.Predict(trainingInputs); err != nil {
		tb.Fatalf("second training Predict returned error: %v", err)
	}
	if secondValues, err = secondPrediction.Values(); err != nil {
		tb.Fatalf("second prediction Values returned error: %v", err)
	}
	testutil.RequireSliceAlmostEqual(tb, secondValues, firstValues, epsilon)

	predictions = firstValues
	return history, predictions, callbackEpochs
}

func comprehensiveDataset(tb testing.TB, samples, offset int) (dataset *data.Dataset) {
	var (
		inputValues  []float32
		targetValues []float32
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		row          int
		column       int
		err          error
	)

	tb.Helper()

	inputValues = make([]float32, samples*3)
	targetValues = make([]float32, samples)
	for row = 0; row < samples; row++ {
		for column = 0; column < 3; column++ {
			inputValues[row*3+column] = float32(((row+offset)*(column+2)+column)%11) / 10
		}

		if inputValues[row*3]+inputValues[row*3+1] > 1 {
			targetValues[row] = 1
		}
	}

	if inputs, err = matrix.FromSlice(samples, 3, inputValues); err != nil {
		tb.Fatalf("FromSlice inputs returned error: %v", err)
	}
	if targets, err = matrix.FromSlice(samples, 1, targetValues); err != nil {
		tb.Fatalf("FromSlice targets returned error: %v", err)
	}
	if dataset, err = data.NewDataset(inputs, targets); err != nil {
		tb.Fatalf("NewDataset returned error: %v", err)
	}

	return dataset
}
