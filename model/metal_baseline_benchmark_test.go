package model_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

var (
	benchmarkMetalBaselineMatrix  *matrix.Matrix
	benchmarkMetalBaselineValues  []float32
	benchmarkMetalBaselineMetrics model.TrainMetrics
	benchmarkMetalBaselineHistory model.TrainingHistory
)

type metalBaselineShape struct {
	name       string
	batchSize  int
	inputSize  int
	hiddenSize int
	classCount int
}

func Benchmark_SequentialMetalBaseline(b *testing.B) {
	var tests []metalBaselineShape
	tests = []metalBaselineShape{
		{name: "SmallBelowThreshold", batchSize: 8, inputSize: 32, hiddenSize: 64, classCount: 16},
		{name: "DirectlyBelowThreshold", batchSize: 63, inputSize: 128, hiddenSize: 128, classCount: 128},
		{name: "AtThreshold", batchSize: 64, inputSize: 128, hiddenSize: 128, classCount: 128},
		{name: "LargeAboveThreshold", batchSize: 128, inputSize: 256, hiddenSize: 128, classCount: 128},
	}

	var operations []struct {
		name  string
		setup func(testing.TB, metalBaselineShape) func() error
	}
	operations = []struct {
		name  string
		setup func(testing.TB, metalBaselineShape) func() error
	}{
		{name: "Predict", setup: setupMetalBaselinePredict},
		{name: "Backward", setup: setupMetalBaselineBackward},
		{name: "TrainBatch", setup: setupMetalBaselineTrainBatch},
		{name: "Fit", setup: setupMetalBaselineFit},
	}

	var (
		operation struct {
			name  string
			setup func(testing.TB, metalBaselineShape) func() error
		}
		test metalBaselineShape
	)

	for _, operation = range operations {
		for _, test = range tests {
			b.Run(operation.name+"/"+test.name+"/ColdFirstUse", func(b *testing.B) {
				benchmarkMetalBaselineCold(b, test, operation.setup)
			})
			b.Run(operation.name+"/"+test.name+"/Warmed", func(b *testing.B) {
				benchmarkMetalBaselineWarmed(b, test, operation.setup)
			})
		}
	}
}

func Benchmark_SequentialMetalDispatch(b *testing.B) {
	var shape metalBaselineShape

	shape.name = "ReadyThreshold"
	shape.batchSize = 256
	shape.inputSize = 128
	shape.hiddenSize = 128
	shape.classCount = 128
	b.Run("Predict/"+shape.name+"/ColdFirstUse", func(b *testing.B) {
		benchmarkMetalBaselineCold(b, shape, setupMetalBaselinePredict)
	})
}

func Benchmark_SequentialResidentPredict(b *testing.B) {
	var tests []metalBaselineShape
	tests = []metalBaselineShape{
		{name: "Small", batchSize: 16, inputSize: 32, hiddenSize: 64, classCount: 10},
		{name: "Uneven", batchSize: 127, inputSize: 257, hiddenSize: 263, classCount: 19},
		{name: "Large", batchSize: 256, inputSize: 512, hiddenSize: 512, classCount: 64},
	}

	var test metalBaselineShape
	for _, test = range tests {
		b.Run(test.name+"/ColdFirstUse", func(b *testing.B) {
			benchmarkResidentPredictCold(b, test)
		})
		b.Run(test.name+"/Warmed", func(b *testing.B) {
			benchmarkResidentPredictWarmed(b, test)
		})
	}
}

func Benchmark_SequentialResidentPredictObserved(b *testing.B) {
	var tests []metalBaselineShape
	tests = []metalBaselineShape{
		{name: "Large", batchSize: 256, inputSize: 512, hiddenSize: 512, classCount: 64},
		{name: "WarmThreshold", batchSize: 256, inputSize: 128, hiddenSize: 128, classCount: 128},
		{name: "ObservedBelowThreshold", batchSize: 64, inputSize: 128, hiddenSize: 128, classCount: 16},
		{name: "Small", batchSize: 16, inputSize: 32, hiddenSize: 64, classCount: 10},
	}

	var test metalBaselineShape
	for _, test = range tests {
		b.Run(test.name+"/Warmed", func(b *testing.B) {
			benchmarkResidentPredictObservedWarmed(b, test)
		})
	}
}

func Benchmark_SequentialResidentBackward(b *testing.B) {
	var tests []metalBaselineShape
	tests = []metalBaselineShape{
		{name: "Small", batchSize: 16, inputSize: 32, hiddenSize: 64, classCount: 10},
		{name: "Uneven", batchSize: 127, inputSize: 257, hiddenSize: 263, classCount: 19},
		{name: "Large", batchSize: 256, inputSize: 512, hiddenSize: 512, classCount: 64},
	}

	var test metalBaselineShape
	for _, test = range tests {
		b.Run(test.name+"/ColdFirstUse", func(b *testing.B) {
			benchmarkResidentBackwardCold(b, test)
		})
		b.Run(test.name+"/Warmed", func(b *testing.B) {
			benchmarkResidentBackwardWarmed(b, test)
		})
	}
}

func Benchmark_SequentialResidentTraining(b *testing.B) {
	var tests []metalBaselineShape
	tests = []metalBaselineShape{
		{name: "Small", batchSize: 16, inputSize: 32, hiddenSize: 64, classCount: 10},
		{name: "Large", batchSize: 256, inputSize: 512, hiddenSize: 512, classCount: 64},
	}

	var operations []struct {
		name  string
		setup func(testing.TB, metalBaselineShape) func() error
	}
	operations = []struct {
		name  string
		setup func(testing.TB, metalBaselineShape) func() error
	}{
		{name: "TrainBatch", setup: setupMetalBaselineTrainBatch},
		{name: "Fit", setup: setupMetalBaselineFit},
	}

	var (
		operation struct {
			name  string
			setup func(testing.TB, metalBaselineShape) func() error
		}
		test metalBaselineShape
	)
	for _, operation = range operations {
		for _, test = range tests {
			b.Run(operation.name+"/"+test.name+"/ColdFirstUse", func(b *testing.B) {
				benchmarkResidentTrainingCold(b, test, operation.setup)
			})
			b.Run(operation.name+"/"+test.name+"/Warmed", func(b *testing.B) {
				benchmarkResidentTrainingWarmed(b, test, operation.setup)
			})
		}
	}
}

func benchmarkResidentTrainingCold(
	b *testing.B,
	shape metalBaselineShape,
	setup func(testing.TB, metalBaselineShape) func() error,
) {
	var (
		run   func() error
		err   error
		index int
	)

	beginResidentTrainingMetrics()
	defer endResidentTrainingMetrics(b)
	b.ReportAllocs()
	for index = 0; index < b.N; index++ {
		b.StopTimer()
		run = setup(b, shape)
		b.StartTimer()
		if err = run(); err != nil {
			b.Fatalf("cold training returned error: %v", err)
		}
	}
	b.StopTimer()
}

func benchmarkResidentTrainingWarmed(
	b *testing.B,
	shape metalBaselineShape,
	setup func(testing.TB, metalBaselineShape) func() error,
) {
	var (
		run   func() error
		err   error
		index int
	)

	run = setup(b, shape)
	if err = run(); err != nil {
		b.Fatalf("warm-up returned error: %v", err)
	}

	beginResidentTrainingMetrics()
	defer endResidentTrainingMetrics(b)
	b.ReportAllocs()
	b.ResetTimer()
	for index = 0; index < b.N; index++ {
		if err = run(); err != nil {
			b.Fatalf("warmed training returned error: %v", err)
		}
	}
	b.StopTimer()
}

func benchmarkResidentBackwardCold(b *testing.B, shape metalBaselineShape) {
	var (
		run   func() error
		err   error
		index int
	)

	beginResidentBackwardMetrics()
	defer endResidentBackwardMetrics(b)
	b.ReportAllocs()
	for index = 0; index < b.N; index++ {
		b.StopTimer()
		run = setupMetalBaselineBackward(b, shape)
		b.StartTimer()
		if err = run(); err != nil {
			b.Fatalf("cold backward returned error: %v", err)
		}
	}
	b.StopTimer()
}

func benchmarkResidentBackwardWarmed(b *testing.B, shape metalBaselineShape) {
	var (
		run   func() error
		err   error
		index int
	)

	run = setupMetalBaselineBackward(b, shape)
	if err = run(); err != nil {
		b.Fatalf("warm-up returned error: %v", err)
	}

	beginResidentBackwardMetrics()
	defer endResidentBackwardMetrics(b)
	b.ReportAllocs()
	b.ResetTimer()
	for index = 0; index < b.N; index++ {
		if err = run(); err != nil {
			b.Fatalf("warmed backward returned error: %v", err)
		}
	}
	b.StopTimer()
}

func benchmarkResidentPredictCold(b *testing.B, shape metalBaselineShape) {
	var (
		run   func() error
		err   error
		index int
	)

	beginResidentPredictMetrics()
	defer endResidentPredictMetrics(b)
	b.ReportAllocs()
	for index = 0; index < b.N; index++ {
		b.StopTimer()
		run = setupMetalBaselinePredict(b, shape)
		b.StartTimer()
		if err = run(); err != nil {
			b.Fatalf("cold prediction returned error: %v", err)
		}
	}
	b.StopTimer()
}

func benchmarkResidentPredictWarmed(b *testing.B, shape metalBaselineShape) {
	var (
		run   func() error
		err   error
		index int
	)

	run = setupMetalBaselinePredict(b, shape)
	if err = run(); err != nil {
		b.Fatalf("warm-up returned error: %v", err)
	}

	beginResidentPredictMetrics()
	defer endResidentPredictMetrics(b)
	b.ReportAllocs()
	b.ResetTimer()
	for index = 0; index < b.N; index++ {
		if err = run(); err != nil {
			b.Fatalf("warmed prediction returned error: %v", err)
		}
	}
	b.StopTimer()
}

func benchmarkResidentPredictObservedWarmed(b *testing.B, shape metalBaselineShape) {
	var (
		run   func() error
		err   error
		index int
	)

	run = setupMetalBaselinePredict(b, shape)
	if err = run(); err != nil {
		b.Fatalf("prediction warm-up returned error: %v", err)
	}
	if benchmarkMetalBaselineValues, err = benchmarkMetalBaselineMatrix.Values(); err != nil {
		b.Fatalf("observation warm-up returned error: %v", err)
	}

	beginResidentPredictMetrics()
	defer endResidentPredictMetrics(b)
	b.ReportAllocs()
	b.ResetTimer()
	for index = 0; index < b.N; index++ {
		if err = run(); err != nil {
			b.Fatalf("warmed prediction returned error: %v", err)
		}
		if benchmarkMetalBaselineValues, err = benchmarkMetalBaselineMatrix.Values(); err != nil {
			b.Fatalf("warmed observation returned error: %v", err)
		}
	}
	b.StopTimer()
}

func benchmarkMetalBaselineCold(
	b *testing.B,
	shape metalBaselineShape,
	setup func(testing.TB, metalBaselineShape) func() error,
) {
	var (
		run   func() error
		err   error
		index int
	)

	b.ReportAllocs()
	for index = 0; index < b.N; index++ {
		b.StopTimer()
		run = setup(b, shape)
		b.StartTimer()

		if err = run(); err != nil {
			b.Fatalf("cold first use returned error: %v", err)
		}
	}
}

func benchmarkMetalBaselineWarmed(
	b *testing.B,
	shape metalBaselineShape,
	setup func(testing.TB, metalBaselineShape) func() error,
) {
	var (
		run   func() error
		err   error
		index int
	)

	run = setup(b, shape)
	if err = run(); err != nil {
		b.Fatalf("warm-up returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for index = 0; index < b.N; index++ {
		if err = run(); err != nil {
			b.Fatalf("warmed execution returned error: %v", err)
		}
	}
}

func setupMetalBaselinePredict(tb testing.TB, shape metalBaselineShape) (run func() error) {
	var (
		network *model.Sequential
		inputs  *matrix.Matrix
	)

	network = metalBaselineModel(tb, shape)
	inputs, _ = metalBaselineMatrices(tb, shape)
	run = func() (err error) {
		benchmarkMetalBaselineMatrix, err = network.Predict(inputs)
		return err
	}
	return run
}

func setupMetalBaselineBackward(tb testing.TB, shape metalBaselineShape) (run func() error) {
	var (
		network        *model.Sequential
		inputs         *matrix.Matrix
		outputGradient *matrix.Matrix
		err            error
	)

	network = metalBaselineModel(tb, shape)
	inputs, _ = metalBaselineMatrices(tb, shape)
	if _, err = network.Predict(inputs); err != nil {
		tb.Fatalf("Predict returned error: %v", err)
	}

	outputGradient = metalBaselineOutputGradient(tb, shape)
	run = func() (err error) {
		benchmarkMetalBaselineMatrix, err = network.Backward(outputGradient)
		return err
	}
	return run
}

func setupMetalBaselineTrainBatch(tb testing.TB, shape metalBaselineShape) (run func() error) {
	var (
		network       *model.Sequential
		optimizerRule *optimizer.SGD
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		err           error
	)

	network = metalBaselineModel(tb, shape)
	inputs, targets = metalBaselineMatrices(tb, shape)
	if optimizerRule, err = optimizer.NewSGD(0.000001); err != nil {
		tb.Fatalf("NewSGD returned error: %v", err)
	}

	run = func() (err error) {
		benchmarkMetalBaselineMetrics, err = network.TrainBatch(
			inputs,
			targets,
			loss.CategoricalCrossEntropy{},
			optimizerRule,
		)
		return err
	}
	return run
}

func setupMetalBaselineFit(tb testing.TB, shape metalBaselineShape) (run func() error) {
	var (
		network       *model.Sequential
		optimizerRule *optimizer.SGD
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		dataset       *data.Dataset
		config        model.FitConfig
		err           error
	)

	network = metalBaselineModel(tb, shape)
	inputs, targets = metalBaselineMatrices(tb, shape)
	if dataset, err = data.NewDataset(inputs, targets); err != nil {
		tb.Fatalf("NewDataset returned error: %v", err)
	}

	if optimizerRule, err = optimizer.NewSGD(0.000001); err != nil {
		tb.Fatalf("NewSGD returned error: %v", err)
	}

	config.Epochs = 1
	config.BatchSize = shape.batchSize
	config.Optimizer = optimizerRule
	config.Loss = loss.CategoricalCrossEntropy{}
	run = func() (err error) {
		benchmarkMetalBaselineHistory, err = network.Fit(dataset, config)
		return err
	}
	return run
}

func metalBaselineModel(tb testing.TB, shape metalBaselineShape) (network *model.Sequential) {
	var (
		random           *rand.Rand
		hidden           *layer.Dense
		hiddenActivation *layer.Activation
		output           *layer.Dense
		outputActivation *layer.Activation
		err              error
	)

	tb.Helper()

	random = rand.New(rand.NewSource(47))
	if hidden, err = layer.NewDense(shape.inputSize, shape.hiddenSize, layer.HeNormalWeights(random)); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}

	if hiddenActivation, err = layer.NewActivation(activation.ReLU{}); err != nil {
		tb.Fatalf("NewActivation returned error: %v", err)
	}

	if output, err = layer.NewDense(shape.hiddenSize, shape.classCount, layer.XavierUniformWeights(random)); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}

	if outputActivation, err = layer.NewActivation(activation.Softmax{}); err != nil {
		tb.Fatalf("NewActivation returned error: %v", err)
	}

	if network, err = model.NewSequential(hidden, hiddenActivation, output, outputActivation); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}

	return network
}

func metalBaselineMatrices(tb testing.TB, shape metalBaselineShape) (inputs, targets *matrix.Matrix) {
	var (
		inputValues  []float32
		targetValues []float32
		row          int
		col          int
		err          error
	)

	tb.Helper()

	inputValues = make([]float32, shape.batchSize*shape.inputSize)
	for row = 0; row < shape.batchSize; row++ {
		for col = 0; col < shape.inputSize; col++ {
			inputValues[row*shape.inputSize+col] = float32((row+3)*(col+5)%29)/29 - 0.5
		}
	}

	targetValues = make([]float32, shape.batchSize*shape.classCount)
	for row = 0; row < shape.batchSize; row++ {
		targetValues[row*shape.classCount+row%shape.classCount] = 1
	}

	if inputs, err = matrix.FromSlice(shape.batchSize, shape.inputSize, inputValues); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	if targets, err = matrix.FromSlice(shape.batchSize, shape.classCount, targetValues); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return inputs, targets
}

func metalBaselineOutputGradient(tb testing.TB, shape metalBaselineShape) (gradient *matrix.Matrix) {
	var (
		values []float32
		index  int
		err    error
	)

	tb.Helper()

	values = make([]float32, shape.batchSize*shape.classCount)
	for index = range values {
		values[index] = float32(index%17)/17 - 0.5
	}

	if gradient, err = matrix.FromSlice(shape.batchSize, shape.classCount, values); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return gradient
}
