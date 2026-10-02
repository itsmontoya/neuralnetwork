package model_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

var benchmarkOCRResult *matrix.Matrix

func Benchmark_SequentialOCRCandidateForward(b *testing.B) {
	var batchSizes []int
	batchSizes = []int{1, 8}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("Batch%d", batchSize), func(b *testing.B) {
			benchmarkSequentialOCRCandidateForward(b, batchSize)
		})
	}
}

func Benchmark_SequentialOCRCandidateTrainBatch(b *testing.B) {
	var batchSizes []int
	batchSizes = []int{1, 8}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("Batch%d", batchSize), func(b *testing.B) {
			benchmarkSequentialOCRCandidateTrainBatch(b, batchSize)
		})
	}
}

func benchmarkSequentialOCRCandidateForward(b *testing.B, batchSize int) {
	var (
		network *model.Sequential
		inputs  *matrix.Matrix
		output  *matrix.Matrix
		err     error
		index   int
	)

	network = benchmarkOCRCandidateModel(b)
	inputs, _ = benchmarkSyntheticMatrices(b, batchSize, 96*256, 163)
	if output, err = network.Predict(inputs); err != nil {
		b.Fatalf("warm-up Predict returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if output, err = network.Predict(inputs); err != nil {
			b.Fatalf("Predict returned error: %v", err)
		}
	}

	benchmarkOCRResult = output
}

func benchmarkSequentialOCRCandidateTrainBatch(b *testing.B, batchSize int) {
	var (
		network       *model.Sequential
		optimizerRule *optimizer.SGD
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		metrics       model.TrainMetrics
		err           error
		index         int
	)

	network = benchmarkOCRCandidateModel(b)
	inputs, targets = benchmarkSyntheticMatrices(b, batchSize, 96*256, 163)
	if optimizerRule, err = optimizer.NewSGD(0.01); err != nil {
		b.Fatalf("NewSGD returned error: %v", err)
	}
	if _, err = network.TrainBatch(inputs, targets, loss.MeanSquaredError{}, optimizerRule); err != nil {
		b.Fatalf("warm-up TrainBatch returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if metrics, err = network.TrainBatch(inputs, targets, loss.MeanSquaredError{}, optimizerRule); err != nil {
			b.Fatalf("TrainBatch returned error: %v", err)
		}
	}

	benchmarkTrainMetrics = metrics
}

func benchmarkOCRCandidateModel(tb testing.TB) (network *model.Sequential) {
	var (
		random     *rand.Rand
		inputShape layer.SpatialShape
		first      *layer.Conv2D
		firstReLU  *layer.Activation
		firstPool  *layer.MaxPool2D
		second     *layer.Conv2D
		secondReLU *layer.Activation
		secondPool *layer.MaxPool2D
		third      *layer.Conv2D
		thirdReLU  *layer.Activation
		thirdPool  *layer.MaxPool2D
		flatten    *layer.Flatten
		output     *layer.Dense
		err        error
	)

	tb.Helper()
	random = rand.New(rand.NewSource(37))
	if inputShape, err = layer.NewSpatialShape(1, 96, 256); err != nil {
		tb.Fatalf("NewSpatialShape returned error: %v", err)
	}
	first = benchmarkOCRConvolution(tb, inputShape, 16, random)
	firstReLU = benchmarkOCRReLU(tb)
	firstPool = benchmarkOCRPooling(tb, first.OutputShape())
	second = benchmarkOCRConvolution(tb, firstPool.OutputShape(), 32, random)
	secondReLU = benchmarkOCRReLU(tb)
	secondPool = benchmarkOCRPooling(tb, second.OutputShape())
	third = benchmarkOCRConvolution(tb, secondPool.OutputShape(), 64, random)
	thirdReLU = benchmarkOCRReLU(tb)
	thirdPool = benchmarkOCRPooling(tb, third.OutputShape())
	if flatten, err = layer.NewFlatten(thirdPool.OutputShape()); err != nil {
		tb.Fatalf("NewFlatten returned error: %v", err)
	}
	if output, err = layer.NewDense(flatten.OutputSize(), 163, layer.XavierUniformWeights(random)); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = model.NewSequential(
		first,
		firstReLU,
		firstPool,
		second,
		secondReLU,
		secondPool,
		third,
		thirdReLU,
		thirdPool,
		flatten,
		output,
	); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}

	return network
}

func benchmarkOCRConvolution(
	tb testing.TB,
	inputShape layer.SpatialShape,
	outputChannels int,
	random *rand.Rand,
) (convolution *layer.Conv2D) {
	var (
		config layer.Conv2DConfig
		err    error
	)

	tb.Helper()
	if config, err = layer.NewConv2DConfig(inputShape, outputChannels, 3, 3, 1, 1, 1, 1); err != nil {
		tb.Fatalf("NewConv2DConfig returned error: %v", err)
	}
	if convolution, err = layer.NewConv2D(config, layer.HeNormalWeights(random)); err != nil {
		tb.Fatalf("NewConv2D returned error: %v", err)
	}

	return convolution
}

func benchmarkOCRReLU(tb testing.TB) (activationLayer *layer.Activation) {
	var err error

	tb.Helper()
	if activationLayer, err = layer.NewActivation(activation.ReLU{}); err != nil {
		tb.Fatalf("NewActivation returned error: %v", err)
	}

	return activationLayer
}

func benchmarkOCRPooling(tb testing.TB, inputShape layer.SpatialShape) (pooling *layer.MaxPool2D) {
	var (
		config layer.MaxPool2DConfig
		err    error
	)

	tb.Helper()
	if config, err = layer.NewMaxPool2DConfig(inputShape, 2, 2, 2, 2); err != nil {
		tb.Fatalf("NewMaxPool2DConfig returned error: %v", err)
	}
	if pooling, err = layer.NewMaxPool2D(config); err != nil {
		tb.Fatalf("NewMaxPool2D returned error: %v", err)
	}

	return pooling
}
