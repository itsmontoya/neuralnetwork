package model

import (
	"math"
	"math/rand"
	"strconv"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

const (
	benchmarkFitDataViewsOrdinarySamples      = 4096
	benchmarkFitDataViewsOrdinaryInputs       = 256
	benchmarkFitDataViewsOrdinaryTargets      = 16
	benchmarkFitDataViewsOrdinaryBatchSize    = 256
	benchmarkFitDataViewsSequenceSamples      = 512
	benchmarkFitDataViewsSequenceValidation   = 256
	benchmarkFitDataViewsSequenceSteps        = 128
	benchmarkFitDataViewsSequenceFeatures     = 32
	benchmarkFitDataViewsSequenceInputs       = 4096
	benchmarkFitDataViewsSequenceTargets      = 8
	benchmarkFitDataViewsSequenceBatchSize    = 256
	benchmarkFitDataViewsSequencePartialBatch = 192
)

var (
	benchmarkFitDataViewsHistory  TrainingHistory
	benchmarkFitDataViewsMetrics  EpochMetrics
	benchmarkFitDataViewsLoss     float32
	benchmarkFitDataViewsChecksum float32
)

func Benchmark_FitDataViewsCopyBaseline(b *testing.B) {
	b.Run("FullTrainingEvaluation/Cold/Ordinary4096x256_4096x16", func(b *testing.B) {
		benchmarkFitDataViewsEvaluation(b, true)
	})
	b.Run("FullTrainingEvaluation/Warm/Ordinary4096x256_4096x16", func(b *testing.B) {
		benchmarkFitDataViewsEvaluation(b, false)
	})
	b.Run("FullValidationEvaluation/Cold/Ordinary4096x256_4096x16", func(b *testing.B) {
		benchmarkFitDataViewsEvaluation(b, true)
	})
	b.Run("FullValidationEvaluation/Warm/Ordinary4096x256_4096x16", func(b *testing.B) {
		benchmarkFitDataViewsEvaluation(b, false)
	})
	b.Run("CompleteEpoch/Ordered/Cold/Train4096_Validation4096_Batch256", func(b *testing.B) {
		benchmarkFitDataViewsCompleteEpoch(b, false, true)
	})
	b.Run("CompleteEpoch/Ordered/Warm/Train4096_Validation4096_Batch256", func(b *testing.B) {
		benchmarkFitDataViewsCompleteEpoch(b, false, false)
	})
	b.Run("CompleteEpoch/Shuffled/Cold/Train4096_Validation4096_Batch256", func(b *testing.B) {
		benchmarkFitDataViewsCompleteEpoch(b, true, true)
	})
	b.Run("CompleteEpoch/Shuffled/Warm/Train4096_Validation4096_Batch256", func(b *testing.B) {
		benchmarkFitDataViewsCompleteEpoch(b, true, false)
	})
}

func Benchmark_FitSequenceDataViewsCopyBaseline(b *testing.B) {
	b.Run("FullTrainingEvaluation/Cold/Sequence512x4096_512x8_Lengths512", func(b *testing.B) {
		benchmarkFitSequenceDataViewsEvaluation(
			b,
			benchmarkFitDataViewsSequenceSamples,
			true,
		)
	})
	b.Run("FullTrainingEvaluation/Warm/Sequence512x4096_512x8_Lengths512", func(b *testing.B) {
		benchmarkFitSequenceDataViewsEvaluation(
			b,
			benchmarkFitDataViewsSequenceSamples,
			false,
		)
	})
	b.Run("FullValidationEvaluation/Cold/Sequence256x4096_256x8_Lengths256", func(b *testing.B) {
		benchmarkFitSequenceDataViewsEvaluation(
			b,
			benchmarkFitDataViewsSequenceValidation,
			true,
		)
	})
	b.Run("FullValidationEvaluation/Warm/Sequence256x4096_256x8_Lengths256", func(b *testing.B) {
		benchmarkFitSequenceDataViewsEvaluation(
			b,
			benchmarkFitDataViewsSequenceValidation,
			false,
		)
	})
	b.Run("CompleteEpoch/Ordered/Cold/Train512_Validation256_Batch192", func(b *testing.B) {
		benchmarkFitSequenceDataViewsCompleteEpoch(b, false, true)
	})
	b.Run("CompleteEpoch/Ordered/Warm/Train512_Validation256_Batch192", func(b *testing.B) {
		benchmarkFitSequenceDataViewsCompleteEpoch(b, false, false)
	})
	b.Run("CompleteEpoch/Shuffled/Cold/Train512_Validation256_Batch256", func(b *testing.B) {
		benchmarkFitSequenceDataViewsCompleteEpoch(b, true, true)
	})
	b.Run("CompleteEpoch/Shuffled/Warm/Train512_Validation256_Batch256", func(b *testing.B) {
		benchmarkFitSequenceDataViewsCompleteEpoch(b, true, false)
	})
}

func benchmarkFitDataViewsEvaluation(b *testing.B, cold bool) {
	var (
		dataset      *data.Dataset
		network      *Sequential
		matrices     fitMatrixPair
		warmMatrices fitMatrixPair
		lossValue    float32
		logicalBytes int64
		err          error
		index        int
	)

	dataset = benchmarkFitDataViewsOrdinaryDataset(b)
	network = benchmarkFitDataViewsOrdinaryModel(b)
	if cold {
		if _, _, _, err = network.evaluateFitDataset(
			dataset,
			loss.MeanSquaredError{},
			nil,
			&warmMatrices,
		); err != nil {
			b.Fatalf("warm-up evaluateFitDataset returned error: %v", err)
		}
		if err = warmMatrices.release(); err != nil {
			b.Fatalf("release warm-up matrices returned error: %v", err)
		}
	} else {
		if _, _, _, err = network.evaluateFitDataset(
			dataset,
			loss.MeanSquaredError{},
			nil,
			&matrices,
		); err != nil {
			b.Fatalf("warm-up evaluateFitDataset returned error: %v", err)
		}
	}

	logicalBytes = benchmarkFitDataViewsOrdinaryBytes(
		benchmarkFitDataViewsOrdinarySamples,
	)
	b.ResetTimer()
	reportFitDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		if cold {
			matrices = fitMatrixPair{}
		}
		if lossValue, _, _, err = network.evaluateFitDataset(
			dataset,
			loss.MeanSquaredError{},
			nil,
			&matrices,
		); err != nil {
			b.Fatalf("evaluateFitDataset returned error: %v", err)
		}
		if cold {
			if err = matrices.release(); err != nil {
				b.Fatalf("release evaluation matrices returned error: %v", err)
			}
		}
	}

	b.StopTimer()
	if !cold {
		if err = matrices.release(); err != nil {
			b.Fatalf("release evaluation matrices returned error: %v", err)
		}
	}
	verifyFitDataViewsFinite(b, "ordinary evaluation loss", lossValue)
	benchmarkFitDataViewsLoss = lossValue
}

func benchmarkFitSequenceDataViewsEvaluation(
	b *testing.B,
	samples int,
	cold bool,
) {
	var (
		dataset        *data.SequenceDataset
		network        *Sequential
		selector       sequenceLengthLayer
		selectorIndex  int
		matrices       fitMatrixPair
		warmMatrices   fitMatrixPair
		lengths        []int
		warmLengths    []int
		lossValue      float32
		logicalBytes   int64
		err            error
		benchmarkIndex int
	)

	dataset = benchmarkFitDataViewsSequenceDataset(b, samples)
	network = benchmarkFitDataViewsSequenceModel(b)
	if selector, selectorIndex, err = network.lengthAwareGraph("benchmark evaluation"); err != nil {
		b.Fatalf("lengthAwareGraph returned error: %v", err)
	}
	if cold {
		if _, _, _, err = network.evaluateSequenceFitDataset(
			dataset,
			loss.MeanSquaredError{},
			nil,
			selector,
			selectorIndex,
			&warmMatrices,
			&warmLengths,
		); err != nil {
			b.Fatalf("warm-up evaluateSequenceFitDataset returned error: %v", err)
		}
		if err = warmMatrices.release(); err != nil {
			b.Fatalf("release warm-up matrices returned error: %v", err)
		}
	} else {
		if _, _, _, err = network.evaluateSequenceFitDataset(
			dataset,
			loss.MeanSquaredError{},
			nil,
			selector,
			selectorIndex,
			&matrices,
			&lengths,
		); err != nil {
			b.Fatalf("warm-up evaluateSequenceFitDataset returned error: %v", err)
		}
	}

	logicalBytes = benchmarkFitDataViewsSequenceBytes(samples)
	b.ResetTimer()
	reportFitDataViewsMetrics(b, logicalBytes, logicalBytes)

	for benchmarkIndex = 0; benchmarkIndex < b.N; benchmarkIndex++ {
		if cold {
			matrices = fitMatrixPair{}
			lengths = nil
		}
		if lossValue, _, _, err = network.evaluateSequenceFitDataset(
			dataset,
			loss.MeanSquaredError{},
			nil,
			selector,
			selectorIndex,
			&matrices,
			&lengths,
		); err != nil {
			b.Fatalf("evaluateSequenceFitDataset returned error: %v", err)
		}
		if cold {
			if err = matrices.release(); err != nil {
				b.Fatalf("release sequence evaluation matrices returned error: %v", err)
			}
		}
	}

	b.StopTimer()
	if !cold {
		if err = matrices.release(); err != nil {
			b.Fatalf("release sequence evaluation matrices returned error: %v", err)
		}
	}
	verifyFitDataViewsFinite(b, "sequence evaluation loss", lossValue)
	benchmarkFitDataViewsLoss = lossValue
}

func benchmarkFitDataViewsCompleteEpoch(
	b *testing.B,
	shuffle,
	cold bool,
) {
	var (
		trainingData   *data.Dataset
		validationData *data.Dataset
		network        *Sequential
		config         FitConfig
		history        TrainingHistory
		metrics        EpochMetrics
		scratch        fitScratch
		trainingBytes  int64
		logicalBytes   int64
		err            error
		index          int
	)

	trainingData = benchmarkFitDataViewsOrdinaryDataset(b)
	validationData = benchmarkFitDataViewsOrdinaryDataset(b)
	network = benchmarkFitDataViewsOrdinaryModel(b)
	config = benchmarkFitDataViewsConfig(b, validationData, shuffle)
	if cold {
		if _, err = network.Fit(trainingData, config); err != nil {
			b.Fatalf("warm-up Fit returned error: %v", err)
		}
	} else {
		if metrics, err = benchmarkFitDataViewsWarmEpoch(
			network,
			trainingData,
			config,
			1,
			&scratch,
		); err != nil {
			b.Fatalf("warm-up ordinary epoch returned error: %v", err)
		}
	}

	trainingBytes = benchmarkFitDataViewsOrdinaryBytes(
		benchmarkFitDataViewsOrdinarySamples,
	)
	logicalBytes = trainingBytes * 3
	b.ResetTimer()
	reportFitDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		if cold {
			history, err = network.Fit(trainingData, config)
			if err != nil {
				b.Fatalf("Fit returned error: %v", err)
			}
			continue
		}

		metrics, err = benchmarkFitDataViewsWarmEpoch(
			network,
			trainingData,
			config,
			index+2,
			&scratch,
		)
		if err != nil {
			b.Fatalf("ordinary epoch returned error: %v", err)
		}
	}

	b.StopTimer()
	if err = scratch.release(); err != nil {
		b.Fatalf("release ordinary fit scratch returned error: %v", err)
	}
	if cold {
		metrics = benchmarkFitDataViewsHistoryMetrics(b, history)
	}
	verifyFitDataViewsEpochMetrics(b, metrics)
	benchmarkFitDataViewsHistory = history
	benchmarkFitDataViewsMetrics = metrics
}

func benchmarkFitSequenceDataViewsCompleteEpoch(
	b *testing.B,
	shuffle,
	cold bool,
) {
	var (
		trainingData    *data.SequenceDataset
		validationData  *data.SequenceDataset
		network         *Sequential
		config          SequenceFitConfig
		selector        sequenceLengthLayer
		selectorIndex   int
		history         TrainingHistory
		metrics         EpochMetrics
		scratch         fitScratch
		trainingBytes   int64
		validationBytes int64
		logicalBytes    int64
		err             error
		index           int
	)

	trainingData = benchmarkFitDataViewsSequenceDataset(
		b,
		benchmarkFitDataViewsSequenceSamples,
	)
	validationData = benchmarkFitDataViewsSequenceDataset(
		b,
		benchmarkFitDataViewsSequenceValidation,
	)
	network = benchmarkFitDataViewsSequenceModel(b)
	config = benchmarkFitSequenceDataViewsConfig(b, validationData, shuffle)
	if selector, selectorIndex, err = network.lengthAwareGraph("benchmark fit"); err != nil {
		b.Fatalf("lengthAwareGraph returned error: %v", err)
	}
	if cold {
		if _, err = network.FitWithLengths(trainingData, config); err != nil {
			b.Fatalf("warm-up FitWithLengths returned error: %v", err)
		}
	} else {
		if metrics, err = benchmarkFitSequenceDataViewsWarmEpoch(
			network,
			trainingData,
			config,
			selector,
			selectorIndex,
			1,
			&scratch,
		); err != nil {
			b.Fatalf("warm-up sequence epoch returned error: %v", err)
		}
	}

	trainingBytes = benchmarkFitDataViewsSequenceBytes(
		benchmarkFitDataViewsSequenceSamples,
	)
	validationBytes = benchmarkFitDataViewsSequenceBytes(
		benchmarkFitDataViewsSequenceValidation,
	)
	logicalBytes = trainingBytes*2 + validationBytes
	b.ResetTimer()
	reportFitDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		if cold {
			history, err = network.FitWithLengths(trainingData, config)
			if err != nil {
				b.Fatalf("FitWithLengths returned error: %v", err)
			}
			continue
		}

		metrics, err = benchmarkFitSequenceDataViewsWarmEpoch(
			network,
			trainingData,
			config,
			selector,
			selectorIndex,
			index+2,
			&scratch,
		)
		if err != nil {
			b.Fatalf("sequence epoch returned error: %v", err)
		}
	}

	b.StopTimer()
	network.invalidateLengthAwareForward()
	if err = scratch.release(); err != nil {
		b.Fatalf("release sequence fit scratch returned error: %v", err)
	}
	if cold {
		metrics = benchmarkFitDataViewsHistoryMetrics(b, history)
	}
	verifyFitDataViewsEpochMetrics(b, metrics)
	benchmarkFitDataViewsHistory = history
	benchmarkFitDataViewsMetrics = metrics
}

func benchmarkFitDataViewsWarmEpoch(
	network *Sequential,
	trainingData *data.Dataset,
	config FitConfig,
	epoch int,
	scratch *fitScratch,
) (metrics EpochMetrics, err error) {
	if err = network.trainFitEpoch(
		trainingData,
		config,
		epoch,
		scratch,
	); err != nil {
		return metrics, err
	}

	metrics, err = network.fitEpochMetrics(
		epoch,
		trainingData,
		config,
		scratch,
	)
	return metrics, err
}

func benchmarkFitSequenceDataViewsWarmEpoch(
	network *Sequential,
	trainingData *data.SequenceDataset,
	config SequenceFitConfig,
	selector sequenceLengthLayer,
	selectorIndex,
	epoch int,
	scratch *fitScratch,
) (metrics EpochMetrics, err error) {
	if err = network.trainSequenceFitEpoch(
		trainingData,
		config,
		epoch,
		selector,
		selectorIndex,
		scratch,
	); err != nil {
		return metrics, err
	}

	metrics, err = network.sequenceFitEpochMetrics(
		epoch,
		trainingData,
		config,
		selector,
		selectorIndex,
		scratch,
	)
	return metrics, err
}

func benchmarkFitDataViewsConfig(
	tb testing.TB,
	validationData *data.Dataset,
	shuffle bool,
) (config FitConfig) {
	var (
		optimizerRule *optimizer.SGD
		err           error
	)

	tb.Helper()

	if optimizerRule, err = optimizer.NewSGD(0.000001); err != nil {
		tb.Fatalf("NewSGD returned error: %v", err)
	}
	config.Epochs = 1
	config.BatchSize = benchmarkFitDataViewsOrdinaryBatchSize
	config.Shuffle = shuffle
	config.Random = rand.New(rand.NewSource(101))
	config.Optimizer = optimizerRule
	config.Loss = loss.MeanSquaredError{}
	config.ValidationData = validationData
	return config
}

func benchmarkFitSequenceDataViewsConfig(
	tb testing.TB,
	validationData *data.SequenceDataset,
	shuffle bool,
) (config SequenceFitConfig) {
	var (
		optimizerRule *optimizer.SGD
		err           error
	)

	tb.Helper()

	if optimizerRule, err = optimizer.NewSGD(0.000001); err != nil {
		tb.Fatalf("NewSGD returned error: %v", err)
	}
	config.Epochs = 1
	if shuffle {
		config.BatchSize = benchmarkFitDataViewsSequenceBatchSize
	} else {
		config.BatchSize = benchmarkFitDataViewsSequencePartialBatch
	}
	config.Shuffle = shuffle
	config.Random = rand.New(rand.NewSource(103))
	config.Optimizer = optimizerRule
	config.Loss = loss.MeanSquaredError{}
	config.ValidationData = validationData
	return config
}

func benchmarkFitDataViewsOrdinaryDataset(
	tb testing.TB,
) (dataset *data.Dataset) {
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

	inputValues = make(
		[]float32,
		benchmarkFitDataViewsOrdinarySamples*benchmarkFitDataViewsOrdinaryInputs,
	)
	targetValues = make(
		[]float32,
		benchmarkFitDataViewsOrdinarySamples*benchmarkFitDataViewsOrdinaryTargets,
	)
	for row = 0; row < benchmarkFitDataViewsOrdinarySamples; row++ {
		for column = 0; column < benchmarkFitDataViewsOrdinaryInputs; column++ {
			inputValues[row*benchmarkFitDataViewsOrdinaryInputs+column] =
				float32((row+1)*(column+3)%29) / 29
		}
		for column = 0; column < benchmarkFitDataViewsOrdinaryTargets; column++ {
			targetValues[row*benchmarkFitDataViewsOrdinaryTargets+column] =
				float32((row*3+column)%17) / 17
		}
	}
	if inputs, err = matrix.FromSlice(
		benchmarkFitDataViewsOrdinarySamples,
		benchmarkFitDataViewsOrdinaryInputs,
		inputValues,
	); err != nil {
		tb.Fatalf("FromSlice ordinary inputs returned error: %v", err)
	}
	if targets, err = matrix.FromSlice(
		benchmarkFitDataViewsOrdinarySamples,
		benchmarkFitDataViewsOrdinaryTargets,
		targetValues,
	); err != nil {
		tb.Fatalf("FromSlice ordinary targets returned error: %v", err)
	}
	if dataset, err = data.NewDataset(inputs, targets); err != nil {
		tb.Fatalf("NewDataset returned error: %v", err)
	}
	verifyFitDataViewsOrdinaryFixture(tb, dataset)
	return dataset
}

func benchmarkFitDataViewsSequenceDataset(
	tb testing.TB,
	samples int,
) (dataset *data.SequenceDataset) {
	var (
		inputValues  []float32
		targetValues []float32
		lengthValues []int
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		lengths      *data.SequenceLengths
		row          int
		column       int
		err          error
	)

	tb.Helper()

	inputValues = make(
		[]float32,
		samples*benchmarkFitDataViewsSequenceInputs,
	)
	targetValues = make(
		[]float32,
		samples*benchmarkFitDataViewsSequenceTargets,
	)
	lengthValues = make([]int, samples)
	for row = 0; row < samples; row++ {
		for column = 0; column < benchmarkFitDataViewsSequenceInputs; column++ {
			inputValues[row*benchmarkFitDataViewsSequenceInputs+column] =
				float32((row+column*3)%31) / 31
		}
		for column = 0; column < benchmarkFitDataViewsSequenceTargets; column++ {
			targetValues[row*benchmarkFitDataViewsSequenceTargets+column] =
				float32((row*5+column)%19) / 19
		}
		lengthValues[row] = row%benchmarkFitDataViewsSequenceSteps + 1
	}
	if inputs, err = matrix.FromSlice(
		samples,
		benchmarkFitDataViewsSequenceInputs,
		inputValues,
	); err != nil {
		tb.Fatalf("FromSlice sequence inputs returned error: %v", err)
	}
	if targets, err = matrix.FromSlice(
		samples,
		benchmarkFitDataViewsSequenceTargets,
		targetValues,
	); err != nil {
		tb.Fatalf("FromSlice sequence targets returned error: %v", err)
	}
	if lengths, err = data.NewSequenceLengths(
		benchmarkFitDataViewsSequenceSteps,
		lengthValues,
	); err != nil {
		tb.Fatalf("NewSequenceLengths returned error: %v", err)
	}
	if dataset, err = data.NewSequenceDataset(inputs, targets, lengths); err != nil {
		tb.Fatalf("NewSequenceDataset returned error: %v", err)
	}
	verifyFitDataViewsSequenceFixture(tb, dataset, samples)
	return dataset
}

func benchmarkFitDataViewsOrdinaryModel(
	tb testing.TB,
) (network *Sequential) {
	var (
		random      *rand.Rand
		initializer layer.WeightInitializer
		hidden      *layer.Dense
		linear      *layer.Activation
		output      *layer.Dense
		err         error
	)

	tb.Helper()

	random = rand.New(rand.NewSource(107))
	initializer = layer.UniformWeights(-0.01, 0.01, random)
	if hidden, err = layer.NewDense(
		benchmarkFitDataViewsOrdinaryInputs,
		32,
		initializer,
	); err != nil {
		tb.Fatalf("NewDense hidden returned error: %v", err)
	}
	if linear, err = layer.NewActivation(activation.Linear{}); err != nil {
		tb.Fatalf("NewActivation returned error: %v", err)
	}
	if output, err = layer.NewDense(
		32,
		benchmarkFitDataViewsOrdinaryTargets,
		initializer,
	); err != nil {
		tb.Fatalf("NewDense output returned error: %v", err)
	}
	if network, err = NewSequential(hidden, linear, output); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	return network
}

func benchmarkFitDataViewsSequenceModel(
	tb testing.TB,
) (network *Sequential) {
	var (
		random      *rand.Rand
		initializer layer.WeightInitializer
		inputShape  layer.SequenceShape
		config      layer.SimpleRNNConfig
		recurrent   *layer.SimpleRNN
		gather      *layer.GatherLastValid
		output      *layer.Dense
		err         error
	)

	tb.Helper()

	random = rand.New(rand.NewSource(109))
	initializer = layer.UniformWeights(-0.01, 0.01, random)
	if inputShape, err = layer.NewSequenceShape(
		benchmarkFitDataViewsSequenceSteps,
		benchmarkFitDataViewsSequenceFeatures,
	); err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if config, err = layer.NewSimpleRNNConfig(inputShape, 8); err != nil {
		tb.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	if recurrent, err = layer.NewSimpleRNN(
		config,
		initializer,
		initializer,
	); err != nil {
		tb.Fatalf("NewSimpleRNN returned error: %v", err)
	}
	if gather, err = layer.NewGatherLastValid(recurrent.OutputShape()); err != nil {
		tb.Fatalf("NewGatherLastValid returned error: %v", err)
	}
	if output, err = layer.NewDense(
		gather.OutputSize(),
		benchmarkFitDataViewsSequenceTargets,
		initializer,
	); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = NewSequential(recurrent, gather, output); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	return network
}

func verifyFitDataViewsOrdinaryFixture(
	tb testing.TB,
	dataset *data.Dataset,
) {
	var (
		inputs      *matrix.Matrix
		targets     *matrix.Matrix
		inputValue  float32
		targetValue float32
		lastRow     int
		err         error
	)

	tb.Helper()

	if inputs, err = dataset.Inputs(); err != nil {
		tb.Fatalf("fixture Inputs returned error: %v", err)
	}
	if targets, err = dataset.Targets(); err != nil {
		tb.Fatalf("fixture Targets returned error: %v", err)
	}
	lastRow = benchmarkFitDataViewsOrdinarySamples - 1
	if inputValue, err = inputs.At(lastRow, 0); err != nil {
		tb.Fatalf("fixture input At returned error: %v", err)
	}
	if targetValue, err = targets.At(lastRow, 0); err != nil {
		tb.Fatalf("fixture target At returned error: %v", err)
	}
	benchmarkFitDataViewsChecksum = inputValue + targetValue
}

func verifyFitDataViewsSequenceFixture(
	tb testing.TB,
	dataset *data.SequenceDataset,
	samples int,
) {
	var (
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		lengths      *data.SequenceLengths
		lengthValues []int
		inputValue   float32
		targetValue  float32
		lastRow      int
		err          error
	)

	tb.Helper()

	if inputs, err = dataset.Inputs(); err != nil {
		tb.Fatalf("fixture sequence Inputs returned error: %v", err)
	}
	if targets, err = dataset.Targets(); err != nil {
		tb.Fatalf("fixture sequence Targets returned error: %v", err)
	}
	if lengths, err = dataset.Lengths(); err != nil {
		tb.Fatalf("fixture sequence Lengths returned error: %v", err)
	}
	if lengthValues, err = lengths.Values(); err != nil {
		tb.Fatalf("fixture sequence length Values returned error: %v", err)
	}
	lastRow = samples - 1
	if inputValue, err = inputs.At(lastRow, 0); err != nil {
		tb.Fatalf("fixture sequence input At returned error: %v", err)
	}
	if targetValue, err = targets.At(lastRow, 0); err != nil {
		tb.Fatalf("fixture sequence target At returned error: %v", err)
	}
	if lengthValues[lastRow] != lastRow%benchmarkFitDataViewsSequenceSteps+1 {
		tb.Fatalf(
			"fixture sequence length = %d, want aligned row %d",
			lengthValues[lastRow],
			lastRow,
		)
	}
	benchmarkFitDataViewsChecksum =
		inputValue + targetValue + float32(lengthValues[lastRow])
}

func benchmarkFitDataViewsHistoryMetrics(
	tb testing.TB,
	history TrainingHistory,
) (metrics EpochMetrics) {
	tb.Helper()

	if len(history.Epochs) != 1 {
		tb.Fatalf("history epoch count = %d, want 1", len(history.Epochs))
	}
	metrics = history.Epochs[0]
	return metrics
}

func verifyFitDataViewsEpochMetrics(
	tb testing.TB,
	metrics EpochMetrics,
) {
	tb.Helper()

	verifyFitDataViewsFinite(tb, "training loss", metrics.Loss)
	if !metrics.HasValidationLoss {
		tb.Fatal("validation loss is absent")
	}
	verifyFitDataViewsFinite(tb, "validation loss", metrics.ValidationLoss)
}

func verifyFitDataViewsFinite(
	tb testing.TB,
	name string,
	value float32,
) {
	tb.Helper()

	if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
		tb.Fatalf("%s = %g, want finite", name, value)
	}
}

func benchmarkFitDataViewsOrdinaryBytes(samples int) (bytes int64) {
	bytes = int64(samples) *
		int64(benchmarkFitDataViewsOrdinaryInputs+benchmarkFitDataViewsOrdinaryTargets) *
		4
	return bytes
}

func benchmarkFitDataViewsSequenceBytes(samples int) (bytes int64) {
	var matrixBytes int64

	matrixBytes = int64(samples) *
		int64(benchmarkFitDataViewsSequenceInputs+benchmarkFitDataViewsSequenceTargets) *
		4
	bytes = matrixBytes + int64(samples)*int64(strconv.IntSize/8)
	return bytes
}

func reportFitDataViewsMetrics(
	b *testing.B,
	logicalBytesRead,
	logicalBytesCopied int64,
) {
	b.Helper()

	b.ReportAllocs()
	b.SetBytes(logicalBytesRead)
	b.ReportMetric(float64(logicalBytesRead), "logical-B-read/op")
	b.ReportMetric(float64(logicalBytesCopied), "logical-B-copied/op")
}
