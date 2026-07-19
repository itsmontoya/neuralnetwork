package model_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

var benchmarkTrainMetrics model.TrainMetrics
var benchmarkTrainingHistory model.TrainingHistory
var benchmarkParameters []*optimizer.Parameter

func Benchmark_SequentialParameters(b *testing.B) {
	var (
		network    *model.Sequential
		parameters []*optimizer.Parameter
		index      int
	)

	network = benchmarkSyntheticModel(b)
	parameters = network.Parameters()

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		parameters = network.Parameters()
	}

	benchmarkParameters = parameters
}

func Benchmark_SequentialTrainBatch_XOR(b *testing.B) {
	var (
		network       *model.Sequential
		optimizerRule optimizer.Optimizer
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		metrics       model.TrainMetrics
		err           error
		index         int
	)

	inputs, targets = benchmarkXORMatrices(b)
	network = benchmarkXORModel(b)
	optimizerRule, err = optimizer.NewAdam(0.05)
	if err != nil {
		b.Fatalf("NewAdam returned error: %v", err)
	}
	if _, err = network.TrainBatch(inputs, targets, loss.BinaryCrossEntropy{}, optimizerRule); err != nil {
		b.Fatalf("warm-up TrainBatch returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		metrics, err = network.TrainBatch(inputs, targets, loss.BinaryCrossEntropy{}, optimizerRule)
		if err != nil {
			b.Fatalf("TrainBatch returned error: %v", err)
		}
	}

	benchmarkTrainMetrics = metrics
}

func Benchmark_SequentialFit_XOR(b *testing.B) {
	var (
		network       *model.Sequential
		optimizerRule optimizer.Optimizer
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		dataset       *data.Dataset
		config        model.FitConfig
		history       model.TrainingHistory
		err           error
		index         int
	)

	inputs, targets = benchmarkXORMatrices(b)
	if dataset, err = data.NewDataset(inputs, targets); err != nil {
		b.Fatalf("NewDataset returned error: %v", err)
	}

	network = benchmarkXORModel(b)
	optimizerRule, err = optimizer.NewAdam(0.05)
	if err != nil {
		b.Fatalf("NewAdam returned error: %v", err)
	}

	config.Epochs = 1
	config.BatchSize = 4
	config.Optimizer = optimizerRule
	config.Loss = loss.BinaryCrossEntropy{}
	if _, err = network.Fit(dataset, config); err != nil {
		b.Fatalf("warm-up Fit returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		history, err = network.Fit(dataset, config)
		if err != nil {
			b.Fatalf("Fit returned error: %v", err)
		}
	}

	benchmarkTrainingHistory = history
}

func Benchmark_SequentialFit_XOR_Accuracy(b *testing.B) {
	var (
		network       *model.Sequential
		optimizerRule optimizer.Optimizer
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		dataset       *data.Dataset
		config        model.FitConfig
		history       model.TrainingHistory
		err           error
		index         int
	)

	inputs, targets = benchmarkXORMatrices(b)
	if dataset, err = data.NewDataset(inputs, targets); err != nil {
		b.Fatalf("NewDataset returned error: %v", err)
	}

	network = benchmarkXORModel(b)
	optimizerRule, err = optimizer.NewAdam(0.05)
	if err != nil {
		b.Fatalf("NewAdam returned error: %v", err)
	}

	config.Epochs = 1
	config.BatchSize = 4
	config.Optimizer = optimizerRule
	config.Loss = loss.BinaryCrossEntropy{}
	config.Accuracy = metric.BinaryAccuracy{}.Value
	if _, err = network.Fit(dataset, config); err != nil {
		b.Fatalf("warm-up Fit returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		history, err = network.Fit(dataset, config)
		if err != nil {
			b.Fatalf("Fit returned error: %v", err)
		}
	}

	benchmarkTrainingHistory = history
}

func Benchmark_SequentialTrainBatch_SyntheticDense(b *testing.B) {
	var (
		network       *model.Sequential
		optimizerRule optimizer.Optimizer
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		metrics       model.TrainMetrics
		err           error
		index         int
	)

	inputs, targets = benchmarkSyntheticMatrices(b, 128, 32, 16)
	network = benchmarkSyntheticModel(b)
	optimizerRule, err = optimizer.NewSGD(0.01)
	if err != nil {
		b.Fatalf("NewSGD returned error: %v", err)
	}
	if _, err = network.TrainBatch(inputs, targets, loss.MeanSquaredError{}, optimizerRule); err != nil {
		b.Fatalf("warm-up TrainBatch returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		metrics, err = network.TrainBatch(inputs, targets, loss.MeanSquaredError{}, optimizerRule)
		if err != nil {
			b.Fatalf("TrainBatch returned error: %v", err)
		}
	}

	benchmarkTrainMetrics = metrics
}

func Benchmark_SequentialTrainBatch_CNN(b *testing.B) {
	var (
		network       *model.Sequential
		optimizerRule *optimizer.SGD
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		metrics       model.TrainMetrics
		err           error
		index         int
	)

	network = benchmarkCNNModel(b)
	inputs, targets = benchmarkSyntheticMatrices(b, 8, 3*16*12, 6)
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

func Benchmark_SequentialFit_SyntheticDense(b *testing.B) {
	var (
		network       *model.Sequential
		optimizerRule optimizer.Optimizer
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		dataset       *data.Dataset
		config        model.FitConfig
		history       model.TrainingHistory
		err           error
		index         int
	)

	inputs, targets = benchmarkSyntheticMatrices(b, 128, 32, 16)
	if dataset, err = data.NewDataset(inputs, targets); err != nil {
		b.Fatalf("NewDataset returned error: %v", err)
	}

	network = benchmarkSyntheticModel(b)
	optimizerRule, err = optimizer.NewSGD(0.01)
	if err != nil {
		b.Fatalf("NewSGD returned error: %v", err)
	}

	config.Epochs = 1
	config.BatchSize = 32
	config.Optimizer = optimizerRule
	config.Loss = loss.MeanSquaredError{}
	if _, err = network.Fit(dataset, config); err != nil {
		b.Fatalf("warm-up Fit returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		history, err = network.Fit(dataset, config)
		if err != nil {
			b.Fatalf("Fit returned error: %v", err)
		}
	}

	benchmarkTrainingHistory = history
}

func Benchmark_SequentialTrainBatch_Activations(b *testing.B) {
	var tests []struct {
		name     string
		function activation.Activation
	}

	tests = []struct {
		name     string
		function activation.Activation
	}{
		{name: "ELU", function: activation.ELU{}},
		{name: "GELU", function: activation.GELU{}},
		{name: "LeakyReLU", function: activation.LeakyReLU{}},
		{name: "Linear", function: activation.Linear{}},
		{name: "ReLU", function: activation.ReLU{}},
		{name: "Sigmoid", function: activation.Sigmoid{}},
		{name: "Tanh", function: activation.Tanh{}},
		{name: "Softmax", function: activation.Softmax{}},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkSequentialTrainBatchActivation(b, tt.function)
		})
	}
}

func Benchmark_SequentialTrainBatch_Regularized(b *testing.B) {
	var tests []struct {
		name string
		new  func(testing.TB) optimizer.Regularizer
	}

	tests = []struct {
		name string
		new  func(testing.TB) optimizer.Regularizer
	}{
		{
			name: "L1",
			new: func(tb testing.TB) (regularizer optimizer.Regularizer) {
				var (
					l1  *optimizer.L1
					err error
				)

				if l1, err = optimizer.NewL1(0.001); err != nil {
					tb.Fatalf("NewL1 returned error: %v", err)
				}

				return l1
			},
		},
		{
			name: "L2",
			new: func(tb testing.TB) (regularizer optimizer.Regularizer) {
				var (
					l2  *optimizer.L2WeightDecay
					err error
				)

				if l2, err = optimizer.NewL2WeightDecay(0.001); err != nil {
					tb.Fatalf("NewL2WeightDecay returned error: %v", err)
				}

				return l2
			},
		},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkSequentialTrainBatchRegularized(b, tt.new(b))
		})
	}
}

func Benchmark_SequentialTrainBatch_AlternatingShapes(b *testing.B) {
	var (
		sampleCounts  []int
		inputs        []*matrix.Matrix
		targets       []*matrix.Matrix
		network       *model.Sequential
		optimizerRule *optimizer.SGD
		metrics       model.TrainMetrics
		err           error
		index         int
		shapeIndex    int
	)

	sampleCounts = []int{128, 17, 64, 31}
	inputs = make([]*matrix.Matrix, len(sampleCounts))
	targets = make([]*matrix.Matrix, len(sampleCounts))
	for shapeIndex = range sampleCounts {
		inputs[shapeIndex], targets[shapeIndex] = benchmarkSyntheticMatrices(
			b,
			sampleCounts[shapeIndex],
			32,
			16,
		)
	}

	network = benchmarkSyntheticModel(b)
	if optimizerRule, err = optimizer.NewSGD(0.01); err != nil {
		b.Fatalf("NewSGD returned error: %v", err)
	}

	for shapeIndex = range sampleCounts {
		if _, err = network.TrainBatch(
			inputs[shapeIndex],
			targets[shapeIndex],
			loss.MeanSquaredError{},
			optimizerRule,
		); err != nil {
			b.Fatalf("warm-up TrainBatch returned error: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		shapeIndex = index % len(sampleCounts)
		if metrics, err = network.TrainBatch(
			inputs[shapeIndex],
			targets[shapeIndex],
			loss.MeanSquaredError{},
			optimizerRule,
		); err != nil {
			b.Fatalf("TrainBatch returned error: %v", err)
		}
	}

	benchmarkTrainMetrics = metrics
}

func Benchmark_SequentialFit_SyntheticDense_ColdOneEpoch(b *testing.B) {
	var (
		network       *model.Sequential
		optimizerRule *optimizer.SGD
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		dataset       *data.Dataset
		config        model.FitConfig
		history       model.TrainingHistory
		err           error
		index         int
	)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		b.StopTimer()
		inputs, targets = benchmarkSyntheticMatrices(b, 128, 32, 16)
		if dataset, err = data.NewDataset(inputs, targets); err != nil {
			b.Fatalf("NewDataset returned error: %v", err)
		}

		network = benchmarkSyntheticModel(b)
		if optimizerRule, err = optimizer.NewSGD(0.01); err != nil {
			b.Fatalf("NewSGD returned error: %v", err)
		}

		config = model.FitConfig{}
		config.Epochs = 1
		config.BatchSize = 32
		config.Optimizer = optimizerRule
		config.Loss = loss.MeanSquaredError{}
		b.StartTimer()

		if history, err = network.Fit(dataset, config); err != nil {
			b.Fatalf("Fit returned error: %v", err)
		}
	}

	benchmarkTrainingHistory = history
}

func Benchmark_SequentialFit_SyntheticDense_TenEpoch(b *testing.B) {
	var (
		network       *model.Sequential
		optimizerRule *optimizer.SGD
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		dataset       *data.Dataset
		config        model.FitConfig
		history       model.TrainingHistory
		err           error
		index         int
	)

	inputs, targets = benchmarkSyntheticMatrices(b, 128, 32, 16)
	if dataset, err = data.NewDataset(inputs, targets); err != nil {
		b.Fatalf("NewDataset returned error: %v", err)
	}

	network = benchmarkSyntheticModel(b)
	if optimizerRule, err = optimizer.NewSGD(0.01); err != nil {
		b.Fatalf("NewSGD returned error: %v", err)
	}

	config.Epochs = 1
	config.BatchSize = 32
	config.Optimizer = optimizerRule
	config.Loss = loss.MeanSquaredError{}
	if _, err = network.Fit(dataset, config); err != nil {
		b.Fatalf("warm-up Fit returned error: %v", err)
	}
	config.Epochs = 10

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if history, err = network.Fit(dataset, config); err != nil {
			b.Fatalf("Fit returned error: %v", err)
		}
	}

	benchmarkTrainingHistory = history
}

func Benchmark_SequentialFit_Scenarios(b *testing.B) {
	var tests []struct {
		name              string
		samples           int
		batchSize         int
		shuffle           bool
		validationSamples int
	}

	tests = []struct {
		name              string
		samples           int
		batchSize         int
		shuffle           bool
		validationSamples int
	}{
		{name: "PartialFinalBatch", samples: 130, batchSize: 32},
		{name: "Shuffle", samples: 128, batchSize: 32, shuffle: true},
		{name: "Validation", samples: 128, batchSize: 32, validationSamples: 65},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkSequentialFitScenario(
				b,
				tt.samples,
				tt.batchSize,
				tt.shuffle,
				tt.validationSamples,
			)
		})
	}
}

func benchmarkXORMatrices(tb testing.TB) (inputs, targets *matrix.Matrix) {
	var err error

	tb.Helper()

	if inputs, err = matrix.FromSlice(4, 2, []float32{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	}); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	if targets, err = matrix.FromSlice(4, 1, []float32{
		0,
		1,
		1,
		0,
	}); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return inputs, targets
}

func benchmarkXORModel(tb testing.TB) (network *model.Sequential) {
	var (
		random           *rand.Rand
		hidden           *layer.Dense
		hiddenActivation *layer.Activation
		output           *layer.Dense
		outputActivation *layer.Activation
		err              error
	)

	tb.Helper()

	random = rand.New(rand.NewSource(1))
	if hidden, err = layer.NewDense(2, 4, layer.XavierUniformWeights(random)); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}

	if hiddenActivation, err = layer.NewActivation(activation.Tanh{}); err != nil {
		tb.Fatalf("NewActivation returned error: %v", err)
	}

	if output, err = layer.NewDense(4, 1, layer.XavierUniformWeights(random)); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}

	if outputActivation, err = layer.NewActivation(activation.Sigmoid{}); err != nil {
		tb.Fatalf("NewActivation returned error: %v", err)
	}

	if network, err = model.NewSequential(hidden, hiddenActivation, output, outputActivation); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}

	return network
}

func benchmarkSyntheticMatrices(tb testing.TB, samples, inputSize, targetSize int) (inputs, targets *matrix.Matrix) {
	var (
		inputValues  []float32
		targetValues []float32
		row          int
		col          int
		err          error
	)

	tb.Helper()

	inputValues = make([]float32, samples*inputSize)
	for row = 0; row < samples; row++ {
		for col = 0; col < inputSize; col++ {
			inputValues[row*inputSize+col] = float32((row+1)*(col+3)%17) / 17
		}
	}

	targetValues = make([]float32, samples*targetSize)
	for row = 0; row < samples; row++ {
		for col = 0; col < targetSize; col++ {
			targetValues[row*targetSize+col] = float32((row+col*2)%11) / 11
		}
	}

	if inputs, err = matrix.FromSlice(samples, inputSize, inputValues); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	if targets, err = matrix.FromSlice(samples, targetSize, targetValues); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return inputs, targets
}

func benchmarkSyntheticModel(tb testing.TB) (network *model.Sequential) {
	var (
		random           *rand.Rand
		hidden           *layer.Dense
		hiddenActivation *layer.Activation
		output           *layer.Dense
		err              error
	)

	tb.Helper()

	random = rand.New(rand.NewSource(3))
	if hidden, err = layer.NewDense(32, 64, layer.HeNormalWeights(random)); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}

	if hiddenActivation, err = layer.NewActivation(activation.ReLU{}); err != nil {
		tb.Fatalf("NewActivation returned error: %v", err)
	}

	if output, err = layer.NewDense(64, 16, layer.XavierUniformWeights(random)); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}

	if network, err = model.NewSequential(hidden, hiddenActivation, output); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}

	return network
}

func benchmarkCNNModel(tb testing.TB) (network *model.Sequential) {
	var (
		random            *rand.Rand
		inputShape        layer.SpatialShape
		convolutionConfig layer.Conv2DConfig
		convolution       *layer.Conv2D
		hiddenActivation  *layer.Activation
		poolingConfig     layer.MaxPool2DConfig
		pooling           *layer.MaxPool2D
		flatten           *layer.Flatten
		output            *layer.Dense
		err               error
	)

	tb.Helper()

	random = rand.New(rand.NewSource(29))
	if inputShape, err = layer.NewSpatialShape(3, 16, 12); err != nil {
		tb.Fatalf("NewSpatialShape returned error: %v", err)
	}
	if convolutionConfig, err = layer.NewConv2DConfig(inputShape, 8, 3, 3, 1, 1, 1, 1); err != nil {
		tb.Fatalf("NewConv2DConfig returned error: %v", err)
	}
	if convolution, err = layer.NewConv2D(convolutionConfig, layer.HeNormalWeights(random)); err != nil {
		tb.Fatalf("NewConv2D returned error: %v", err)
	}
	if hiddenActivation, err = layer.NewActivation(activation.ReLU{}); err != nil {
		tb.Fatalf("NewActivation returned error: %v", err)
	}
	if poolingConfig, err = layer.NewMaxPool2DConfig(convolution.OutputShape(), 2, 3, 2, 2); err != nil {
		tb.Fatalf("NewMaxPool2DConfig returned error: %v", err)
	}
	if pooling, err = layer.NewMaxPool2D(poolingConfig); err != nil {
		tb.Fatalf("NewMaxPool2D returned error: %v", err)
	}
	if flatten, err = layer.NewFlatten(pooling.OutputShape()); err != nil {
		tb.Fatalf("NewFlatten returned error: %v", err)
	}
	if output, err = layer.NewDense(flatten.OutputSize(), 6, layer.XavierUniformWeights(random)); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = model.NewSequential(convolution, hiddenActivation, pooling, flatten, output); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}

	return network
}

func benchmarkSequentialTrainBatchActivation(b *testing.B, function activation.Activation) {
	var (
		random          *rand.Rand
		dense           *layer.Dense
		activationLayer *layer.Activation
		network         *model.Sequential
		optimizerRule   *optimizer.SGD
		inputs          *matrix.Matrix
		targets         *matrix.Matrix
		metrics         model.TrainMetrics
		err             error
		index           int
	)

	random = rand.New(rand.NewSource(17))
	if dense, err = layer.NewDense(16, 8, layer.XavierUniformWeights(random)); err != nil {
		b.Fatalf("NewDense returned error: %v", err)
	}

	if activationLayer, err = layer.NewActivation(function); err != nil {
		b.Fatalf("NewActivation returned error: %v", err)
	}

	if network, err = model.NewSequential(dense, activationLayer); err != nil {
		b.Fatalf("NewSequential returned error: %v", err)
	}

	if optimizerRule, err = optimizer.NewSGD(0.01); err != nil {
		b.Fatalf("NewSGD returned error: %v", err)
	}

	inputs, targets = benchmarkSyntheticMatrices(b, 64, 16, 8)
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

func benchmarkSequentialTrainBatchRegularized(b *testing.B, regularizer optimizer.Regularizer) {
	var (
		network       *model.Sequential
		base          *optimizer.SGD
		optimizerRule *optimizer.Regularized
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		metrics       model.TrainMetrics
		err           error
		index         int
	)

	inputs, targets = benchmarkSyntheticMatrices(b, 128, 32, 16)
	network = benchmarkSyntheticModel(b)
	if base, err = optimizer.NewSGD(0.01); err != nil {
		b.Fatalf("NewSGD returned error: %v", err)
	}

	if optimizerRule, err = optimizer.NewRegularized(base, regularizer); err != nil {
		b.Fatalf("NewRegularized returned error: %v", err)
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

func benchmarkSequentialFitScenario(
	b *testing.B,
	samples, batchSize int,
	shuffle bool,
	validationSamples int,
) {
	var (
		network           *model.Sequential
		optimizerRule     *optimizer.SGD
		inputs            *matrix.Matrix
		targets           *matrix.Matrix
		validationInputs  *matrix.Matrix
		validationTargets *matrix.Matrix
		trainingData      *data.Dataset
		validationData    *data.Dataset
		config            model.FitConfig
		history           model.TrainingHistory
		err               error
		index             int
	)

	inputs, targets = benchmarkSyntheticMatrices(b, samples, 32, 16)
	if trainingData, err = data.NewDataset(inputs, targets); err != nil {
		b.Fatalf("NewDataset returned error: %v", err)
	}

	if validationSamples > 0 {
		validationInputs, validationTargets = benchmarkSyntheticMatrices(b, validationSamples, 32, 16)
		if validationData, err = data.NewDataset(validationInputs, validationTargets); err != nil {
			b.Fatalf("validation NewDataset returned error: %v", err)
		}
	}

	network = benchmarkSyntheticModel(b)
	if optimizerRule, err = optimizer.NewSGD(0.01); err != nil {
		b.Fatalf("NewSGD returned error: %v", err)
	}

	config.Epochs = 1
	config.BatchSize = batchSize
	config.Shuffle = shuffle
	if shuffle {
		config.Random = rand.New(rand.NewSource(23))
	}
	config.Optimizer = optimizerRule
	config.Loss = loss.MeanSquaredError{}
	config.ValidationData = validationData
	if _, err = network.Fit(trainingData, config); err != nil {
		b.Fatalf("warm-up Fit returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if history, err = network.Fit(trainingData, config); err != nil {
			b.Fatalf("Fit returned error: %v", err)
		}
	}

	benchmarkTrainingHistory = history
}
