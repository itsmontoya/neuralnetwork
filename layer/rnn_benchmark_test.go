package layer_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Benchmark_SimpleRNNForward(b *testing.B) {
	type testcase struct {
		name             string
		batchSize        int
		steps            int
		inputFeatureSize int
		hiddenSize       int
	}

	var tests []testcase
	tests = []testcase{
		{name: "SingleSequence", batchSize: 1, steps: 4, inputFeatureSize: 3, hiddenSize: 5},
		{name: "Batched", batchSize: 16, steps: 8, inputFeatureSize: 16, hiddenSize: 32},
		{name: "LongSequence", batchSize: 8, steps: 32, inputFeatureSize: 8, hiddenSize: 16},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkSimpleRNNForward(
				b,
				tt.batchSize,
				tt.steps,
				tt.inputFeatureSize,
				tt.hiddenSize,
			)
		})
	}
}

func Benchmark_SimpleRNNBackward(b *testing.B) {
	type testcase struct {
		name             string
		batchSize        int
		steps            int
		inputFeatureSize int
		hiddenSize       int
	}

	var tests []testcase
	tests = []testcase{
		{name: "SingleSequence", batchSize: 1, steps: 4, inputFeatureSize: 3, hiddenSize: 5},
		{name: "Batched", batchSize: 16, steps: 8, inputFeatureSize: 16, hiddenSize: 32},
		{name: "LongSequence", batchSize: 8, steps: 32, inputFeatureSize: 8, hiddenSize: 16},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkSimpleRNNBackward(
				b,
				tt.batchSize,
				tt.steps,
				tt.inputFeatureSize,
				tt.hiddenSize,
			)
		})
	}
}

func Benchmark_LastStepForward(b *testing.B) {
	type testcase struct {
		name        string
		batchSize   int
		steps       int
		featureSize int
	}

	var tests []testcase
	tests = []testcase{
		{name: "SingleSequence", batchSize: 1, steps: 4, featureSize: 5},
		{name: "Batched", batchSize: 16, steps: 8, featureSize: 32},
		{name: "LongSequence", batchSize: 8, steps: 32, featureSize: 16},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkLastStepForward(b, tt.batchSize, tt.steps, tt.featureSize)
		})
	}
}

func Benchmark_LastStepBackward(b *testing.B) {
	type testcase struct {
		name        string
		batchSize   int
		steps       int
		featureSize int
	}

	var tests []testcase
	tests = []testcase{
		{name: "SingleSequence", batchSize: 1, steps: 4, featureSize: 5},
		{name: "Batched", batchSize: 16, steps: 8, featureSize: 32},
		{name: "LongSequence", batchSize: 8, steps: 32, featureSize: 16},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkLastStepBackward(b, tt.batchSize, tt.steps, tt.featureSize)
		})
	}
}

func benchmarkSimpleRNNForward(
	b *testing.B,
	batchSize, steps, inputFeatureSize, hiddenSize int,
) {
	var (
		recurrent *layer.SimpleRNN
		input     *matrix.Matrix
		output    *matrix.Matrix
		err       error
		index     int
	)

	recurrent = benchmarkSimpleRNN(b, steps, inputFeatureSize, hiddenSize)
	input = benchmarkLayerMatrix(b, batchSize, recurrent.InputShape().Size())
	if output, err = recurrent.Forward(input); err != nil {
		b.Fatalf("warm-up Forward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if output, err = recurrent.Forward(input); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = output
}

func benchmarkSimpleRNNBackward(
	b *testing.B,
	batchSize, steps, inputFeatureSize, hiddenSize int,
) {
	var (
		recurrent      *layer.SimpleRNN
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
		index          int
	)

	recurrent = benchmarkSimpleRNN(b, steps, inputFeatureSize, hiddenSize)
	input = benchmarkLayerMatrix(b, batchSize, recurrent.InputShape().Size())
	outputGradient = benchmarkLayerMatrix(b, batchSize, recurrent.OutputShape().Size())
	if _, err = recurrent.Forward(input); err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}
	if inputGradient, err = recurrent.Backward(outputGradient); err != nil {
		b.Fatalf("warm-up Backward returned error: %v", err)
	}
	if err = recurrent.ResetGradients(); err != nil {
		b.Fatalf("ResetGradients returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if inputGradient, err = recurrent.Backward(outputGradient); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = inputGradient
}

func benchmarkLastStepForward(b *testing.B, batchSize, steps, featureSize int) {
	var (
		lastStep *layer.LastStep
		input    *matrix.Matrix
		output   *matrix.Matrix
		err      error
		index    int
	)

	lastStep = benchmarkLastStep(b, steps, featureSize)
	input = benchmarkLayerMatrix(b, batchSize, lastStep.InputShape().Size())
	if output, err = lastStep.Forward(input); err != nil {
		b.Fatalf("warm-up Forward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if output, err = lastStep.Forward(input); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = output
}

func benchmarkLastStepBackward(b *testing.B, batchSize, steps, featureSize int) {
	var (
		lastStep       *layer.LastStep
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
		index          int
	)

	lastStep = benchmarkLastStep(b, steps, featureSize)
	input = benchmarkLayerMatrix(b, batchSize, lastStep.InputShape().Size())
	outputGradient = benchmarkLayerMatrix(b, batchSize, lastStep.OutputSize())
	if _, err = lastStep.Forward(input); err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}
	if inputGradient, err = lastStep.Backward(outputGradient); err != nil {
		b.Fatalf("warm-up Backward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if inputGradient, err = lastStep.Backward(outputGradient); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = inputGradient
}

func benchmarkSimpleRNN(
	tb testing.TB,
	steps, inputFeatureSize, hiddenSize int,
) (recurrent *layer.SimpleRNN) {
	var (
		random      *rand.Rand
		inputShape  layer.SequenceShape
		config      layer.SimpleRNNConfig
		initializer layer.WeightInitializer
		err         error
	)

	tb.Helper()

	random = rand.New(rand.NewSource(41))
	if inputShape, err = layer.NewSequenceShape(steps, inputFeatureSize); err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if config, err = layer.NewSimpleRNNConfig(inputShape, hiddenSize); err != nil {
		tb.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	initializer = layer.UniformWeights(-0.1, 0.1, random)
	if recurrent, err = layer.NewSimpleRNN(config, initializer, initializer); err != nil {
		tb.Fatalf("NewSimpleRNN returned error: %v", err)
	}

	return recurrent
}

func benchmarkLastStep(tb testing.TB, steps, featureSize int) (lastStep *layer.LastStep) {
	var (
		inputShape layer.SequenceShape
		err        error
	)

	tb.Helper()

	if inputShape, err = layer.NewSequenceShape(steps, featureSize); err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if lastStep, err = layer.NewLastStep(inputShape); err != nil {
		tb.Fatalf("NewLastStep returned error: %v", err)
	}

	return lastStep
}
