package layer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Benchmark_Conv2DForward(b *testing.B) {
	type testcase struct {
		name           string
		batchSize      int
		inputChannels  int
		inputHeight    int
		inputWidth     int
		outputChannels int
	}

	var tests []testcase
	tests = []testcase{
		{
			name:           "SingleImage",
			batchSize:      1,
			inputChannels:  1,
			inputHeight:    12,
			inputWidth:     10,
			outputChannels: 4,
		},
		{
			name:           "BatchMultiChannel",
			batchSize:      8,
			inputChannels:  3,
			inputHeight:    16,
			inputWidth:     12,
			outputChannels: 8,
		},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkConv2DForward(b, tt.batchSize, tt.inputChannels, tt.inputHeight, tt.inputWidth, tt.outputChannels)
		})
	}
}

func Benchmark_Conv2DBackward(b *testing.B) {
	type testcase struct {
		name           string
		batchSize      int
		inputChannels  int
		inputHeight    int
		inputWidth     int
		outputChannels int
	}

	var tests []testcase
	tests = []testcase{
		{
			name:           "SingleImage",
			batchSize:      1,
			inputChannels:  1,
			inputHeight:    12,
			inputWidth:     10,
			outputChannels: 4,
		},
		{
			name:           "BatchMultiChannel",
			batchSize:      8,
			inputChannels:  3,
			inputHeight:    16,
			inputWidth:     12,
			outputChannels: 8,
		},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkConv2DBackward(b, tt.batchSize, tt.inputChannels, tt.inputHeight, tt.inputWidth, tt.outputChannels)
		})
	}
}

func Benchmark_MaxPool2DForward(b *testing.B) {
	type testcase struct {
		name          string
		batchSize     int
		inputChannels int
		inputHeight   int
		inputWidth    int
	}

	var tests []testcase
	tests = []testcase{
		{
			name:          "SingleImage",
			batchSize:     1,
			inputChannels: 1,
			inputHeight:   12,
			inputWidth:    10,
		},
		{
			name:          "BatchMultiChannel",
			batchSize:     8,
			inputChannels: 8,
			inputHeight:   16,
			inputWidth:    12,
		},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkMaxPool2DForward(b, tt.batchSize, tt.inputChannels, tt.inputHeight, tt.inputWidth)
		})
	}
}

func Benchmark_MaxPool2DBackward(b *testing.B) {
	type testcase struct {
		name          string
		batchSize     int
		inputChannels int
		inputHeight   int
		inputWidth    int
	}

	var tests []testcase
	tests = []testcase{
		{
			name:          "SingleImage",
			batchSize:     1,
			inputChannels: 1,
			inputHeight:   12,
			inputWidth:    10,
		},
		{
			name:          "BatchMultiChannel",
			batchSize:     8,
			inputChannels: 8,
			inputHeight:   16,
			inputWidth:    12,
		},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkMaxPool2DBackward(b, tt.batchSize, tt.inputChannels, tt.inputHeight, tt.inputWidth)
		})
	}
}

func benchmarkConv2DForward(
	b *testing.B,
	batchSize, inputChannels, inputHeight, inputWidth, outputChannels int,
) {
	var (
		convolution *layer.Conv2D
		input       *matrix.Matrix
		output      *matrix.Matrix
		err         error
		index       int
	)

	convolution = benchmarkConv2D(b, inputChannels, inputHeight, inputWidth, outputChannels)
	input = benchmarkLayerMatrix(b, batchSize, convolution.InputShape().Size())
	if output, err = convolution.Forward(input); err != nil {
		b.Fatalf("warm-up Forward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if output, err = convolution.Forward(input); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = output
}

func benchmarkConv2DBackward(
	b *testing.B,
	batchSize, inputChannels, inputHeight, inputWidth, outputChannels int,
) {
	var (
		convolution    *layer.Conv2D
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
		index          int
	)

	convolution = benchmarkConv2D(b, inputChannels, inputHeight, inputWidth, outputChannels)
	input = benchmarkLayerMatrix(b, batchSize, convolution.InputShape().Size())
	outputGradient = benchmarkLayerMatrix(b, batchSize, convolution.OutputShape().Size())
	if _, err = convolution.Forward(input); err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}
	if inputGradient, err = convolution.Backward(outputGradient); err != nil {
		b.Fatalf("warm-up Backward returned error: %v", err)
	}
	if err = convolution.ResetGradients(); err != nil {
		b.Fatalf("ResetGradients returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if inputGradient, err = convolution.Backward(outputGradient); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = inputGradient
}

func benchmarkMaxPool2DForward(
	b *testing.B,
	batchSize, inputChannels, inputHeight, inputWidth int,
) {
	var (
		pooling *layer.MaxPool2D
		input   *matrix.Matrix
		output  *matrix.Matrix
		err     error
		index   int
	)

	pooling = benchmarkMaxPool2D(b, inputChannels, inputHeight, inputWidth)
	input = benchmarkLayerMatrix(b, batchSize, pooling.InputShape().Size())
	if output, err = pooling.Forward(input); err != nil {
		b.Fatalf("warm-up Forward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if output, err = pooling.Forward(input); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = output
}

func benchmarkMaxPool2DBackward(
	b *testing.B,
	batchSize, inputChannels, inputHeight, inputWidth int,
) {
	var (
		pooling        *layer.MaxPool2D
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
		index          int
	)

	pooling = benchmarkMaxPool2D(b, inputChannels, inputHeight, inputWidth)
	input = benchmarkLayerMatrix(b, batchSize, pooling.InputShape().Size())
	outputGradient = benchmarkLayerMatrix(b, batchSize, pooling.OutputShape().Size())
	if _, err = pooling.Forward(input); err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}
	if inputGradient, err = pooling.Backward(outputGradient); err != nil {
		b.Fatalf("warm-up Backward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if inputGradient, err = pooling.Backward(outputGradient); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = inputGradient
}

func benchmarkConv2D(
	tb testing.TB,
	inputChannels, inputHeight, inputWidth, outputChannels int,
) (convolution *layer.Conv2D) {
	var (
		inputShape layer.SpatialShape
		config     layer.Conv2DConfig
		err        error
	)

	tb.Helper()

	if inputShape, err = layer.NewSpatialShape(inputChannels, inputHeight, inputWidth); err != nil {
		tb.Fatalf("NewSpatialShape returned error: %v", err)
	}
	if config, err = layer.NewConv2DConfig(inputShape, outputChannels, 3, 3, 1, 1, 1, 1); err != nil {
		tb.Fatalf("NewConv2DConfig returned error: %v", err)
	}
	if convolution, err = layer.NewConv2D(config, layer.ZeroWeights); err != nil {
		tb.Fatalf("NewConv2D returned error: %v", err)
	}

	return convolution
}

func benchmarkMaxPool2D(
	tb testing.TB,
	inputChannels, inputHeight, inputWidth int,
) (pooling *layer.MaxPool2D) {
	var (
		inputShape layer.SpatialShape
		config     layer.MaxPool2DConfig
		err        error
	)

	tb.Helper()

	if inputShape, err = layer.NewSpatialShape(inputChannels, inputHeight, inputWidth); err != nil {
		tb.Fatalf("NewSpatialShape returned error: %v", err)
	}
	if config, err = layer.NewMaxPool2DConfig(inputShape, 2, 3, 2, 2); err != nil {
		tb.Fatalf("NewMaxPool2DConfig returned error: %v", err)
	}
	if pooling, err = layer.NewMaxPool2D(config); err != nil {
		tb.Fatalf("NewMaxPool2D returned error: %v", err)
	}

	return pooling
}
