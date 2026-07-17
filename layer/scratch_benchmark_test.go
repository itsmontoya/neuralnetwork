package layer_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var benchmarkScratchLayerResult *matrix.Matrix

func Benchmark_ActivationForward_MediumBatch(b *testing.B) {
	var (
		activationLayer *layer.Activation
		input           *matrix.Matrix
		output          *matrix.Matrix
		err             error
		index           int
	)

	activationLayer, err = layer.NewActivation(activation.Sigmoid{})
	if err != nil {
		b.Fatalf("NewActivation returned error: %v", err)
	}

	input = benchmarkLayerMatrix(b, 128, 64)
	output, err = activationLayer.Forward(input)
	if err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		output, err = activationLayer.Forward(input)
		if err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = output
}

func Benchmark_ActivationBackward_MediumBatch(b *testing.B) {
	var (
		activationLayer *layer.Activation
		input           *matrix.Matrix
		outputGradient  *matrix.Matrix
		inputGradient   *matrix.Matrix
		err             error
		index           int
	)

	activationLayer, err = layer.NewActivation(activation.Sigmoid{})
	if err != nil {
		b.Fatalf("NewActivation returned error: %v", err)
	}

	input = benchmarkLayerMatrix(b, 128, 64)
	outputGradient = benchmarkLayerMatrix(b, 128, 64)
	if _, err = activationLayer.Forward(input); err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}

	inputGradient, err = activationLayer.Backward(outputGradient)
	if err != nil {
		b.Fatalf("Backward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		inputGradient, err = activationLayer.Backward(outputGradient)
		if err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = inputGradient
}

func Benchmark_ActivationForward_Softmax_MediumBatch(b *testing.B) {
	var (
		activationLayer *layer.Activation
		input           *matrix.Matrix
		output          *matrix.Matrix
		err             error
		index           int
	)

	activationLayer, err = layer.NewActivation(activation.Softmax{})
	if err != nil {
		b.Fatalf("NewActivation returned error: %v", err)
	}

	input = benchmarkLayerMatrix(b, 128, 64)
	if output, err = activationLayer.Forward(input); err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if output, err = activationLayer.Forward(input); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = output
}

func Benchmark_ActivationBackward_Softmax_MediumBatch(b *testing.B) {
	var (
		activationLayer *layer.Activation
		input           *matrix.Matrix
		outputGradient  *matrix.Matrix
		inputGradient   *matrix.Matrix
		err             error
		index           int
	)

	activationLayer, err = layer.NewActivation(activation.Softmax{})
	if err != nil {
		b.Fatalf("NewActivation returned error: %v", err)
	}

	input = benchmarkLayerMatrix(b, 128, 64)
	outputGradient = benchmarkLayerMatrix(b, 128, 64)
	if _, err = activationLayer.Forward(input); err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}

	if inputGradient, err = activationLayer.Backward(outputGradient); err != nil {
		b.Fatalf("Backward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if inputGradient, err = activationLayer.Backward(outputGradient); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = inputGradient
}

func Benchmark_DropoutForwardTraining_MediumBatch(b *testing.B) {
	var (
		dropout *layer.Dropout
		input   *matrix.Matrix
		output  *matrix.Matrix
		err     error
		index   int
	)

	dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(7)))
	if err != nil {
		b.Fatalf("NewDropout returned error: %v", err)
	}

	input = benchmarkLayerMatrix(b, 128, 64)
	output, err = dropout.Forward(input)
	if err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		output, err = dropout.Forward(input)
		if err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = output
}

func Benchmark_DropoutBackwardTraining_MediumBatch(b *testing.B) {
	var (
		dropout        *layer.Dropout
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
		index          int
	)

	dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(7)))
	if err != nil {
		b.Fatalf("NewDropout returned error: %v", err)
	}

	input = benchmarkLayerMatrix(b, 128, 64)
	outputGradient = benchmarkLayerMatrix(b, 128, 64)
	if _, err = dropout.Forward(input); err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}

	inputGradient, err = dropout.Backward(outputGradient)
	if err != nil {
		b.Fatalf("Backward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		inputGradient, err = dropout.Backward(outputGradient)
		if err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = inputGradient
}

func Benchmark_BatchNormalizationForwardTraining_MediumBatch(b *testing.B) {
	var (
		batchNorm *layer.BatchNormalization
		input     *matrix.Matrix
		output    *matrix.Matrix
		err       error
		index     int
	)

	batchNorm, err = layer.NewBatchNormalization(64)
	if err != nil {
		b.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	input = benchmarkLayerMatrix(b, 128, 64)
	output, err = batchNorm.Forward(input)
	if err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		output, err = batchNorm.Forward(input)
		if err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = output
}

func Benchmark_BatchNormalizationBackwardTraining_MediumBatch(b *testing.B) {
	var (
		batchNorm      *layer.BatchNormalization
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
		index          int
	)

	batchNorm, err = layer.NewBatchNormalization(64)
	if err != nil {
		b.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	input = benchmarkLayerMatrix(b, 128, 64)
	outputGradient = benchmarkLayerMatrix(b, 128, 64)
	if _, err = batchNorm.Forward(input); err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}

	inputGradient, err = batchNorm.Backward(outputGradient)
	if err != nil {
		b.Fatalf("Backward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		inputGradient, err = batchNorm.Backward(outputGradient)
		if err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = inputGradient
}
