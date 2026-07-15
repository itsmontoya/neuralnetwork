package layer_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Benchmark_DenseForwardBackward_AlternatingShapes(b *testing.B) {
	var (
		dense           *layer.Dense
		inputs          []*matrix.Matrix
		outputGradients []*matrix.Matrix
		result          *matrix.Matrix
		err             error
		index           int
		shapeIndex      int
	)

	dense = benchmarkDense(b, 32, 64)
	inputs = benchmarkAlternatingLayerMatrices(b, 32)
	outputGradients = benchmarkAlternatingLayerMatrices(b, 64)
	for shapeIndex = range inputs {
		if _, err = dense.Forward(inputs[shapeIndex]); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}

		if result, err = dense.Backward(outputGradients[shapeIndex]); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		shapeIndex = index % len(inputs)
		if _, err = dense.Forward(inputs[shapeIndex]); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}

		if result, err = dense.Backward(outputGradients[shapeIndex]); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = result
}

func Benchmark_ActivationForwardBackward_AlternatingShapes(b *testing.B) {
	var (
		activationLayer *layer.Activation
		inputs          []*matrix.Matrix
		outputGradients []*matrix.Matrix
		result          *matrix.Matrix
		err             error
		index           int
		shapeIndex      int
	)

	activationLayer, err = layer.NewActivation(activation.Sigmoid{})
	if err != nil {
		b.Fatalf("NewActivation returned error: %v", err)
	}

	inputs = benchmarkAlternatingLayerMatrices(b, 64)
	outputGradients = benchmarkAlternatingLayerMatrices(b, 64)
	for shapeIndex = range inputs {
		if _, err = activationLayer.Forward(inputs[shapeIndex]); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}

		if result, err = activationLayer.Backward(outputGradients[shapeIndex]); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		shapeIndex = index % len(inputs)
		if _, err = activationLayer.Forward(inputs[shapeIndex]); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}

		if result, err = activationLayer.Backward(outputGradients[shapeIndex]); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = result
}

func Benchmark_ActivationForwardBackward_Softmax_AlternatingShapes(b *testing.B) {
	var (
		activationLayer *layer.Activation
		inputs          []*matrix.Matrix
		outputGradients []*matrix.Matrix
		result          *matrix.Matrix
		err             error
		index           int
		shapeIndex      int
	)

	activationLayer, err = layer.NewActivation(activation.Softmax{})
	if err != nil {
		b.Fatalf("NewActivation returned error: %v", err)
	}

	inputs = benchmarkAlternatingLayerMatrices(b, 64)
	outputGradients = benchmarkAlternatingLayerMatrices(b, 64)
	for shapeIndex = range inputs {
		if _, err = activationLayer.Forward(inputs[shapeIndex]); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}

		if result, err = activationLayer.Backward(outputGradients[shapeIndex]); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		shapeIndex = index % len(inputs)
		if _, err = activationLayer.Forward(inputs[shapeIndex]); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}

		if result, err = activationLayer.Backward(outputGradients[shapeIndex]); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = result
}

func Benchmark_DropoutForwardBackward_AlternatingShapes(b *testing.B) {
	var (
		dropout         *layer.Dropout
		inputs          []*matrix.Matrix
		outputGradients []*matrix.Matrix
		result          *matrix.Matrix
		err             error
		index           int
		shapeIndex      int
	)

	dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(7)))
	if err != nil {
		b.Fatalf("NewDropout returned error: %v", err)
	}

	inputs = benchmarkAlternatingLayerMatrices(b, 64)
	outputGradients = benchmarkAlternatingLayerMatrices(b, 64)
	for shapeIndex = range inputs {
		if _, err = dropout.Forward(inputs[shapeIndex]); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}

		if result, err = dropout.Backward(outputGradients[shapeIndex]); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		shapeIndex = index % len(inputs)
		if _, err = dropout.Forward(inputs[shapeIndex]); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}

		if result, err = dropout.Backward(outputGradients[shapeIndex]); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = result
}

func Benchmark_BatchNormalizationForwardBackward_AlternatingShapes(b *testing.B) {
	var (
		batchNorm       *layer.BatchNormalization
		inputs          []*matrix.Matrix
		outputGradients []*matrix.Matrix
		result          *matrix.Matrix
		err             error
		index           int
		shapeIndex      int
	)

	batchNorm, err = layer.NewBatchNormalization(64)
	if err != nil {
		b.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	inputs = benchmarkAlternatingLayerMatrices(b, 64)
	outputGradients = benchmarkAlternatingLayerMatrices(b, 64)
	for shapeIndex = range inputs {
		if _, err = batchNorm.Forward(inputs[shapeIndex]); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}

		if result, err = batchNorm.Backward(outputGradients[shapeIndex]); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		shapeIndex = index % len(inputs)
		if _, err = batchNorm.Forward(inputs[shapeIndex]); err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}

		if result, err = batchNorm.Backward(outputGradients[shapeIndex]); err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkScratchLayerResult = result
}

func benchmarkAlternatingLayerMatrices(tb testing.TB, cols int) (matrices []*matrix.Matrix) {
	var (
		rows  []int
		index int
	)

	tb.Helper()

	rows = []int{128, 17, 1024, 257}
	matrices = make([]*matrix.Matrix, len(rows))
	for index = range rows {
		matrices[index] = benchmarkLayerMatrix(tb, rows[index], cols)
	}

	return matrices
}
