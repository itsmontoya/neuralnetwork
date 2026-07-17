package model_test

import (
	"bytes"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/model"
)

var benchmarkSerializedModel *model.Sequential
var benchmarkSerializedBytes []byte

func Benchmark_SequentialSave_ColdPath(b *testing.B) {
	var (
		network *model.Sequential
		buffer  bytes.Buffer
		err     error
		index   int
	)

	network = benchmarkSerializationModel(b)
	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		buffer.Reset()
		if err = network.Save(&buffer); err != nil {
			b.Fatalf("Save returned error: %v", err)
		}
	}

	benchmarkSerializedBytes = buffer.Bytes()
}

func Benchmark_LoadSequential_ColdPath(b *testing.B) {
	var (
		network  *model.Sequential
		loaded   *model.Sequential
		buffer   bytes.Buffer
		reader   *bytes.Reader
		document []byte
		err      error
		index    int
	)

	network = benchmarkSerializationModel(b)
	if err = network.Save(&buffer); err != nil {
		b.Fatalf("Save returned error: %v", err)
	}

	document = append(document, buffer.Bytes()...)
	reader = bytes.NewReader(document)
	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		reader.Reset(document)
		loaded, err = model.LoadSequential(reader)
		if err != nil {
			b.Fatalf("LoadSequential returned error: %v", err)
		}
	}

	benchmarkSerializedModel = loaded
}

func benchmarkSerializationModel(tb testing.TB) (network *model.Sequential) {
	var (
		inputLayer      *layer.Dense
		batchNorm       *layer.BatchNormalization
		activationLayer *layer.Activation
		outputLayer     *layer.Dense
		err             error
	)

	tb.Helper()

	if inputLayer, err = layer.NewDense(32, 64, layer.ZeroWeights); err != nil {
		tb.Fatalf("NewDense input layer returned error: %v", err)
	}

	if batchNorm, err = layer.NewBatchNormalization(64); err != nil {
		tb.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	if activationLayer, err = layer.NewActivation(activation.ReLU{}); err != nil {
		tb.Fatalf("NewActivation returned error: %v", err)
	}

	if outputLayer, err = layer.NewDense(64, 8, layer.ZeroWeights); err != nil {
		tb.Fatalf("NewDense output layer returned error: %v", err)
	}

	if network, err = model.NewSequential(inputLayer, batchNorm, activationLayer, outputLayer); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}

	return network
}
