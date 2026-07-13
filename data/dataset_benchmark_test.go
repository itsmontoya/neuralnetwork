package data_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var benchmarkBatches []*data.Batch
var benchmarkBatchMatrix *matrix.Matrix

func Benchmark_DatasetBatches_Unshuffled(b *testing.B) {
	var (
		dataset *data.Dataset
		batches []*data.Batch
		err     error
		index   int
	)

	dataset = benchmarkDataset(b, 1024, 32, 8)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		batches, err = dataset.Batches(64, nil)
		if err != nil {
			b.Fatalf("Batches returned error: %v", err)
		}
	}

	benchmarkBatches = batches
}

func Benchmark_DatasetBatches_Shuffled(b *testing.B) {
	var (
		random  *rand.Rand
		dataset *data.Dataset
		batches []*data.Batch
		err     error
		index   int
	)

	random = rand.New(rand.NewSource(7))
	dataset = benchmarkDataset(b, 1024, 32, 8)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		batches, err = dataset.Batches(64, random)
		if err != nil {
			b.Fatalf("Batches returned error: %v", err)
		}
	}

	benchmarkBatches = batches
}

func Benchmark_BatchInputs(b *testing.B) {
	var (
		batch  *data.Batch
		inputs *matrix.Matrix
		err    error
		index  int
	)

	batch = benchmarkBatch(b)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		inputs, err = batch.Inputs()
		if err != nil {
			b.Fatalf("Inputs returned error: %v", err)
		}
	}

	benchmarkBatchMatrix = inputs
}

func Benchmark_BatchTargets(b *testing.B) {
	var (
		batch   *data.Batch
		targets *matrix.Matrix
		err     error
		index   int
	)

	batch = benchmarkBatch(b)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		targets, err = batch.Targets()
		if err != nil {
			b.Fatalf("Targets returned error: %v", err)
		}
	}

	benchmarkBatchMatrix = targets
}

func benchmarkBatch(tb testing.TB) (batch *data.Batch) {
	var (
		dataset *data.Dataset
		batches []*data.Batch
		err     error
	)

	tb.Helper()

	dataset = benchmarkDataset(tb, 64, 32, 8)
	batches, err = dataset.Batches(64, nil)
	if err != nil {
		tb.Fatalf("Batches returned error: %v", err)
	}

	batch = batches[0]
	return batch
}

func benchmarkDataset(tb testing.TB, samples, inputSize, targetSize int) (dataset *data.Dataset) {
	var (
		inputValues  []float32
		targetValues []float32
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		row          int
		col          int
		err          error
	)

	tb.Helper()

	inputValues = make([]float32, samples*inputSize)
	targetValues = make([]float32, samples*targetSize)

	for row = 0; row < samples; row++ {
		for col = 0; col < inputSize; col++ {
			inputValues[row*inputSize+col] = float32(row+col) / float32(inputSize)
		}

		for col = 0; col < targetSize; col++ {
			targetValues[row*targetSize+col] = float32(row-col) / float32(targetSize)
		}
	}

	if inputs, err = matrix.FromSlice(samples, inputSize, inputValues); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	if targets, err = matrix.FromSlice(samples, targetSize, targetValues); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	dataset, err = data.NewDataset(inputs, targets)
	if err != nil {
		tb.Fatalf("NewDataset returned error: %v", err)
	}

	return dataset
}
