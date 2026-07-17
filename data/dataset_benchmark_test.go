package data_test

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var benchmarkBatches []*data.Batch
var benchmarkBatchMatrix *matrix.Matrix
var benchmarkDatasetResult *data.Dataset

func Benchmark_LoadCSV_ColdPath(b *testing.B) {
	var (
		csvData string
		reader  *strings.Reader
		config  data.CSVConfig
		dataset *data.Dataset
		err     error
		index   int
	)

	csvData = benchmarkCSVData(1024, 32, 8)
	reader = strings.NewReader(csvData)
	config.InputColumns = 32
	config.TargetColumns = 8

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		reader.Reset(csvData)
		dataset, err = data.LoadCSV(reader, config)
		if err != nil {
			b.Fatalf("LoadCSV returned error: %v", err)
		}
	}

	benchmarkDatasetResult = dataset
}

func Benchmark_DatasetSplit_ColdPath(b *testing.B) {
	var (
		dataset *data.Dataset
		train   *data.Dataset
		test    *data.Dataset
		err     error
		index   int
	)

	dataset = benchmarkDataset(b, 1024, 32, 8)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		train, test, err = dataset.Split(0.2, nil)
		if err != nil {
			b.Fatalf("Split returned error: %v", err)
		}
	}

	benchmarkDatasetResult = train
	if test == nil {
		b.Fatal("Split returned nil test dataset")
	}
}

func Benchmark_NewDataset_ColdPath(b *testing.B) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		dataset *data.Dataset
		err     error
		index   int
	)

	inputs = benchmarkMatrix(b, 1024, 32)
	targets = benchmarkMatrix(b, 1024, 8)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		dataset, err = data.NewDataset(inputs, targets)
		if err != nil {
			b.Fatalf("NewDataset returned error: %v", err)
		}
	}

	benchmarkDatasetResult = dataset
}

func Benchmark_DatasetCopyAccessors_ColdPath(b *testing.B) {
	var dataset *data.Dataset

	dataset = benchmarkDataset(b, 1024, 32, 8)
	b.Run("Inputs", func(b *testing.B) {
		var (
			inputs *matrix.Matrix
			err    error
			index  int
		)

		b.ReportAllocs()
		b.ResetTimer()

		for index = 0; index < b.N; index++ {
			inputs, err = dataset.Inputs()
			if err != nil {
				b.Fatalf("Inputs returned error: %v", err)
			}
		}

		benchmarkBatchMatrix = inputs
	})
	b.Run("Targets", func(b *testing.B) {
		var (
			targets *matrix.Matrix
			err     error
			index   int
		)

		b.ReportAllocs()
		b.ResetTimer()

		for index = 0; index < b.N; index++ {
			targets, err = dataset.Targets()
			if err != nil {
				b.Fatalf("Targets returned error: %v", err)
			}
		}

		benchmarkBatchMatrix = targets
	})
}

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

func benchmarkMatrix(tb testing.TB, rows, cols int) (m *matrix.Matrix) {
	var (
		values []float32
		err    error
	)

	tb.Helper()

	values = make([]float32, rows*cols)
	if m, err = matrix.FromSlice(rows, cols, values); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}

func benchmarkCSVData(samples, inputSize, targetSize int) (csvData string) {
	var (
		builder strings.Builder
		row     int
		column  int
		columns int
	)

	columns = inputSize + targetSize
	builder.Grow(samples * columns * 2)
	for row = 0; row < samples; row++ {
		for column = 0; column < columns; column++ {
			if column > 0 {
				builder.WriteByte(',')
			}

			builder.WriteByte('1')
		}

		builder.WriteByte('\n')
	}

	csvData = builder.String()
	return csvData
}
