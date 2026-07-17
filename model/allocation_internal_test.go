package model

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

var allocationParameters []*optimizer.Parameter
var allocationTrainMetrics TrainMetrics

func Test_Sequential_RebuildParametersDoesNotAllocateAfterWarmUp(t *testing.T) {
	var (
		dense       *layer.Dense
		batchNorm   *layer.BatchNormalization
		network     *Sequential
		allocations float64
		err         error
	)

	dense, err = layer.NewDense(2, 2, layer.ZeroWeights)
	if err != nil {
		t.Fatalf("NewDense returned error: %v", err)
	}

	batchNorm, err = layer.NewBatchNormalization(2)
	if err != nil {
		t.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	network, err = NewSequential(dense, batchNorm)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	allocationParameters = network.rebuildParameters()
	allocations = testing.AllocsPerRun(100, func() {
		allocationParameters = network.rebuildParameters()
	})
	if allocations != 0 {
		t.Fatalf("rebuildParameters allocations = %g, want 0", allocations)
	}
}

func Test_Sequential_TrainBatchDoesNotAllocateAfterWarmUp(t *testing.T) {
	var (
		dense         *layer.Dense
		network       *Sequential
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		optimizerRule *optimizer.SGD
		allocations   float64
		err           error
	)

	if dense, err = layer.NewDense(2, 1, layer.ZeroWeights); err != nil {
		t.Fatalf("NewDense returned error: %v", err)
	}

	if network, err = NewSequential(dense); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	if inputs, err = matrix.New(4, 2); err != nil {
		t.Fatalf("New inputs returned error: %v", err)
	}
	if targets, err = matrix.New(4, 1); err != nil {
		t.Fatalf("New targets returned error: %v", err)
	}
	if optimizerRule, err = optimizer.NewSGD(0.01); err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	if _, err = network.TrainBatch(inputs, targets, loss.MeanSquaredError{}, optimizerRule); err != nil {
		t.Fatalf("warm-up TrainBatch returned error: %v", err)
	}

	allocations = testing.AllocsPerRun(100, func() {
		allocationTrainMetrics, err = network.TrainBatch(inputs, targets, loss.MeanSquaredError{}, optimizerRule)
		if err != nil {
			panic(err)
		}
	})
	if allocations != 0 {
		t.Fatalf("warmed TrainBatch allocations = %g, want 0", allocations)
	}
}

func Test_Sequential_TrainBatchAlternatingShapesDoesNotAllocateAfterWarmUp(t *testing.T) {
	var (
		sampleCounts  []int
		inputs        []*matrix.Matrix
		targets       []*matrix.Matrix
		dense         *layer.Dense
		network       *Sequential
		optimizerRule *optimizer.SGD
		allocations   float64
		err           error
		index         int
		shapeIndex    int
	)

	sampleCounts = []int{8, 3, 5, 1}
	inputs = make([]*matrix.Matrix, len(sampleCounts))
	targets = make([]*matrix.Matrix, len(sampleCounts))
	for shapeIndex = range sampleCounts {
		if inputs[shapeIndex], err = matrix.New(sampleCounts[shapeIndex], 2); err != nil {
			t.Fatalf("New inputs returned error: %v", err)
		}
		if targets[shapeIndex], err = matrix.New(sampleCounts[shapeIndex], 1); err != nil {
			t.Fatalf("New targets returned error: %v", err)
		}
	}

	if dense, err = layer.NewDense(2, 1, layer.ZeroWeights); err != nil {
		t.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = NewSequential(dense); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	if optimizerRule, err = optimizer.NewSGD(0.01); err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	for shapeIndex = range sampleCounts {
		if _, err = network.TrainBatch(
			inputs[shapeIndex],
			targets[shapeIndex],
			loss.MeanSquaredError{},
			optimizerRule,
		); err != nil {
			t.Fatalf("warm-up TrainBatch returned error: %v", err)
		}
	}

	allocations = testing.AllocsPerRun(100, func() {
		shapeIndex = index % len(sampleCounts)
		allocationTrainMetrics, err = network.TrainBatch(
			inputs[shapeIndex],
			targets[shapeIndex],
			loss.MeanSquaredError{},
			optimizerRule,
		)
		if err != nil {
			panic(err)
		}
		index++
	})
	if allocations != 0 {
		t.Fatalf("warmed alternating TrainBatch allocations = %g, want 0", allocations)
	}
}

func Test_Sequential_TrainFitEpochDoesNotAllocateAfterWorkspaceWarmUp(t *testing.T) {
	var tests []struct {
		name    string
		shuffle bool
	}

	tests = []struct {
		name    string
		shuffle bool
	}{
		{name: "ordered"},
		{name: "shuffled", shuffle: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				network     *Sequential
				dataset     *data.Dataset
				config      FitConfig
				scratch     fitScratch
				allocations float64
				err         error
			)

			network, dataset, config = fitEpochAllocationFixture(t, tt.shuffle)
			if err = network.trainFitEpoch(dataset, config, 1, &scratch); err != nil {
				t.Fatalf("warm-up trainFitEpoch returned error: %v", err)
			}

			allocations = testing.AllocsPerRun(100, func() {
				if err = network.trainFitEpoch(dataset, config, 2, &scratch); err != nil {
					panic(err)
				}
			})
			if allocations != 0 {
				t.Fatalf("warmed trainFitEpoch allocations = %g, want 0", allocations)
			}
		})
	}
}

func Benchmark_SequentialTrainFitEpoch_Warmed(b *testing.B) {
	var (
		network *Sequential
		dataset *data.Dataset
		config  FitConfig
		scratch fitScratch
		err     error
		index   int
	)

	network, dataset, config = fitEpochAllocationFixture(b, false)
	if err = network.trainFitEpoch(dataset, config, 1, &scratch); err != nil {
		b.Fatalf("warm-up trainFitEpoch returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if err = network.trainFitEpoch(dataset, config, index+2, &scratch); err != nil {
			b.Fatalf("trainFitEpoch returned error: %v", err)
		}
	}
}

func fitEpochAllocationFixture(tb testing.TB, shuffle bool) (network *Sequential, dataset *data.Dataset, config FitConfig) {
	var (
		dense         *layer.Dense
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		optimizerRule *optimizer.SGD
		err           error
	)

	tb.Helper()

	if inputs, err = matrix.New(5, 2); err != nil {
		tb.Fatalf("New inputs returned error: %v", err)
	}
	if targets, err = matrix.New(5, 1); err != nil {
		tb.Fatalf("New targets returned error: %v", err)
	}
	if dataset, err = data.NewDataset(inputs, targets); err != nil {
		tb.Fatalf("NewDataset returned error: %v", err)
	}
	if dense, err = layer.NewDense(2, 1, layer.ZeroWeights); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = NewSequential(dense); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	if optimizerRule, err = optimizer.NewSGD(0.01); err != nil {
		tb.Fatalf("NewSGD returned error: %v", err)
	}

	config.Epochs = 1
	config.BatchSize = 2
	config.Shuffle = shuffle
	if shuffle {
		config.Random = rand.New(rand.NewSource(7))
	}
	config.Optimizer = optimizerRule
	config.Loss = loss.MeanSquaredError{}
	return network, dataset, config
}
