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

func Test_Sequential_RNNTrainBatchDoesNotAllocateAfterWarmUp(t *testing.T) {
	var (
		inputShape    layer.SequenceShape
		config        layer.SimpleRNNConfig
		recurrent     *layer.SimpleRNN
		lastStep      *layer.LastStep
		output        *layer.Dense
		network       *Sequential
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		optimizerRule *optimizer.SGD
		allocations   float64
		err           error
	)

	if inputShape, err = layer.NewSequenceShape(8, 16); err != nil {
		t.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if config, err = layer.NewSimpleRNNConfig(inputShape, 32); err != nil {
		t.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	if recurrent, err = layer.NewSimpleRNN(config, layer.ZeroWeights, layer.ZeroWeights); err != nil {
		t.Fatalf("NewSimpleRNN returned error: %v", err)
	}
	if lastStep, err = layer.NewLastStep(recurrent.OutputShape()); err != nil {
		t.Fatalf("NewLastStep returned error: %v", err)
	}
	if output, err = layer.NewDense(lastStep.OutputSize(), 8, layer.ZeroWeights); err != nil {
		t.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = NewSequential(recurrent, lastStep, output); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	if inputs, err = matrix.New(16, inputShape.Size()); err != nil {
		t.Fatalf("New inputs returned error: %v", err)
	}
	if targets, err = matrix.New(16, output.OutputSize()); err != nil {
		t.Fatalf("New targets returned error: %v", err)
	}
	if optimizerRule, err = optimizer.NewSGD(0.001); err != nil {
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
		t.Fatalf("warmed RNN TrainBatch allocations = %g, want 0", allocations)
	}
}

func Test_Sequential_LengthAwareRNNTrainBatchDoesNotAllocateAfterWarmUp(t *testing.T) {
	var (
		network       *Sequential
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		lengths       *data.SequenceLengths
		optimizerRule *optimizer.SGD
		allocations   float64
		err           error
	)

	network, inputs, targets, lengths, optimizerRule = lengthAwareAllocationFixture(t)
	if _, err = network.TrainBatchWithLengths(
		inputs,
		targets,
		lengths,
		loss.MeanSquaredError{},
		optimizerRule,
	); err != nil {
		t.Fatalf("warm-up TrainBatchWithLengths returned error: %v", err)
	}

	allocations = testing.AllocsPerRun(100, func() {
		allocationTrainMetrics, err = network.TrainBatchWithLengths(
			inputs,
			targets,
			lengths,
			loss.MeanSquaredError{},
			optimizerRule,
		)
		if err != nil {
			panic(err)
		}
	})
	if allocations != 0 {
		t.Fatalf("warmed TrainBatchWithLengths allocations = %g, want 0", allocations)
	}
}

func Test_Sequential_ClippedRNNTrainBatchDoesNotAllocateAfterWarmUp(t *testing.T) {
	var (
		network       *Sequential
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		base          *optimizer.SGD
		optimizerRule *optimizer.GradientClipping
		allocations   float64
		err           error
	)

	network, inputs, targets, base = fixedLengthAllocationFixture(t)
	if optimizerRule, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{
			MaxValue: 0.01,
			MaxNorm:  0.02,
		},
	); err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}
	if _, err = network.TrainBatch(
		inputs,
		targets,
		loss.MeanSquaredError{},
		optimizerRule,
	); err != nil {
		t.Fatalf("warm-up clipped TrainBatch returned error: %v", err)
	}

	allocations = testing.AllocsPerRun(100, func() {
		allocationTrainMetrics, err = network.TrainBatch(
			inputs,
			targets,
			loss.MeanSquaredError{},
			optimizerRule,
		)
		if err != nil {
			panic(err)
		}
	})
	if allocations != 0 {
		t.Fatalf("warmed clipped RNN TrainBatch allocations = %g, want 0", allocations)
	}
}

func Test_Sequential_ClippedLengthAwareRNNTrainBatchDoesNotAllocateAfterWarmUp(
	t *testing.T,
) {
	var (
		network       *Sequential
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		lengths       *data.SequenceLengths
		base          *optimizer.SGD
		optimizerRule *optimizer.GradientClipping
		allocations   float64
		err           error
	)

	network, inputs, targets, lengths, base = lengthAwareAllocationFixture(t)
	if optimizerRule, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{
			MaxValue: 0.01,
			MaxNorm:  0.02,
		},
	); err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}
	if _, err = network.TrainBatchWithLengths(
		inputs,
		targets,
		lengths,
		loss.MeanSquaredError{},
		optimizerRule,
	); err != nil {
		t.Fatalf("warm-up clipped TrainBatchWithLengths returned error: %v", err)
	}

	allocations = testing.AllocsPerRun(100, func() {
		allocationTrainMetrics, err = network.TrainBatchWithLengths(
			inputs,
			targets,
			lengths,
			loss.MeanSquaredError{},
			optimizerRule,
		)
		if err != nil {
			panic(err)
		}
	})
	if allocations != 0 {
		t.Fatalf(
			"warmed clipped TrainBatchWithLengths allocations = %g, want 0",
			allocations,
		)
	}
}

func Test_Sequential_TrainSequenceFitEpochDoesNotAllocateAfterWorkspaceWarmUp(t *testing.T) {
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
				network       *Sequential
				dataset       *data.SequenceDataset
				selector      sequenceLengthLayer
				selectorIndex int
				config        SequenceFitConfig
				scratch       fitScratch
				allocations   float64
				err           error
			)

			network, dataset, config = sequenceFitAllocationFixture(t, tt.shuffle)
			if selector, selectorIndex, err = network.lengthAwareGraph("test"); err != nil {
				t.Fatalf("lengthAwareGraph returned error: %v", err)
			}
			if err = network.trainSequenceFitEpoch(
				dataset,
				config,
				1,
				selector,
				selectorIndex,
				&scratch,
			); err != nil {
				t.Fatalf("warm-up trainSequenceFitEpoch returned error: %v", err)
			}

			allocations = testing.AllocsPerRun(100, func() {
				if err = network.trainSequenceFitEpoch(
					dataset,
					config,
					2,
					selector,
					selectorIndex,
					&scratch,
				); err != nil {
					panic(err)
				}
			})
			if allocations != 0 {
				t.Fatalf(
					"warmed trainSequenceFitEpoch allocations = %g, want 0",
					allocations,
				)
			}
		})
	}
}

func fixedLengthAllocationFixture(
	tb testing.TB,
) (
	network *Sequential,
	inputs,
	targets *matrix.Matrix,
	optimizerRule *optimizer.SGD,
) {
	var (
		inputShape layer.SequenceShape
		config     layer.SimpleRNNConfig
		recurrent  *layer.SimpleRNN
		lastStep   *layer.LastStep
		output     *layer.Dense
		err        error
	)

	tb.Helper()
	if inputShape, err = layer.NewSequenceShape(8, 16); err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if config, err = layer.NewSimpleRNNConfig(inputShape, 32); err != nil {
		tb.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	if recurrent, err = layer.NewSimpleRNN(
		config,
		layer.ZeroWeights,
		layer.ZeroWeights,
	); err != nil {
		tb.Fatalf("NewSimpleRNN returned error: %v", err)
	}
	if lastStep, err = layer.NewLastStep(recurrent.OutputShape()); err != nil {
		tb.Fatalf("NewLastStep returned error: %v", err)
	}
	if output, err = layer.NewDense(lastStep.OutputSize(), 8, layer.ZeroWeights); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = NewSequential(recurrent, lastStep, output); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	if inputs, err = matrix.New(16, inputShape.Size()); err != nil {
		tb.Fatalf("New inputs returned error: %v", err)
	}
	if targets, err = matrix.New(16, output.OutputSize()); err != nil {
		tb.Fatalf("New targets returned error: %v", err)
	}
	if optimizerRule, err = optimizer.NewSGD(0.001); err != nil {
		tb.Fatalf("NewSGD returned error: %v", err)
	}
	return network, inputs, targets, optimizerRule
}

func lengthAwareAllocationFixture(
	tb testing.TB,
) (
	network *Sequential,
	inputs,
	targets *matrix.Matrix,
	lengths *data.SequenceLengths,
	optimizerRule *optimizer.SGD,
) {
	var (
		inputShape   layer.SequenceShape
		config       layer.SimpleRNNConfig
		recurrent    *layer.SimpleRNN
		gather       *layer.GatherLastValid
		output       *layer.Dense
		lengthValues []int
		row          int
		err          error
	)

	tb.Helper()
	if inputShape, err = layer.NewSequenceShape(8, 16); err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if config, err = layer.NewSimpleRNNConfig(inputShape, 32); err != nil {
		tb.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	if recurrent, err = layer.NewSimpleRNN(
		config,
		layer.ZeroWeights,
		layer.ZeroWeights,
	); err != nil {
		tb.Fatalf("NewSimpleRNN returned error: %v", err)
	}
	if gather, err = layer.NewGatherLastValid(recurrent.OutputShape()); err != nil {
		tb.Fatalf("NewGatherLastValid returned error: %v", err)
	}
	if output, err = layer.NewDense(gather.OutputSize(), 8, layer.ZeroWeights); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = NewSequential(recurrent, gather, output); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	if inputs, err = matrix.New(16, inputShape.Size()); err != nil {
		tb.Fatalf("New inputs returned error: %v", err)
	}
	if targets, err = matrix.New(16, output.OutputSize()); err != nil {
		tb.Fatalf("New targets returned error: %v", err)
	}

	lengthValues = make([]int, inputs.Rows())
	for row = range lengthValues {
		lengthValues[row] = 1 + row%inputShape.Steps()
	}
	if lengths, err = data.NewSequenceLengths(inputShape.Steps(), lengthValues); err != nil {
		tb.Fatalf("NewSequenceLengths returned error: %v", err)
	}
	if optimizerRule, err = optimizer.NewSGD(0.001); err != nil {
		tb.Fatalf("NewSGD returned error: %v", err)
	}
	return network, inputs, targets, lengths, optimizerRule
}

func sequenceFitAllocationFixture(
	tb testing.TB,
	shuffle bool,
) (
	network *Sequential,
	dataset *data.SequenceDataset,
	config SequenceFitConfig,
) {
	var (
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		lengths       *data.SequenceLengths
		optimizerRule *optimizer.SGD
		err           error
	)

	tb.Helper()
	network, inputs, targets, lengths, optimizerRule = lengthAwareAllocationFixture(tb)
	if dataset, err = data.NewSequenceDataset(inputs, targets, lengths); err != nil {
		tb.Fatalf("NewSequenceDataset returned error: %v", err)
	}

	config.Epochs = 1
	config.BatchSize = 5
	config.Shuffle = shuffle
	if shuffle {
		config.Random = rand.New(rand.NewSource(17))
	}
	config.Optimizer = optimizerRule
	config.Loss = loss.MeanSquaredError{}
	return network, dataset, config
}
