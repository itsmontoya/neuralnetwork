package model_test

import (
	"bytes"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_Sequential_GradientClippingBoundsFixedLengthRNNUpdate(t *testing.T) {
	var (
		clippedNetwork      *model.Sequential
		clippedRecurrent    *layer.SimpleRNN
		repeatedNetwork     *model.Sequential
		repeatedRecurrent   *layer.SimpleRNN
		unwrappedNetwork    *model.Sequential
		unwrappedRecurrent  *layer.SimpleRNN
		input               *matrix.Matrix
		targets             *matrix.Matrix
		clippedBase         *optimizer.SGD
		repeatedBase        *optimizer.SGD
		unwrappedOptimizer  *optimizer.SGD
		clipping            *optimizer.GradientClipping
		repeatedClipping    *optimizer.GradientClipping
		observation         optimizer.GradientClippingObservation
		repeatedObservation optimizer.GradientClippingObservation
		available           bool
		clippedNorm         float64
		wantNorm            float64
		wantScale           float64
		err                 error
	)

	clippedNetwork, clippedRecurrent = mustFixedGradientClippingRNN(t, 2)
	repeatedNetwork, repeatedRecurrent = mustFixedGradientClippingRNN(t, 2)
	unwrappedNetwork, unwrappedRecurrent = mustFixedGradientClippingRNN(t, 2)
	input = mustMatrix(t, 1, 2, []float32{2, 3})
	targets = mustMatrix(t, 1, 1, []float32{100})

	if clippedBase, err = optimizer.NewSGD(0.1); err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}
	if clipping, err = optimizer.NewGradientClipping(
		clippedBase,
		optimizer.GradientClippingConfig{
			MaxValue: 300,
			MaxNorm:  10,
		},
	); err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}
	if _, err = clippedNetwork.TrainBatch(
		input,
		targets,
		loss.MeanSquaredError{},
		clipping,
	); err != nil {
		t.Fatalf("clipped TrainBatch returned error: %v", err)
	}

	// Forward is zero. MSE contributes -200 at the final step, so full
	// recurrent backward produces gradients [-600, 0, -200]. Value clipping
	// produces [-300, 0, -200], whose norm is 100*sqrt(13).
	wantNorm = 100 * math.Sqrt(13)
	wantScale = 10 / wantNorm
	observation, available = clipping.Observation()
	if !available {
		t.Fatal("clipping observation unavailable after TrainBatch")
	}
	if !observation.ValueClipped || !observation.BaseUpdateCompleted {
		t.Fatalf("clipping observation = %+v, want applied successful update", observation)
	}
	requireFloat64AlmostEqual(t, observation.GlobalNorm, wantNorm, 1e-12)
	requireFloat64AlmostEqual(t, observation.Scale, wantScale, 1e-12)

	requireMatrixValues(
		t,
		clippedRecurrent.InputWeights().Values(),
		[]float32{float32(30 * wantScale)},
	)
	requireMatrixValues(t, clippedRecurrent.RecurrentWeights().Values(), []float32{0})
	requireMatrixValues(
		t,
		clippedRecurrent.Biases().Values(),
		[]float32{float32(20 * wantScale)},
	)
	requireMatrixValues(t, clippedRecurrent.InputWeights().Gradient(), []float32{0})
	requireMatrixValues(t, clippedRecurrent.RecurrentWeights().Gradient(), []float32{0})
	requireMatrixValues(t, clippedRecurrent.Biases().Gradient(), []float32{0})

	clippedNorm = math.Hypot(
		float64(mustValues(t, clippedRecurrent.InputWeights().Values())[0]/0.1),
		float64(mustValues(t, clippedRecurrent.Biases().Values())[0]/0.1),
	)
	requireFloat64AlmostEqual(t, clippedNorm, 10, 1e-5)

	if unwrappedOptimizer, err = optimizer.NewSGD(0.1); err != nil {
		t.Fatalf("unwrapped NewSGD returned error: %v", err)
	}
	if _, err = unwrappedNetwork.TrainBatch(
		input,
		targets,
		loss.MeanSquaredError{},
		unwrappedOptimizer,
	); err != nil {
		t.Fatalf("unwrapped TrainBatch returned error: %v", err)
	}
	requireMatrixValues(t, unwrappedRecurrent.InputWeights().Values(), []float32{60})
	requireMatrixValues(t, unwrappedRecurrent.RecurrentWeights().Values(), []float32{0})
	requireMatrixValues(t, unwrappedRecurrent.Biases().Values(), []float32{20})

	if repeatedBase, err = optimizer.NewSGD(0.1); err != nil {
		t.Fatalf("repeated NewSGD returned error: %v", err)
	}
	if repeatedClipping, err = optimizer.NewGradientClipping(
		repeatedBase,
		clipping.Config(),
	); err != nil {
		t.Fatalf("repeated NewGradientClipping returned error: %v", err)
	}
	if _, err = repeatedNetwork.TrainBatch(
		input,
		targets,
		loss.MeanSquaredError{},
		repeatedClipping,
	); err != nil {
		t.Fatalf("repeated TrainBatch returned error: %v", err)
	}
	repeatedObservation, available = repeatedClipping.Observation()
	if !available || repeatedObservation != observation {
		t.Fatalf(
			"repeated observation = %+v/%t, want %+v/true",
			repeatedObservation,
			available,
			observation,
		)
	}
	requireMatrixValues(
		t,
		repeatedRecurrent.InputWeights().Values(),
		mustValues(t, clippedRecurrent.InputWeights().Values()),
	)
	requireMatrixValues(
		t,
		repeatedRecurrent.RecurrentWeights().Values(),
		mustValues(t, clippedRecurrent.RecurrentWeights().Values()),
	)
	requireMatrixValues(
		t,
		repeatedRecurrent.Biases().Values(),
		mustValues(t, clippedRecurrent.Biases().Values()),
	)
}

func Test_Sequential_GradientClippingBoundsMixedLengthRNNUpdate(t *testing.T) {
	var (
		paddedNetwork     *model.Sequential
		paddedRecurrent   *layer.SimpleRNN
		cleanNetwork      *model.Sequential
		cleanRecurrent    *layer.SimpleRNN
		paddedInput       *matrix.Matrix
		cleanInput        *matrix.Matrix
		targets           *matrix.Matrix
		lengths           *data.SequenceLengths
		paddedBase        *optimizer.SGD
		cleanBase         *optimizer.SGD
		paddedClipping    *optimizer.GradientClipping
		cleanClipping     *optimizer.GradientClipping
		paddedObservation optimizer.GradientClippingObservation
		cleanObservation  optimizer.GradientClippingObservation
		available         bool
		wantNorm          float64
		wantScale         float64
		err               error
	)

	paddedNetwork, paddedRecurrent = mustLengthAwareGradientClippingRNN(t, 3)
	cleanNetwork, cleanRecurrent = mustLengthAwareGradientClippingRNN(t, 3)
	paddedInput = mustMatrix(t, 2, 3, []float32{
		2, 99, -99,
		1, 2, 3,
	})
	cleanInput = mustMatrix(t, 2, 3, []float32{
		2, 0, 0,
		1, 2, 3,
	})
	targets = mustMatrix(t, 2, 1, []float32{100, 100})
	lengths = mustSequenceLengths(t, 3, []int{1, 3})

	if paddedBase, err = optimizer.NewSGD(0.1); err != nil {
		t.Fatalf("padded NewSGD returned error: %v", err)
	}
	if paddedClipping, err = optimizer.NewGradientClipping(
		paddedBase,
		optimizer.GradientClippingConfig{MaxNorm: 5},
	); err != nil {
		t.Fatalf("padded NewGradientClipping returned error: %v", err)
	}
	if _, err = paddedNetwork.TrainBatchWithLengths(
		paddedInput,
		targets,
		lengths,
		loss.MeanSquaredError{},
		paddedClipping,
	); err != nil {
		t.Fatalf("padded TrainBatchWithLengths returned error: %v", err)
	}

	// The selected rows contribute -100 prediction gradients. Gather routes
	// them to steps zero and two, producing [-500, 0, -200]. The padded
	// suffix in the first row contributes nothing to this parameter update.
	wantNorm = 100 * math.Sqrt(29)
	wantScale = 5 / wantNorm
	paddedObservation, available = paddedClipping.Observation()
	if !available {
		t.Fatal("padded clipping observation unavailable")
	}
	if paddedObservation.ValueClipped || !paddedObservation.BaseUpdateCompleted {
		t.Fatalf("padded clipping observation = %+v, want norm-only success", paddedObservation)
	}
	requireFloat64AlmostEqual(t, paddedObservation.GlobalNorm, wantNorm, 1e-12)
	requireFloat64AlmostEqual(t, paddedObservation.Scale, wantScale, 1e-12)
	requireMatrixValues(
		t,
		paddedRecurrent.InputWeights().Values(),
		[]float32{float32(50 * wantScale)},
	)
	requireMatrixValues(t, paddedRecurrent.RecurrentWeights().Values(), []float32{0})
	requireMatrixValues(
		t,
		paddedRecurrent.Biases().Values(),
		[]float32{float32(20 * wantScale)},
	)

	if cleanBase, err = optimizer.NewSGD(0.1); err != nil {
		t.Fatalf("clean NewSGD returned error: %v", err)
	}
	if cleanClipping, err = optimizer.NewGradientClipping(
		cleanBase,
		paddedClipping.Config(),
	); err != nil {
		t.Fatalf("clean NewGradientClipping returned error: %v", err)
	}
	if _, err = cleanNetwork.TrainBatchWithLengths(
		cleanInput,
		targets,
		lengths,
		loss.MeanSquaredError{},
		cleanClipping,
	); err != nil {
		t.Fatalf("clean TrainBatchWithLengths returned error: %v", err)
	}
	cleanObservation, available = cleanClipping.Observation()
	if !available || cleanObservation != paddedObservation {
		t.Fatalf(
			"clean observation = %+v/%t, want %+v/true",
			cleanObservation,
			available,
			paddedObservation,
		)
	}
	requireMatrixValues(
		t,
		cleanRecurrent.InputWeights().Values(),
		mustValues(t, paddedRecurrent.InputWeights().Values()),
	)
	requireMatrixValues(
		t,
		cleanRecurrent.RecurrentWeights().Values(),
		mustValues(t, paddedRecurrent.RecurrentWeights().Values()),
	)
	requireMatrixValues(
		t,
		cleanRecurrent.Biases().Values(),
		mustValues(t, paddedRecurrent.Biases().Values()),
	)
}

func Test_Sequential_GradientClippingWorksThroughOrdinaryFitControls(t *testing.T) {
	var (
		parameterOne   *optimizer.Parameter
		parameterTwo   *optimizer.Parameter
		parameterThree *optimizer.Parameter
		network        *model.Sequential
		input          *matrix.Matrix
		targets        *matrix.Matrix
		dataset        *data.Dataset
		base           *recordingOptimizer
		clipping       *optimizer.GradientClipping
		schedule       *recordingSchedule
		history        model.TrainingHistory
		observation    optimizer.GradientClippingObservation
		available      bool
		err            error
	)

	parameterOne = mustParameter(t, []float32{1})
	parameterTwo = mustParameter(t, []float32{2})
	parameterThree = mustParameter(t, []float32{3})
	mustAccumulateGradient(t, parameterOne, []float32{2})
	mustAccumulateGradient(t, parameterTwo, []float32{-3})
	mustAccumulateGradient(t, parameterThree, []float32{0.5})
	if network, err = model.NewSequential(
		&parameterLayer{parameters: []*optimizer.Parameter{parameterOne, parameterTwo}},
		&parameterLayer{parameters: []*optimizer.Parameter{parameterThree}},
	); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	input = mustMatrix(t, 1, 1, []float32{1})
	targets = mustMatrix(t, 1, 1, []float32{1})
	base = &recordingOptimizer{learningRate: 0.1}
	if clipping, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{MaxValue: 1},
	); err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}

	if _, err = network.TrainBatch(
		input,
		targets,
		loss.MeanSquaredError{},
		clipping,
	); err != nil {
		t.Fatalf("TrainBatch returned error: %v", err)
	}
	if len(base.parameterCalls) != 1 || len(base.parameterCalls[0]) != 3 {
		t.Fatalf("base parameter calls = %v, want one three-parameter call", base.parameterCalls)
	}
	if base.parameterCalls[0][0] != parameterOne ||
		base.parameterCalls[0][1] != parameterTwo ||
		base.parameterCalls[0][2] != parameterThree {
		t.Fatal("clipped TrainBatch did not preserve layer and parameter order")
	}
	requireMatrixValues(t, parameterOne.Gradient(), []float32{1})
	requireMatrixValues(t, parameterTwo.Gradient(), []float32{-1})
	requireMatrixValues(t, parameterThree.Gradient(), []float32{0.5})

	if dataset, err = data.NewDataset(input, targets); err != nil {
		t.Fatalf("NewDataset returned error: %v", err)
	}
	schedule = &recordingSchedule{rates: []float32{0.05, 0.025}}
	if history, err = network.Fit(dataset, model.FitConfig{
		Epochs:               2,
		BatchSize:            1,
		Optimizer:            clipping,
		LearningRateSchedule: schedule,
		Loss:                 loss.MeanSquaredError{},
	}); err != nil {
		t.Fatalf("Fit returned error: %v", err)
	}
	requireEpochCount(t, history, 2)
	requireInts(t, schedule.epochs, []int{1, 2})
	testutil.RequireSliceAlmostEqual(t, base.setRates, []float32{0.05, 0.025}, epsilon)
	testutil.RequireAlmostEqual(t, clipping.LearningRate(), 0.025, epsilon)

	observation, available = clipping.Observation()
	if !available || !observation.BaseUpdateCompleted {
		t.Fatalf("Fit clipping observation = %+v/%t, want successful observation", observation, available)
	}
}

func Test_Sequential_GradientClippingWorksThroughLengthAwareFitControls(t *testing.T) {
	var (
		network     *model.Sequential
		inputs      *matrix.Matrix
		targets     *matrix.Matrix
		lengths     *data.SequenceLengths
		dataset     *data.SequenceDataset
		base        *optimizer.SGD
		clipping    *optimizer.GradientClipping
		schedule    *recordingSchedule
		history     model.TrainingHistory
		observation optimizer.GradientClippingObservation
		available   bool
		err         error
	)

	network, _ = mustLengthAwareGradientClippingRNN(t, 3)
	inputs = mustMatrix(t, 2, 3, []float32{
		2, 90, -90,
		1, 2, 3,
	})
	targets = mustMatrix(t, 2, 1, []float32{100, 100})
	lengths = mustSequenceLengths(t, 3, []int{1, 3})
	if dataset, err = data.NewSequenceDataset(inputs, targets, lengths); err != nil {
		t.Fatalf("NewSequenceDataset returned error: %v", err)
	}
	if base, err = optimizer.NewSGD(0.1); err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}
	if clipping, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{MaxNorm: 5},
	); err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}
	schedule = &recordingSchedule{rates: []float32{0.1, 0.05}}
	if history, err = network.FitWithLengths(dataset, model.SequenceFitConfig{
		Epochs:               2,
		BatchSize:            2,
		Optimizer:            clipping,
		LearningRateSchedule: schedule,
		Loss:                 loss.MeanSquaredError{},
	}); err != nil {
		t.Fatalf("FitWithLengths returned error: %v", err)
	}

	requireEpochCount(t, history, 2)
	requireInts(t, schedule.epochs, []int{1, 2})
	testutil.RequireAlmostEqual(t, base.LearningRate(), 0.05, epsilon)
	observation, available = clipping.Observation()
	if !available || !observation.BaseUpdateCompleted || !(observation.Scale < 1) {
		t.Fatalf(
			"FitWithLengths clipping observation = %+v/%t, want applied success",
			observation,
			available,
		)
	}
}

func Test_Sequential_GradientClippingFailuresPreserveTrainingLifecycle(t *testing.T) {
	t.Run("ordinary", func(t *testing.T) {
		var (
			parameter   *optimizer.Parameter
			mode        *modeLayer
			network     *model.Sequential
			base        *recordingOptimizer
			clipping    *optimizer.GradientClipping
			observation optimizer.GradientClippingObservation
			available   bool
			err         error
		)

		parameter = mustParameter(t, []float32{1})
		mustAccumulateGradient(t, parameter, []float32{float32(math.NaN())})
		mode = &modeLayer{}
		if network, err = model.NewSequential(
			mode,
			&parameterLayer{parameters: []*optimizer.Parameter{parameter}},
		); err != nil {
			t.Fatalf("NewSequential returned error: %v", err)
		}
		if err = network.SetTraining(false); err != nil {
			t.Fatalf("SetTraining returned error: %v", err)
		}
		base = &recordingOptimizer{}
		if clipping, err = optimizer.NewGradientClipping(
			base,
			optimizer.GradientClippingConfig{MaxNorm: 1},
		); err != nil {
			t.Fatalf("NewGradientClipping returned error: %v", err)
		}

		if _, err = network.TrainBatch(
			mustMatrix(t, 1, 1, []float32{1}),
			mustMatrix(t, 1, 1, []float32{1}),
			loss.MeanSquaredError{},
			clipping,
		); err == nil {
			t.Fatal("TrainBatch error = nil, want clipping error")
		} else if !strings.Contains(
			err.Error(),
			"model: optimizer update failed: optimizer: gradient clipping non-finite gradient",
		) {
			t.Fatalf("TrainBatch error = %q, want optimizer clipping context", err)
		}
		if network.Training() || mode.modes[len(mode.modes)-1] {
			t.Fatal("TrainBatch did not restore ordinary training mode")
		}
		if base.updateCalls != 0 {
			t.Fatalf("base update calls = %d, want 0", base.updateCalls)
		}
		observation, available = clipping.Observation()
		if available || observation != (optimizer.GradientClippingObservation{}) {
			t.Fatalf("failed clipping observation = %+v/%t, want unavailable", observation, available)
		}
	})

	t.Run("length aware", func(t *testing.T) {
		var (
			parameter   *optimizer.Parameter
			mode        *modeLayer
			gather      *layer.GatherLastValid
			network     *model.Sequential
			base        *recordingOptimizer
			clipping    *optimizer.GradientClipping
			observation optimizer.GradientClippingObservation
			available   bool
			err         error
		)

		parameter = mustParameter(t, []float32{1})
		mustAccumulateGradient(t, parameter, []float32{float32(math.Inf(1))})
		mode = &modeLayer{}
		gather = mustGatherLastValid(t, 2, 1)
		if network, err = model.NewSequential(
			gather,
			mode,
			&parameterLayer{parameters: []*optimizer.Parameter{parameter}},
		); err != nil {
			t.Fatalf("NewSequential returned error: %v", err)
		}
		if err = network.SetTraining(false); err != nil {
			t.Fatalf("SetTraining returned error: %v", err)
		}
		base = &recordingOptimizer{}
		if clipping, err = optimizer.NewGradientClipping(
			base,
			optimizer.GradientClippingConfig{MaxValue: 1},
		); err != nil {
			t.Fatalf("NewGradientClipping returned error: %v", err)
		}

		if _, err = network.TrainBatchWithLengths(
			mustMatrix(t, 1, 2, []float32{1, 2}),
			mustMatrix(t, 1, 1, []float32{1}),
			mustSequenceLengths(t, 2, []int{1}),
			loss.MeanSquaredError{},
			clipping,
		); err == nil {
			t.Fatal("TrainBatchWithLengths error = nil, want clipping error")
		} else if !strings.Contains(
			err.Error(),
			"model: length-aware optimizer update failed: optimizer: gradient clipping non-finite gradient",
		) {
			t.Fatalf(
				"TrainBatchWithLengths error = %q, want length-aware clipping context",
				err,
			)
		}
		if network.Training() || mode.modes[len(mode.modes)-1] {
			t.Fatal("TrainBatchWithLengths did not restore training mode")
		}
		if base.updateCalls != 0 {
			t.Fatalf("base update calls = %d, want 0", base.updateCalls)
		}
		observation, available = clipping.Observation()
		if available || observation != (optimizer.GradientClippingObservation{}) {
			t.Fatalf("failed clipping observation = %+v/%t, want unavailable", observation, available)
		}
		if _, err = network.BackwardWithLengths(
			mustMatrix(t, 1, 1, []float32{1}),
		); err == nil {
			t.Fatal("BackwardWithLengths after clipping error = nil, want cleared association")
		}
	})
}

func Test_Sequential_GradientClippingBaseErrorsPreserveFitHistory(t *testing.T) {
	t.Run("ordinary", func(t *testing.T) {
		var (
			updateErr   error
			mode        *modeLayer
			network     *model.Sequential
			dataset     *data.Dataset
			base        *recordingOptimizer
			clipping    *optimizer.GradientClipping
			history     model.TrainingHistory
			observation optimizer.GradientClippingObservation
			available   bool
			err         error
		)

		updateErr = errors.New("clipped base update failed")
		mode = &modeLayer{}
		if network, err = model.NewSequential(mode); err != nil {
			t.Fatalf("NewSequential returned error: %v", err)
		}
		if err = network.SetTraining(false); err != nil {
			t.Fatalf("SetTraining returned error: %v", err)
		}
		if dataset, err = data.NewDataset(
			mustMatrix(t, 1, 1, []float32{1}),
			mustMatrix(t, 1, 1, []float32{1}),
		); err != nil {
			t.Fatalf("NewDataset returned error: %v", err)
		}
		base = &recordingOptimizer{updateErr: updateErr}
		if clipping, err = optimizer.NewGradientClipping(
			base,
			optimizer.GradientClippingConfig{MaxNorm: 1},
		); err != nil {
			t.Fatalf("NewGradientClipping returned error: %v", err)
		}

		history, err = network.Fit(dataset, model.FitConfig{
			Epochs:    2,
			BatchSize: 1,
			Optimizer: clipping,
			Loss:      loss.MeanSquaredError{},
		})
		if !errors.Is(err, updateErr) {
			t.Fatalf("Fit error = %v, want %v", err, updateErr)
		}
		if !strings.Contains(
			err.Error(),
			"model: epoch 1 train batch failed: model: optimizer update failed",
		) {
			t.Fatalf("Fit error = %q, want epoch optimizer context", err)
		}
		requireEpochCount(t, history, 0)
		if network.Training() || mode.modes[len(mode.modes)-1] {
			t.Fatal("Fit did not restore ordinary training mode")
		}
		observation, available = clipping.Observation()
		if !available || observation.BaseUpdateCompleted {
			t.Fatalf("base failure observation = %+v/%t, want failed base", observation, available)
		}
	})

	t.Run("length aware", func(t *testing.T) {
		var (
			updateErr   error
			mode        *modeLayer
			gather      *layer.GatherLastValid
			network     *model.Sequential
			dataset     *data.SequenceDataset
			base        *recordingOptimizer
			clipping    *optimizer.GradientClipping
			history     model.TrainingHistory
			observation optimizer.GradientClippingObservation
			available   bool
			err         error
		)

		updateErr = errors.New("length-aware clipped base update failed")
		mode = &modeLayer{}
		gather = mustGatherLastValid(t, 2, 1)
		if network, err = model.NewSequential(gather, mode); err != nil {
			t.Fatalf("NewSequential returned error: %v", err)
		}
		if err = network.SetTraining(false); err != nil {
			t.Fatalf("SetTraining returned error: %v", err)
		}
		if dataset, err = data.NewSequenceDataset(
			mustMatrix(t, 1, 2, []float32{1, 2}),
			mustMatrix(t, 1, 1, []float32{1}),
			mustSequenceLengths(t, 2, []int{1}),
		); err != nil {
			t.Fatalf("NewSequenceDataset returned error: %v", err)
		}
		base = &recordingOptimizer{updateErr: updateErr}
		if clipping, err = optimizer.NewGradientClipping(
			base,
			optimizer.GradientClippingConfig{MaxValue: 1},
		); err != nil {
			t.Fatalf("NewGradientClipping returned error: %v", err)
		}

		history, err = network.FitWithLengths(dataset, model.SequenceFitConfig{
			Epochs:    2,
			BatchSize: 1,
			Optimizer: clipping,
			Loss:      loss.MeanSquaredError{},
		})
		if !errors.Is(err, updateErr) {
			t.Fatalf("FitWithLengths error = %v, want %v", err, updateErr)
		}
		if !strings.Contains(
			err.Error(),
			"model: sequence epoch 1 batch 1 training failed: model: length-aware optimizer update failed",
		) {
			t.Fatalf("FitWithLengths error = %q, want epoch optimizer context", err)
		}
		requireEpochCount(t, history, 0)
		if network.Training() || mode.modes[len(mode.modes)-1] {
			t.Fatal("FitWithLengths did not restore training mode")
		}
		observation, available = clipping.Observation()
		if !available || observation.BaseUpdateCompleted {
			t.Fatalf("base failure observation = %+v/%t, want failed base", observation, available)
		}
	})
}

func Test_Sequential_GradientClippingStateIsNotSerialized(t *testing.T) {
	var (
		network   *model.Sequential
		restored  *model.Sequential
		base      *optimizer.Adam
		clipping  *optimizer.GradientClipping
		before    bytes.Buffer
		after     bytes.Buffer
		reencoded bytes.Buffer
		err       error
	)

	network, _ = mustFixedGradientClippingRNN(t, 2)
	if err = network.Save(&before); err != nil {
		t.Fatalf("Save before optimizer construction returned error: %v", err)
	}
	if base, err = optimizer.NewAdam(0.01); err != nil {
		t.Fatalf("NewAdam returned error: %v", err)
	}
	if clipping, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{MaxValue: 1, MaxNorm: 5},
	); err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}
	if clipping == nil {
		t.Fatal("NewGradientClipping returned nil optimizer")
	}
	if err = network.Save(&after); err != nil {
		t.Fatalf("Save after optimizer construction returned error: %v", err)
	}
	if !bytes.Equal(before.Bytes(), after.Bytes()) {
		t.Fatal("constructing clipping changed version 1 model bytes")
	}
	if strings.Contains(after.String(), "clipping") ||
		strings.Contains(after.String(), "optimizer") ||
		strings.Contains(after.String(), "gradient") {
		t.Fatalf("serialized model contains optimizer state: %s", after.String())
	}

	if restored, err = model.LoadSequential(bytes.NewReader(after.Bytes())); err != nil {
		t.Fatalf("LoadSequential returned error: %v", err)
	}
	if err = restored.Save(&reencoded); err != nil {
		t.Fatalf("restored Save returned error: %v", err)
	}
	if !bytes.Equal(after.Bytes(), reencoded.Bytes()) {
		t.Fatal("loaded model did not preserve version 1 bytes")
	}
}

func mustFixedGradientClippingRNN(
	tb testing.TB,
	steps int,
) (network *model.Sequential, recurrent *layer.SimpleRNN) {
	var (
		inputShape layer.SequenceShape
		config     layer.SimpleRNNConfig
		lastStep   *layer.LastStep
		err        error
	)

	tb.Helper()
	if inputShape, err = layer.NewSequenceShape(steps, 1); err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if config, err = layer.NewSimpleRNNConfig(inputShape, 1); err != nil {
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
	if network, err = model.NewSequential(recurrent, lastStep); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	return network, recurrent
}

func mustLengthAwareGradientClippingRNN(
	tb testing.TB,
	steps int,
) (network *model.Sequential, recurrent *layer.SimpleRNN) {
	var (
		inputShape layer.SequenceShape
		config     layer.SimpleRNNConfig
		gather     *layer.GatherLastValid
		err        error
	)

	tb.Helper()
	if inputShape, err = layer.NewSequenceShape(steps, 1); err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if config, err = layer.NewSimpleRNNConfig(inputShape, 1); err != nil {
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
	if network, err = model.NewSequential(recurrent, gather); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	return network, recurrent
}

func mustAccumulateGradient(
	tb testing.TB,
	parameter *optimizer.Parameter,
	values []float32,
) {
	var gradient *matrix.Matrix
	var err error

	tb.Helper()
	gradient = mustMatrix(tb, parameter.Gradient().Rows(), parameter.Gradient().Cols(), values)
	if err = parameter.AccumulateGradient(gradient); err != nil {
		tb.Fatalf("AccumulateGradient returned error: %v", err)
	}
}

func requireFloat64AlmostEqual(
	tb testing.TB,
	got,
	want,
	tolerance float64,
) {
	var difference float64

	tb.Helper()
	difference = math.Abs(got - want)
	if math.IsNaN(got) || math.IsInf(got, 0) ||
		difference > tolerance+tolerance*math.Abs(want) {
		tb.Fatalf("value = %.17g, want %.17g, difference %.17g", got, want, difference)
	}
}
