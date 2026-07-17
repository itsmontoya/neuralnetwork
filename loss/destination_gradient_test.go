package loss_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_DestinationGradient_Interface(t *testing.T) {
	var _ loss.DestinationGradient = loss.MeanSquaredError{}
	var _ loss.DestinationGradient = loss.BinaryCrossEntropy{}
	var _ loss.DestinationGradient = loss.CategoricalCrossEntropy{}
}

func Test_DestinationGradient_EquivalentToAllocatingGradient(t *testing.T) {
	var tests []struct {
		name        string
		lossFunc    loss.Loss
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	}

	tests = []struct {
		name        string
		lossFunc    loss.Loss
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	}{
		{
			name:        "MeanSquaredError",
			lossFunc:    loss.MeanSquaredError{},
			predictions: mustMatrix(t, 2, 2, []float32{1, 2, 3, 4}),
			targets:     mustMatrix(t, 2, 2, []float32{1.5, 1, 2, 5}),
		},
		{
			name:        "BinaryCrossEntropy",
			lossFunc:    loss.BinaryCrossEntropy{},
			predictions: mustMatrix(t, 2, 1, []float32{0.8, 0.25}),
			targets:     mustMatrix(t, 2, 1, []float32{1, 0}),
		},
		{
			name:        "BinaryCrossEntropyBoundaries",
			lossFunc:    loss.BinaryCrossEntropy{},
			predictions: mustMatrix(t, 2, 1, []float32{0, 1}),
			targets:     mustMatrix(t, 2, 1, []float32{1, 0}),
		},
		{
			name:        "CategoricalCrossEntropy",
			lossFunc:    loss.CategoricalCrossEntropy{},
			predictions: mustMatrix(t, 2, 3, []float32{0.7, 0.2, 0.1, 0.1, 0.6, 0.3}),
			targets:     mustMatrix(t, 2, 3, []float32{1, 0, 0, 0, 1, 0}),
		},
		{
			name:        "CategoricalCrossEntropyBoundaries",
			lossFunc:    loss.CategoricalCrossEntropy{},
			predictions: mustMatrix(t, 2, 3, []float32{1, 0, 0, 0, 1, 0}),
			targets:     mustMatrix(t, 2, 3, []float32{0, 1, 0, 0, 0, 1}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				destinationLoss loss.DestinationGradient
				want            *matrix.Matrix
				second          *matrix.Matrix
				destination     *matrix.Matrix
				wantValues      []float32
				err             error
			)

			destinationLoss = tt.lossFunc.(loss.DestinationGradient)
			if want, err = tt.lossFunc.Gradient(tt.predictions, tt.targets); err != nil {
				t.Fatalf("Gradient returned error: %v", err)
			}

			if destination, err = matrix.New(tt.predictions.Rows(), tt.predictions.Cols()); err != nil {
				t.Fatalf("New destination returned error: %v", err)
			}

			if err = destination.Fill(42); err != nil {
				t.Fatalf("destination Fill returned error: %v", err)
			}

			if err = destinationLoss.GradientInto(tt.predictions, tt.targets, destination); err != nil {
				t.Fatalf("GradientInto returned error: %v", err)
			}

			if wantValues, err = want.Values(); err != nil {
				t.Fatalf("gradient Values returned error: %v", err)
			}
			requireMatrixValues(t, destination, wantValues)

			if second, err = tt.lossFunc.Gradient(tt.predictions, tt.targets); err != nil {
				t.Fatalf("second Gradient returned error: %v", err)
			}
			if second == want {
				t.Fatal("separate Gradient calls returned the same matrix")
			}
			if err = want.Fill(42); err != nil {
				t.Fatalf("first gradient Fill returned error: %v", err)
			}
			requireMatrixValues(t, second, wantValues)
		})
	}
}

func Test_DestinationGradient_PermitsInputAliasing(t *testing.T) {
	var tests []struct {
		name        string
		lossFunc    loss.Loss
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	}

	tests = []struct {
		name        string
		lossFunc    loss.Loss
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	}{
		{
			name:        "MeanSquaredError",
			lossFunc:    loss.MeanSquaredError{},
			predictions: mustMatrix(t, 2, 2, []float32{1, 2, 3, 4}),
			targets:     mustMatrix(t, 2, 2, []float32{1.5, 1, 2, 5}),
		},
		{
			name:        "BinaryCrossEntropy",
			lossFunc:    loss.BinaryCrossEntropy{},
			predictions: mustMatrix(t, 2, 1, []float32{0.8, 0.25}),
			targets:     mustMatrix(t, 2, 1, []float32{1, 0}),
		},
		{
			name:        "CategoricalCrossEntropy",
			lossFunc:    loss.CategoricalCrossEntropy{},
			predictions: mustMatrix(t, 2, 3, []float32{0.7, 0.2, 0.1, 0.1, 0.6, 0.3}),
			targets:     mustMatrix(t, 2, 3, []float32{1, 0, 0, 0, 1, 0}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				destinationLoss  loss.DestinationGradient
				want             *matrix.Matrix
				predictionAlias  *matrix.Matrix
				predictionTarget *matrix.Matrix
				targetAlias      *matrix.Matrix
				targetPrediction *matrix.Matrix
				wantValues       []float32
				err              error
			)

			destinationLoss = tt.lossFunc.(loss.DestinationGradient)
			if want, err = tt.lossFunc.Gradient(tt.predictions, tt.targets); err != nil {
				t.Fatalf("Gradient returned error: %v", err)
			}

			if wantValues, err = want.Values(); err != nil {
				t.Fatalf("gradient Values returned error: %v", err)
			}

			if predictionAlias, err = tt.predictions.Clone(); err != nil {
				t.Fatalf("predictions Clone returned error: %v", err)
			}
			if predictionTarget, err = tt.targets.Clone(); err != nil {
				t.Fatalf("targets Clone returned error: %v", err)
			}
			if err = destinationLoss.GradientInto(predictionAlias, predictionTarget, predictionAlias); err != nil {
				t.Fatalf("prediction-aliased GradientInto returned error: %v", err)
			}
			requireMatrixValues(t, predictionAlias, wantValues)

			if targetPrediction, err = tt.predictions.Clone(); err != nil {
				t.Fatalf("predictions Clone returned error: %v", err)
			}
			if targetAlias, err = tt.targets.Clone(); err != nil {
				t.Fatalf("targets Clone returned error: %v", err)
			}
			if err = destinationLoss.GradientInto(targetPrediction, targetAlias, targetAlias); err != nil {
				t.Fatalf("target-aliased GradientInto returned error: %v", err)
			}
			requireMatrixValues(t, targetAlias, wantValues)
		})
	}
}

func Test_DestinationGradient_PreservesInputErrors(t *testing.T) {
	var (
		oneByOne *matrix.Matrix
		oneByTwo *matrix.Matrix
		twoByOne *matrix.Matrix
		twoByTwo *matrix.Matrix
	)

	oneByOne = mustMatrix(t, 1, 1, []float32{0.5})
	oneByTwo = mustMatrix(t, 1, 2, []float32{0.5, 0.5})
	twoByOne = mustMatrix(t, 2, 1, []float32{1, 0})
	twoByTwo = mustMatrix(t, 2, 2, []float32{0.5, 0.5, 0.5, 0.5})

	var tests []struct {
		name        string
		lossFunc    loss.Loss
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		destination *matrix.Matrix
	}

	tests = []struct {
		name        string
		lossFunc    loss.Loss
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		destination *matrix.Matrix
	}{
		{
			name:        "nil predictions",
			lossFunc:    loss.MeanSquaredError{},
			targets:     oneByOne,
			destination: twoByTwo,
		},
		{
			name:        "nil targets",
			lossFunc:    loss.MeanSquaredError{},
			predictions: oneByOne,
			destination: twoByTwo,
		},
		{
			name:        "shape mismatch",
			lossFunc:    loss.MeanSquaredError{},
			predictions: oneByTwo,
			targets:     twoByOne,
			destination: twoByTwo,
		},
		{
			name:        "binary multiple outputs",
			lossFunc:    loss.BinaryCrossEntropy{},
			predictions: oneByTwo,
			targets:     oneByTwo,
			destination: oneByOne,
		},
		{
			name:        "binary invalid target",
			lossFunc:    loss.BinaryCrossEntropy{},
			predictions: oneByOne,
			targets:     oneByOne,
			destination: twoByOne,
		},
		{
			name:        "categorical no class",
			lossFunc:    loss.CategoricalCrossEntropy{},
			predictions: oneByTwo,
			targets:     mustMatrix(t, 1, 2, []float32{0, 0}),
			destination: oneByOne,
		},
		{
			name:        "categorical multiple classes",
			lossFunc:    loss.CategoricalCrossEntropy{},
			predictions: oneByTwo,
			targets:     mustMatrix(t, 1, 2, []float32{1, 1}),
			destination: oneByOne,
		},
		{
			name:        "categorical fractional class",
			lossFunc:    loss.CategoricalCrossEntropy{},
			predictions: oneByTwo,
			targets:     oneByTwo,
			destination: oneByOne,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				destinationLoss loss.DestinationGradient
				allocatingErr   error
				destinationErr  error
			)

			destinationLoss = tt.lossFunc.(loss.DestinationGradient)
			_, allocatingErr = tt.lossFunc.Gradient(tt.predictions, tt.targets)
			if allocatingErr == nil {
				t.Fatal("Gradient error = nil, want error")
			}

			destinationErr = destinationLoss.GradientInto(tt.predictions, tt.targets, tt.destination)
			if destinationErr == nil {
				t.Fatal("GradientInto error = nil, want error")
			}

			if destinationErr.Error() != allocatingErr.Error() {
				t.Fatalf("GradientInto error = %q, want %q", destinationErr, allocatingErr)
			}
		})
	}
}

func Test_DestinationGradient_ValidatesDestination(t *testing.T) {
	var tests []struct {
		name        string
		lossFunc    loss.DestinationGradient
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	}

	tests = []struct {
		name        string
		lossFunc    loss.DestinationGradient
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	}{
		{
			name:        "MeanSquaredError",
			lossFunc:    loss.MeanSquaredError{},
			predictions: mustMatrix(t, 1, 2, []float32{1, 2}),
			targets:     mustMatrix(t, 1, 2, []float32{0, 1}),
		},
		{
			name:        "BinaryCrossEntropy",
			lossFunc:    loss.BinaryCrossEntropy{},
			predictions: mustMatrix(t, 2, 1, []float32{0.25, 0.75}),
			targets:     mustMatrix(t, 2, 1, []float32{0, 1}),
		},
		{
			name:        "CategoricalCrossEntropy",
			lossFunc:    loss.CategoricalCrossEntropy{},
			predictions: mustMatrix(t, 1, 2, []float32{0.25, 0.75}),
			targets:     mustMatrix(t, 1, 2, []float32{0, 1}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				wrongShape *matrix.Matrix
				err        error
			)

			wrongShape = mustMatrix(t, 1, 1, []float32{0})
			if err = tt.lossFunc.GradientInto(tt.predictions, tt.targets, nil); err == nil {
				t.Fatal("GradientInto error = nil for nil destination")
			}

			if err = tt.lossFunc.GradientInto(tt.predictions, tt.targets, wrongShape); err == nil {
				t.Fatal("GradientInto error = nil for wrong destination shape")
			}
		})
	}
}
