package main

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_NewRegressionDataset(t *testing.T) {
	var (
		random  *rand.Rand
		dataset *data.Dataset
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		first   float64
		last    float64
		err     error
	)

	random = rand.New(rand.NewSource(11))
	dataset, err = newRegressionDataset(random)
	if err != nil {
		t.Fatalf("newRegressionDataset returned error: %v", err)
	}

	if dataset.SampleCount() != sampleCount {
		t.Fatalf("SampleCount = %d, want %d", dataset.SampleCount(), sampleCount)
	}

	if dataset.InputSize() != 1 {
		t.Fatalf("InputSize = %d, want 1", dataset.InputSize())
	}

	if dataset.TargetSize() != 1 {
		t.Fatalf("TargetSize = %d, want 1", dataset.TargetSize())
	}

	inputs, err = dataset.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	first, err = inputs.At(0, 0)
	if err != nil {
		t.Fatalf("At first input returned error: %v", err)
	}

	if first != -1 {
		t.Fatalf("first input = %g, want -1", first)
	}

	last, err = inputs.At(sampleCount-1, 0)
	if err != nil {
		t.Fatalf("At last input returned error: %v", err)
	}

	if last != 1 {
		t.Fatalf("last input = %g, want 1", last)
	}

	targets, err = dataset.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	if targets.Rows() != sampleCount {
		t.Fatalf("target Rows = %d, want %d", targets.Rows(), sampleCount)
	}

	if targets.Cols() != 1 {
		t.Fatalf("target Cols = %d, want 1", targets.Cols())
	}
}

func Test_NewRegressionModelPredictsInputShape(t *testing.T) {
	var (
		random      *rand.Rand
		network     *model.Sequential
		inputs      *matrix.Matrix
		predictions *matrix.Matrix
		err         error
	)

	random = rand.New(rand.NewSource(13))
	if network, err = newRegressionModel(random); err != nil {
		t.Fatalf("newRegressionModel returned error: %v", err)
	}

	inputs, err = matrix.FromSlice(3, 1, []float64{-1, 0, 1})
	if err != nil {
		t.Fatalf("FromSlice returned error: %v", err)
	}

	predictions, err = network.Predict(inputs)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	if predictions.Rows() != 3 {
		t.Fatalf("prediction Rows = %d, want 3", predictions.Rows())
	}

	if predictions.Cols() != 1 {
		t.Fatalf("prediction Cols = %d, want 1", predictions.Cols())
	}
}
