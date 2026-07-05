package main

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_NewXORDataset(t *testing.T) {
	var (
		dataset      *data.Dataset
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		inputValues  []float64
		targetValues []float64
		wantInputs   []float64
		wantTargets  []float64
		index        int
		err          error
	)

	dataset, err = newXORDataset()
	if err != nil {
		t.Fatalf("newXORDataset returned error: %v", err)
	}

	if dataset.SampleCount() != 4 {
		t.Fatalf("SampleCount = %d, want 4", dataset.SampleCount())
	}

	if dataset.InputSize() != 2 {
		t.Fatalf("InputSize = %d, want 2", dataset.InputSize())
	}

	if dataset.TargetSize() != 1 {
		t.Fatalf("TargetSize = %d, want 1", dataset.TargetSize())
	}

	inputs, err = dataset.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	inputValues, err = inputs.Values()
	if err != nil {
		t.Fatalf("input Values returned error: %v", err)
	}

	wantInputs = []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	}
	for index = range wantInputs {
		if inputValues[index] == wantInputs[index] {
			continue
		}

		t.Fatalf("input value %d = %g, want %g", index, inputValues[index], wantInputs[index])
	}

	targets, err = dataset.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	targetValues, err = targets.Values()
	if err != nil {
		t.Fatalf("target Values returned error: %v", err)
	}

	wantTargets = []float64{0, 1, 1, 0}
	for index = range wantTargets {
		if targetValues[index] == wantTargets[index] {
			continue
		}

		t.Fatalf("target value %d = %g, want %g", index, targetValues[index], wantTargets[index])
	}
}

func Test_NewXORModelPredictsDatasetShape(t *testing.T) {
	var (
		random      *rand.Rand
		dataset     *data.Dataset
		network     *model.Sequential
		inputs      *matrix.Matrix
		predictions *matrix.Matrix
		err         error
	)

	random = rand.New(rand.NewSource(1))
	dataset, err = newXORDataset()
	if err != nil {
		t.Fatalf("newXORDataset returned error: %v", err)
	}

	network, err = newXORModel(random)
	if err != nil {
		t.Fatalf("newXORModel returned error: %v", err)
	}

	inputs, err = dataset.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	predictions, err = network.Predict(inputs)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	if predictions.Rows() != dataset.SampleCount() {
		t.Fatalf("prediction Rows = %d, want %d", predictions.Rows(), dataset.SampleCount())
	}

	if predictions.Cols() != dataset.TargetSize() {
		t.Fatalf("prediction Cols = %d, want %d", predictions.Cols(), dataset.TargetSize())
	}
}
