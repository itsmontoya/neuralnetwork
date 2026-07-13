package main

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_NewClusterDataset(t *testing.T) {
	var (
		random       *rand.Rand
		dataset      *data.Dataset
		targets      *matrix.Matrix
		targetValues []float32
		wantSamples  int
		classIndex   int
		col          int
		row          int
		want         float32
		got          float32
		err          error
	)

	random = rand.New(rand.NewSource(21))
	dataset, err = newClusterDataset(random)
	if err != nil {
		t.Fatalf("newClusterDataset returned error: %v", err)
	}

	wantSamples = len(clusterCenters()) * samplesPerClass
	if dataset.SampleCount() != wantSamples {
		t.Fatalf("SampleCount = %d, want %d", dataset.SampleCount(), wantSamples)
	}

	if dataset.InputSize() != 2 {
		t.Fatalf("InputSize = %d, want 2", dataset.InputSize())
	}

	if dataset.TargetSize() != classCount {
		t.Fatalf("TargetSize = %d, want %d", dataset.TargetSize(), classCount)
	}

	targets, err = dataset.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	targetValues, err = targets.Values()
	if err != nil {
		t.Fatalf("Values returned error: %v", err)
	}

	for classIndex = 0; classIndex < classCount; classIndex++ {
		row = classIndex * samplesPerClass
		for col = 0; col < classCount; col++ {
			want = 0
			if col == classIndex {
				want = 1
			}

			got = targetValues[row*classCount+col]
			if got == want {
				continue
			}

			t.Fatalf("target row %d col %d = %g, want %g", row, col, got, want)
		}
	}
}

func Test_NewClusterModelPredictsClassShape(t *testing.T) {
	var (
		random      *rand.Rand
		network     *model.Sequential
		centers     []point
		inputValues []float32
		inputs      *matrix.Matrix
		predictions *matrix.Matrix
		center      point
		err         error
	)

	random = rand.New(rand.NewSource(23))
	if network, err = newClusterModel(random); err != nil {
		t.Fatalf("newClusterModel returned error: %v", err)
	}

	centers = clusterCenters()
	inputValues = make([]float32, 0, len(centers)*2)
	for _, center = range centers {
		inputValues = append(inputValues, center.x, center.y)
	}

	inputs, err = matrix.FromSlice(len(centers), 2, inputValues)
	if err != nil {
		t.Fatalf("FromSlice returned error: %v", err)
	}

	predictions, err = network.Predict(inputs)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	if predictions.Rows() != len(centers) {
		t.Fatalf("prediction Rows = %d, want %d", predictions.Rows(), len(centers))
	}

	if predictions.Cols() != classCount {
		t.Fatalf("prediction Cols = %d, want %d", predictions.Cols(), classCount)
	}
}

func Test_ClassHelpers(t *testing.T) {
	type classCase struct {
		index int
		name  string
	}

	type argmaxCase struct {
		name   string
		values []float32
		want   int
	}

	var (
		classTests  []classCase
		argmaxTests []argmaxCase
		classTest   classCase
		argmaxTest  argmaxCase
		gotIndex    int
	)

	classTests = []classCase{
		{index: 0, name: "left"},
		{index: 1, name: "right"},
		{index: 2, name: "top"},
		{index: 3, name: "unknown"},
	}

	for _, classTest = range classTests {
		if className(classTest.index) == classTest.name {
			continue
		}

		t.Fatalf("className(%d) = %q, want %q", classTest.index, className(classTest.index), classTest.name)
	}

	argmaxTests = []argmaxCase{
		{name: "largest", values: []float32{0.1, 0.7, 0.2}, want: 1},
		{name: "tie keeps first maximum", values: []float32{0.1, 0.7, 0.7}, want: 1},
	}

	for _, argmaxTest = range argmaxTests {
		gotIndex = argmax(argmaxTest.values)
		if gotIndex == argmaxTest.want {
			continue
		}

		t.Fatalf("%s: argmax = %d, want %d", argmaxTest.name, gotIndex, argmaxTest.want)
	}
}
