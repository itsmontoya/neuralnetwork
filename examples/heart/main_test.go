package main

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_NewHeartDataset(t *testing.T) {
	var (
		random       *rand.Rand
		dataset      *data.Dataset
		targets      *matrix.Matrix
		targetValues []float32
		zeros        int
		ones         int
		value        float32
		err          error
	)

	random = rand.New(rand.NewSource(31))
	dataset, err = newHeartDataset(random)
	if err != nil {
		t.Fatalf("newHeartDataset returned error: %v", err)
	}

	if dataset.SampleCount() != sampleCount {
		t.Fatalf("SampleCount = %d, want %d", dataset.SampleCount(), sampleCount)
	}

	if dataset.InputSize() != 2 {
		t.Fatalf("InputSize = %d, want 2", dataset.InputSize())
	}

	if dataset.TargetSize() != 1 {
		t.Fatalf("TargetSize = %d, want 1", dataset.TargetSize())
	}

	targets, err = dataset.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	targetValues, err = targets.Values()
	if err != nil {
		t.Fatalf("Values returned error: %v", err)
	}

	for _, value = range targetValues {
		switch value {
		case 0:
			zeros++
		case 1:
			ones++
		default:
			t.Fatalf("target value = %g, want binary value", value)
		}
	}

	if zeros == 0 {
		t.Fatal("target values contained no zero class samples")
	}

	if ones == 0 {
		t.Fatal("target values contained no one class samples")
	}
}

func Test_NewHeartModelPredictsInputShape(t *testing.T) {
	var (
		random      *rand.Rand
		network     *model.Sequential
		inputs      *matrix.Matrix
		predictions *matrix.Matrix
		err         error
	)

	random = rand.New(rand.NewSource(37))
	if network, err = newHeartModel(random); err != nil {
		t.Fatalf("newHeartModel returned error: %v", err)
	}

	inputs, err = matrix.FromSlice(3, 2, []float32{
		0, 0,
		minX, minY,
		maxX, maxY,
	})
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

func Test_HeartHelpers(t *testing.T) {
	type heartCase struct {
		name string
		x    float32
		y    float32
		want bool
	}

	type shadeCase struct {
		value float32
		char  string
	}

	var (
		heartTests []heartCase
		shadeTests []shadeCase
		heartTest  heartCase
		shadeTest  shadeCase
	)

	heartTests = []heartCase{
		{name: "center", x: 0, y: 0, want: true},
		{name: "corner", x: maxX, y: maxY, want: false},
	}

	for _, heartTest = range heartTests {
		if insideHeart(heartTest.x, heartTest.y) == heartTest.want {
			continue
		}

		t.Fatalf(
			"%s: insideHeart(%g, %g) = %t, want %t",
			heartTest.name,
			heartTest.x,
			heartTest.y,
			insideHeart(heartTest.x, heartTest.y),
			heartTest.want,
		)
	}

	shadeTests = []shadeCase{
		{value: 0.85, char: "@"},
		{value: 0.65, char: "#"},
		{value: 0.45, char: "+"},
		{value: 0.25, char: "."},
		{value: 0.10, char: " "},
	}

	for _, shadeTest = range shadeTests {
		if shade(shadeTest.value) == shadeTest.char {
			continue
		}

		t.Fatalf("shade(%g) = %q, want %q", shadeTest.value, shade(shadeTest.value), shadeTest.char)
	}
}
