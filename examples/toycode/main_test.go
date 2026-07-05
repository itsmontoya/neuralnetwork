package main

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_NewNextTokenDataset(t *testing.T) {
	var (
		tokens       []string
		tokenIndexes map[string]int
		sequences    [][]int
		dataset      *data.Dataset
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		inputValues  []float64
		targetValues []float64
		sequence     []int
		expectedRows int
		index        int
		offset       int
		err          error
	)

	tokens = vocabulary()
	tokenIndexes = indexTokens(tokens)
	sequences, err = encodedPrograms(tokenIndexes)
	if err != nil {
		t.Fatalf("encodedPrograms returned error: %v", err)
	}

	for _, sequence = range sequences {
		expectedRows += len(sequence) + 1
	}

	dataset, err = newNextTokenDataset(sequences, len(tokens), tokenIndexes[padToken], tokenIndexes[endToken])
	if err != nil {
		t.Fatalf("newNextTokenDataset returned error: %v", err)
	}

	if dataset.SampleCount() != expectedRows {
		t.Fatalf("SampleCount = %d, want %d", dataset.SampleCount(), expectedRows)
	}

	if dataset.InputSize() != contextSize*len(tokens) {
		t.Fatalf("InputSize = %d, want %d", dataset.InputSize(), contextSize*len(tokens))
	}

	if dataset.TargetSize() != len(tokens) {
		t.Fatalf("TargetSize = %d, want %d", dataset.TargetSize(), len(tokens))
	}

	inputs, err = dataset.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	inputValues, err = inputs.Values()
	if err != nil {
		t.Fatalf("input Values returned error: %v", err)
	}

	for index = 0; index < contextSize; index++ {
		offset = index*len(tokens) + tokenIndexes[padToken]
		if inputValues[offset] == 1 {
			continue
		}

		t.Fatalf("initial context slot %d pad value = %g, want 1", index, inputValues[offset])
	}

	targets, err = dataset.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	targetValues, err = targets.Values()
	if err != nil {
		t.Fatalf("target Values returned error: %v", err)
	}

	if targetValues[tokenIndexes["fn"]] != 1 {
		t.Fatalf("first target fn value = %g, want 1", targetValues[tokenIndexes["fn"]])
	}
}

func Test_NewToyCodeModelPredictsVocabularyShape(t *testing.T) {
	var (
		tokens       []string
		tokenIndexes map[string]int
		random       *rand.Rand
		network      *model.Sequential
		context      []int
		inputValues  []float64
		input        *matrix.Matrix
		predictions  *matrix.Matrix
		err          error
	)

	tokens = vocabulary()
	tokenIndexes = indexTokens(tokens)
	random = rand.New(rand.NewSource(101))
	if network, err = newToyCodeModel(len(tokens), random); err != nil {
		t.Fatalf("newToyCodeModel returned error: %v", err)
	}

	context = paddedContext(tokenIndexes[padToken])
	inputValues = appendContext(nil, context, len(tokens))
	input, err = matrix.FromSlice(1, contextSize*len(tokens), inputValues)
	if err != nil {
		t.Fatalf("FromSlice returned error: %v", err)
	}

	predictions, err = network.Predict(input)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	if predictions.Rows() != 1 {
		t.Fatalf("prediction Rows = %d, want 1", predictions.Rows())
	}

	if predictions.Cols() != len(tokens) {
		t.Fatalf("prediction Cols = %d, want %d", predictions.Cols(), len(tokens))
	}
}

func Test_ToyCodeHelpers(t *testing.T) {
	var (
		tokens       []string
		tokenIndexes map[string]int
		sequence     []int
		generated    string
		token        int
		err          error
	)

	tokens = vocabulary()
	tokenIndexes = indexTokens(tokens)
	sequence, err = encodeProgram([]string{"fn", "add", "(", "a", ")"}, tokenIndexes)
	if err != nil {
		t.Fatalf("encodeProgram returned error: %v", err)
	}

	if len(sequence) != 5 {
		t.Fatalf("sequence length = %d, want 5", len(sequence))
	}

	if sequence[0] != tokenIndexes["fn"] {
		t.Fatalf("first sequence token = %d, want %d", sequence[0], tokenIndexes["fn"])
	}

	_, err = encodeProgram([]string{"fn", "missing"}, tokenIndexes)
	if err == nil {
		t.Fatal("encodeProgram missing token error = nil, want error")
	}

	generated = formatToyCode([]string{"fn", "add", "(", "a", ",", "b", ")", "{", "ret", "a", ";", "}"})
	if generated != "fn add (a, b) {\n  ret a;\n}\n" {
		t.Fatalf("formatToyCode output = %q", generated)
	}

	token = sampleToken([]float64{0.9, 0.1}, 0, rand.New(rand.NewSource(107)))
	if token != 1 {
		t.Fatalf("sampleToken blocked token = %d, want 1", token)
	}

	token = argmax([]float64{0.1, 0.7, 0.7})
	if token != 1 {
		t.Fatalf("argmax tie token = %d, want 1", token)
	}
}
