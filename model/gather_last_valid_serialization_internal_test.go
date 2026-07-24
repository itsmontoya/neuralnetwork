package model

import (
	"bytes"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_LoadSequential_GatherLastValidRuntimeStateStartsFresh(t *testing.T) {
	var (
		inputShape     layer.SequenceShape
		gather         *layer.GatherLastValid
		network        *Sequential
		loaded         *Sequential
		loadedGather   *layer.GatherLastValid
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		document       bytes.Buffer
		ok             bool
		err            error
	)

	inputShape, err = layer.NewSequenceShape(2, 2)
	if err != nil {
		t.Fatalf("NewSequenceShape returned error: %v", err)
	}

	gather, err = layer.NewGatherLastValid(inputShape)
	if err != nil {
		t.Fatalf("NewGatherLastValid returned error: %v", err)
	}

	input, err = matrix.FromSlice(1, 4, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("FromSlice input returned error: %v", err)
	}

	if _, err = gather.ForwardWithLengths(input, []int{1}); err != nil {
		t.Fatalf("ForwardWithLengths returned error: %v", err)
	}

	network, err = NewSequential(gather)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	if err = network.Save(&document); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	loaded, err = LoadSequential(&document)
	if err != nil {
		t.Fatalf("LoadSequential returned error: %v", err)
	}

	if len(loaded.layers) != 1 {
		t.Fatalf("loaded layer count = %d, want 1", len(loaded.layers))
	}

	loadedGather, ok = loaded.layers[0].(*layer.GatherLastValid)
	if !ok {
		t.Fatalf("loaded layer type = %T, want *layer.GatherLastValid", loaded.layers[0])
	}

	outputGradient, err = matrix.FromSlice(1, 2, []float32{1, -1})
	if err != nil {
		t.Fatalf("FromSlice output gradient returned error: %v", err)
	}

	if _, err = loadedGather.BackwardWithLengths(outputGradient); err == nil {
		t.Fatal("loaded BackwardWithLengths error = nil, want fresh forward-state error")
	} else if !strings.Contains(err.Error(), "backward called before length-aware forward") {
		t.Fatalf("loaded BackwardWithLengths error = %q, want fresh forward-state context", err)
	}

	if _, err = loadedGather.ForwardWithLengths(input, []int{2}); err != nil {
		t.Fatalf("loaded ForwardWithLengths returned error: %v", err)
	}

	if _, err = loadedGather.BackwardWithLengths(outputGradient); err != nil {
		t.Fatalf("loaded BackwardWithLengths returned error after forward: %v", err)
	}
}
