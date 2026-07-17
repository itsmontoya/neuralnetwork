package data

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_newDataset_StoresOwnedMatrices(t *testing.T) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		dataset *Dataset
		err     error
	)

	inputs = mustOwnedMatrix(t, 2, 1, []float32{1, 2})
	targets = mustOwnedMatrix(t, 2, 1, []float32{10, 20})
	if dataset, err = newDataset(inputs, targets); err != nil {
		t.Fatalf("newDataset returned error: %v", err)
	}

	if dataset.inputs != inputs {
		t.Fatal("newDataset copied inputs, want owned matrix")
	}

	if dataset.targets != targets {
		t.Fatal("newDataset copied targets, want owned matrix")
	}
}

func Test_DatasetSplit_IsolatesOwnership(t *testing.T) {
	var (
		sourceInputs    *matrix.Matrix
		sourceTargets   *matrix.Matrix
		source          *Dataset
		train           *Dataset
		test            *Dataset
		accessorInputs  *matrix.Matrix
		accessorTargets *matrix.Matrix
		err             error
		value           float32
	)

	sourceInputs = mustOwnedMatrix(t, 4, 1, []float32{1, 2, 3, 4})
	sourceTargets = mustOwnedMatrix(t, 4, 1, []float32{10, 20, 30, 40})
	if source, err = NewDataset(sourceInputs, sourceTargets); err != nil {
		t.Fatalf("NewDataset returned error: %v", err)
	}

	if train, test, err = source.Split(0.25, nil); err != nil {
		t.Fatalf("Split returned error: %v", err)
	}

	if train.inputs == source.inputs || test.inputs == source.inputs || train.inputs == test.inputs {
		t.Fatal("Split input matrices share ownership")
	}

	if train.targets == source.targets || test.targets == source.targets || train.targets == test.targets {
		t.Fatal("Split target matrices share ownership")
	}

	if err = sourceInputs.Set(0, 0, 91); err != nil {
		t.Fatalf("source inputs Set returned error: %v", err)
	}

	if err = sourceTargets.Set(0, 0, 92); err != nil {
		t.Fatalf("source targets Set returned error: %v", err)
	}

	if err = source.inputs.Set(0, 0, 93); err != nil {
		t.Fatalf("dataset inputs Set returned error: %v", err)
	}

	if err = source.targets.Set(0, 0, 94); err != nil {
		t.Fatalf("dataset targets Set returned error: %v", err)
	}

	if value, err = train.inputs.At(0, 0); err != nil {
		t.Fatalf("train inputs At returned error: %v", err)
	}

	if value != 1 {
		t.Fatalf("train input = %g, want 1", value)
	}

	if accessorInputs, err = train.Inputs(); err != nil {
		t.Fatalf("train Inputs returned error: %v", err)
	}

	if err = accessorInputs.Set(0, 0, 95); err != nil {
		t.Fatalf("accessor inputs Set returned error: %v", err)
	}

	if accessorTargets, err = train.Targets(); err != nil {
		t.Fatalf("train Targets returned error: %v", err)
	}

	if err = accessorTargets.Set(0, 0, 97); err != nil {
		t.Fatalf("accessor targets Set returned error: %v", err)
	}

	if value, err = train.inputs.At(0, 0); err != nil {
		t.Fatalf("train inputs At after accessor mutation returned error: %v", err)
	}

	if value != 1 {
		t.Fatalf("train input after accessor mutation = %g, want 1", value)
	}

	if value, err = train.targets.At(0, 0); err != nil {
		t.Fatalf("train targets At after accessor mutation returned error: %v", err)
	}

	if value != 10 {
		t.Fatalf("train target after accessor mutation = %g, want 10", value)
	}

	if err = train.inputs.Set(0, 0, 96); err != nil {
		t.Fatalf("train inputs Set returned error: %v", err)
	}

	if err = train.targets.Set(0, 0, 98); err != nil {
		t.Fatalf("train targets Set returned error: %v", err)
	}

	if value, err = source.inputs.At(0, 0); err != nil {
		t.Fatalf("source inputs At returned error: %v", err)
	}

	if value != 93 {
		t.Fatalf("source input = %g, want 93", value)
	}

	if value, err = test.inputs.At(0, 0); err != nil {
		t.Fatalf("test inputs At returned error: %v", err)
	}

	if value != 4 {
		t.Fatalf("test input = %g, want 4", value)
	}

	if value, err = source.targets.At(0, 0); err != nil {
		t.Fatalf("source targets At returned error: %v", err)
	}

	if value != 94 {
		t.Fatalf("source target = %g, want 94", value)
	}

	if value, err = test.targets.At(0, 0); err != nil {
		t.Fatalf("test targets At returned error: %v", err)
	}

	if value != 40 {
		t.Fatalf("test target = %g, want 40", value)
	}
}

func mustOwnedMatrix(tb testing.TB, rows, cols int, values []float32) (m *matrix.Matrix) {
	var err error

	tb.Helper()

	if m, err = matrix.FromSlice(rows, cols, values); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}
