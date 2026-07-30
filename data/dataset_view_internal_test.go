package data

import (
	"reflect"
	"testing"
	"unsafe"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_DatasetViewSharesStorageOnlyOnExplicitViewPath(t *testing.T) {
	var (
		sourceInputs   *matrix.Matrix
		sourceTargets  *matrix.Matrix
		dataset        *Dataset
		accessorInputs *matrix.Matrix
		batches        []*Batch
		inputView      *matrix.Matrix
		targetView     *matrix.Matrix
		err            error
	)

	sourceInputs = mustOwnedMatrix(t, 3, 2, []float32{1, 10, 2, 20, 3, 30})
	sourceTargets = mustOwnedMatrix(t, 3, 1, []float32{101, 102, 103})
	if dataset, err = NewDataset(sourceInputs, sourceTargets); err != nil {
		t.Fatalf("NewDataset returned error: %v", err)
	}
	if matrixStoragePointer(sourceInputs) == matrixStoragePointer(dataset.inputs) {
		t.Fatal("NewDataset input storage aliases constructor input")
	}
	if matrixStoragePointer(sourceTargets) == matrixStoragePointer(dataset.targets) {
		t.Fatal("NewDataset target storage aliases constructor target")
	}
	if accessorInputs, err = dataset.Inputs(); err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}
	if matrixStoragePointer(accessorInputs) == matrixStoragePointer(dataset.inputs) {
		t.Fatal("Inputs accessor storage aliases dataset")
	}

	err = dataset.WithRowView(1, 3, func(view *DatasetView) (viewErr error) {
		if inputView, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if targetView, viewErr = view.Targets(); viewErr != nil {
			return viewErr
		}
		if matrixStoragePointer(inputView) != matrixStoragePointer(dataset.inputs)+2*unsafe.Sizeof(float32(0)) {
			t.Fatal("dataset input view does not alias selected owner row")
		}
		if matrixStoragePointer(targetView) != matrixStoragePointer(dataset.targets)+unsafe.Sizeof(float32(0)) {
			t.Fatal("dataset target view does not alias selected owner row")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("WithRowView returned error: %v", err)
	}

	if batches, err = dataset.Batches(2, nil); err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}
	if matrixStoragePointer(batches[0].inputs) == matrixStoragePointer(dataset.inputs) {
		t.Fatal("safe Batch input storage aliases dataset")
	}
	err = batches[0].WithView(func(view *DatasetView) (viewErr error) {
		if inputView, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if matrixStoragePointer(inputView) != matrixStoragePointer(batches[0].inputs) {
			t.Fatal("batch input view does not alias batch owner")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("batch WithView returned error: %v", err)
	}
}

func Test_DatasetViewRejectsInvalidPrivateAssociationBeforeCallback(t *testing.T) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		dataset Dataset
		called  bool
		err     error
	)

	inputs = mustOwnedMatrix(t, 2, 1, []float32{1, 2})
	targets = mustOwnedMatrix(t, 1, 1, []float32{10})
	dataset.inputs = inputs
	dataset.targets = targets
	err = dataset.WithView(func(*DatasetView) (viewErr error) {
		called = true
		return nil
	})
	if err == nil {
		t.Fatal("WithView error = nil, want invalid association error")
	}
	if called {
		t.Fatal("WithView invoked callback for invalid association")
	}
}

func matrixStoragePointer(value *matrix.Matrix) (pointer uintptr) {
	var storage reflect.Value

	storage = reflect.ValueOf(value).Elem().FieldByName("data")
	pointer = storage.Pointer()
	return pointer
}
