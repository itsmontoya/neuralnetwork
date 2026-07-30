package data

import (
	"testing"
	"unsafe"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_SequenceDatasetViewSharesOnlyExplicitAlignedStorage(t *testing.T) {
	var (
		sourceInputs    *matrix.Matrix
		sourceTargets   *matrix.Matrix
		sourceValues    []int
		sourceLengths   *SequenceLengths
		dataset         *SequenceDataset
		accessorInputs  *matrix.Matrix
		accessorTargets *matrix.Matrix
		accessorLengths *SequenceLengths
		batches         []*SequenceBatch
		train           *SequenceDataset
		test            *SequenceDataset
		inputView       *matrix.Matrix
		targetView      *matrix.Matrix
		lengthView      []int
		err             error
	)

	sourceInputs = mustOwnedMatrix(t, 4, 4, []float32{
		1, 10, 100, 1000,
		2, 20, 200, 2000,
		3, 30, 300, 3000,
		4, 40, 400, 4000,
	})
	sourceTargets = mustOwnedMatrix(t, 4, 1, []float32{101, 102, 103, 104})
	sourceValues = []int{2, 1, 2, 1}
	if sourceLengths, err = NewSequenceLengths(2, sourceValues); err != nil {
		t.Fatalf("NewSequenceLengths returned error: %v", err)
	}
	if dataset, err = NewSequenceDataset(
		sourceInputs,
		sourceTargets,
		sourceLengths,
	); err != nil {
		t.Fatalf("NewSequenceDataset returned error: %v", err)
	}

	if matrixStoragePointer(sourceInputs) == matrixStoragePointer(dataset.inputs) {
		t.Fatal("NewSequenceDataset input storage aliases constructor input")
	}
	if matrixStoragePointer(sourceTargets) == matrixStoragePointer(dataset.targets) {
		t.Fatal("NewSequenceDataset target storage aliases constructor target")
	}
	if &sourceValues[0] == &sourceLengths.values[0] ||
		&sourceLengths.values[0] == &dataset.lengths.values[0] {
		t.Fatal("sequence length constructors retained caller storage")
	}
	if accessorInputs, err = dataset.Inputs(); err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}
	if accessorTargets, err = dataset.Targets(); err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}
	if accessorLengths, err = dataset.Lengths(); err != nil {
		t.Fatalf("Lengths returned error: %v", err)
	}
	if matrixStoragePointer(accessorInputs) == matrixStoragePointer(dataset.inputs) ||
		matrixStoragePointer(accessorTargets) == matrixStoragePointer(dataset.targets) ||
		&accessorLengths.values[0] == &dataset.lengths.values[0] {
		t.Fatal("safe sequence accessors alias dataset storage")
	}

	err = dataset.WithRowView(1, 3, func(view *SequenceDatasetView) (viewErr error) {
		if inputView, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if targetView, viewErr = view.Targets(); viewErr != nil {
			return viewErr
		}
		if lengthView, viewErr = view.Lengths(); viewErr != nil {
			return viewErr
		}
		if matrixStoragePointer(inputView) !=
			matrixStoragePointer(dataset.inputs)+4*unsafe.Sizeof(float32(0)) {
			t.Fatal("sequence input view does not alias selected owner row")
		}
		if matrixStoragePointer(targetView) !=
			matrixStoragePointer(dataset.targets)+unsafe.Sizeof(float32(0)) {
			t.Fatal("sequence target view does not alias selected owner row")
		}
		if &lengthView[0] != &dataset.lengths.values[1] {
			t.Fatal("sequence length view does not alias selected owner row")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("WithRowView returned error: %v", err)
	}

	if batches, err = dataset.Batches(2, nil); err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}
	if train, test, err = dataset.Split(0.5, nil); err != nil {
		t.Fatalf("Split returned error: %v", err)
	}
	if matrixStoragePointer(batches[0].inputs) == matrixStoragePointer(dataset.inputs) ||
		&batches[0].lengths.values[0] == &dataset.lengths.values[0] {
		t.Fatal("safe sequence batch aliases dataset storage")
	}
	if matrixStoragePointer(train.inputs) == matrixStoragePointer(dataset.inputs) ||
		matrixStoragePointer(test.inputs) == matrixStoragePointer(dataset.inputs) ||
		&train.lengths.values[0] == &dataset.lengths.values[0] ||
		&test.lengths.values[0] == &dataset.lengths.values[0] {
		t.Fatal("safe sequence split aliases dataset storage")
	}

	err = batches[0].WithView(func(view *SequenceDatasetView) (viewErr error) {
		if inputView, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if lengthView, viewErr = view.Lengths(); viewErr != nil {
			return viewErr
		}
		if matrixStoragePointer(inputView) != matrixStoragePointer(batches[0].inputs) ||
			&lengthView[0] != &batches[0].lengths.values[0] {
			t.Fatal("sequence batch view does not alias its batch owner")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("batch WithView returned error: %v", err)
	}
}

func Test_SequenceViewsRejectInvalidPrivateAssociationBeforeCallback(t *testing.T) {
	type testcase struct {
		name string
		run  func(SequenceDatasetViewFunc) error
	}

	var (
		inputs       *matrix.Matrix
		narrowInputs *matrix.Matrix
		targets      *matrix.Matrix
		shortTargets *matrix.Matrix
		validLengths *SequenceLengths
		shortLengths *SequenceLengths
		zeroLengths  SequenceLengths
		dataset      SequenceDataset
		batch        SequenceBatch
		tests        []testcase
		err          error
	)

	inputs = mustOwnedMatrix(t, 2, 4, make([]float32, 8))
	narrowInputs = mustOwnedMatrix(t, 2, 3, make([]float32, 6))
	targets = mustOwnedMatrix(t, 2, 1, make([]float32, 2))
	shortTargets = mustOwnedMatrix(t, 1, 1, []float32{0})
	if validLengths, err = NewSequenceLengths(2, []int{1, 2}); err != nil {
		t.Fatalf("NewSequenceLengths returned error: %v", err)
	}
	if shortLengths, err = NewSequenceLengths(2, []int{1}); err != nil {
		t.Fatalf("NewSequenceLengths short returned error: %v", err)
	}
	zeroLengths.steps = 2
	zeroLengths.values = []int{0, 2}

	tests = []testcase{
		{
			name: "dataset target rows mismatch",
			run: func(use SequenceDatasetViewFunc) (runErr error) {
				dataset.inputs = inputs
				dataset.targets = shortTargets
				dataset.lengths = validLengths
				runErr = dataset.WithView(use)
				return runErr
			},
		},
		{
			name: "dataset length rows mismatch",
			run: func(use SequenceDatasetViewFunc) (runErr error) {
				dataset.inputs = inputs
				dataset.targets = targets
				dataset.lengths = shortLengths
				runErr = dataset.WithView(use)
				return runErr
			},
		},
		{
			name: "dataset input width indivisible",
			run: func(use SequenceDatasetViewFunc) (runErr error) {
				dataset.inputs = narrowInputs
				dataset.targets = targets
				dataset.lengths = validLengths
				runErr = dataset.WithView(use)
				return runErr
			},
		},
		{
			name: "dataset stored length invalid",
			run: func(use SequenceDatasetViewFunc) (runErr error) {
				dataset.inputs = inputs
				dataset.targets = targets
				dataset.lengths = &zeroLengths
				runErr = dataset.WithView(use)
				return runErr
			},
		},
		{
			name: "batch stored length invalid",
			run: func(use SequenceDatasetViewFunc) (runErr error) {
				batch.inputs = inputs
				batch.targets = targets
				batch.lengths = &zeroLengths
				runErr = batch.WithView(use)
				return runErr
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				called bool
				runErr error
			)

			runErr = tt.run(func(*SequenceDatasetView) (viewErr error) {
				called = true
				return nil
			})
			if runErr == nil {
				t.Fatal("WithView error = nil, want invalid association error")
			}
			if called {
				t.Fatal("WithView invoked callback for invalid association")
			}
		})
	}
}
