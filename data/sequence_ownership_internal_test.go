package data

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_NewSequenceDataset_StoresIndependentCopies(t *testing.T) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		lengths *SequenceLengths
		dataset *SequenceDataset
		err     error
	)

	inputs = mustOwnedMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	targets = mustOwnedMatrix(t, 2, 1, []float32{10, 20})
	lengths, err = NewSequenceLengths(2, []int{1, 2})
	if err != nil {
		t.Fatalf("NewSequenceLengths returned error: %v", err)
	}

	dataset, err = NewSequenceDataset(inputs, targets, lengths)
	if err != nil {
		t.Fatalf("NewSequenceDataset returned error: %v", err)
	}

	if dataset.inputs == inputs {
		t.Fatal("NewSequenceDataset retained the input matrix")
	}

	if dataset.targets == targets {
		t.Fatal("NewSequenceDataset retained the target matrix")
	}

	if dataset.lengths == lengths {
		t.Fatal("NewSequenceDataset retained the length carrier")
	}

	if &dataset.lengths.values[0] == &lengths.values[0] {
		t.Fatal("NewSequenceDataset retained the length storage")
	}
}

func Test_SequenceDataset_BatchesAndSplitOwnSelectedValues(t *testing.T) {
	var (
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		lengths      *SequenceLengths
		source       *SequenceDataset
		batches      []*SequenceBatch
		train        *SequenceDataset
		test         *SequenceDataset
		batchInputs  *matrix.Matrix
		trainInputs  *matrix.Matrix
		testInputs   *matrix.Matrix
		batchLengths []int
		trainLengths []int
		testLengths  []int
		err          error
	)

	inputs = mustOwnedMatrix(t, 4, 2, []float32{
		1, 10,
		2, 20,
		3, 30,
		4, 40,
	})
	targets = mustOwnedMatrix(t, 4, 1, []float32{101, 102, 103, 104})
	lengths, err = NewSequenceLengths(2, []int{1, 2, 1, 2})
	if err != nil {
		t.Fatalf("NewSequenceLengths returned error: %v", err)
	}

	source, err = NewSequenceDataset(inputs, targets, lengths)
	if err != nil {
		t.Fatalf("NewSequenceDataset returned error: %v", err)
	}

	batches, err = source.Batches(2, nil)
	if err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}

	train, test, err = source.Split(0.5, nil)
	if err != nil {
		t.Fatalf("Split returned error: %v", err)
	}

	if batches[0].inputs == source.inputs || train.inputs == source.inputs ||
		test.inputs == source.inputs || train.inputs == test.inputs {
		t.Fatal("selected input matrices share ownership")
	}

	if batches[0].targets == source.targets || train.targets == source.targets ||
		test.targets == source.targets || train.targets == test.targets {
		t.Fatal("selected target matrices share ownership")
	}

	if batches[0].lengths == source.lengths || train.lengths == source.lengths ||
		test.lengths == source.lengths || train.lengths == test.lengths {
		t.Fatal("selected length carriers share ownership")
	}

	if err = source.inputs.Set(0, 0, 99); err != nil {
		t.Fatalf("source inputs Set returned error: %v", err)
	}

	if err = source.targets.Set(0, 0, 99); err != nil {
		t.Fatalf("source targets Set returned error: %v", err)
	}
	source.lengths.values[0] = 2

	batchInputs, err = batches[0].Inputs()
	if err != nil {
		t.Fatalf("batch Inputs returned error: %v", err)
	}

	trainInputs, err = train.Inputs()
	if err != nil {
		t.Fatalf("train Inputs returned error: %v", err)
	}

	testInputs, err = test.Inputs()
	if err != nil {
		t.Fatalf("test Inputs returned error: %v", err)
	}

	batchLengths = make([]int, batches[0].SampleCount())
	if err = batches[0].LengthsInto(batchLengths); err != nil {
		t.Fatalf("batch LengthsInto returned error: %v", err)
	}

	trainLengths = make([]int, train.SampleCount())
	if err = train.LengthsInto(trainLengths); err != nil {
		t.Fatalf("train LengthsInto returned error: %v", err)
	}

	testLengths = make([]int, test.SampleCount())
	if err = test.LengthsInto(testLengths); err != nil {
		t.Fatalf("test LengthsInto returned error: %v", err)
	}

	requireOwnedMatrixValue(t, batchInputs, 0, 0, 1)
	requireOwnedMatrixValue(t, trainInputs, 0, 0, 1)
	requireOwnedMatrixValue(t, testInputs, 0, 0, 3)
	requireOwnedIntValues(t, batchLengths, []int{1, 2})
	requireOwnedIntValues(t, trainLengths, []int{1, 2})
	requireOwnedIntValues(t, testLengths, []int{1, 2})
}

func requireOwnedMatrixValue(
	tb testing.TB,
	m *matrix.Matrix,
	row, col int,
	want float32,
) {
	var (
		got float32
		err error
	)

	tb.Helper()

	got, err = m.At(row, col)
	if err != nil {
		tb.Fatalf("At returned error: %v", err)
	}

	if got != want {
		tb.Fatalf("value at %d,%d = %g, want %g", row, col, got, want)
	}
}

func requireOwnedIntValues(tb testing.TB, got, want []int) {
	var index int

	tb.Helper()

	if len(got) != len(want) {
		tb.Fatalf("value count = %d, want %d", len(got), len(want))
	}

	for index = range got {
		if got[index] != want[index] {
			tb.Fatalf("value %d = %d, want %d", index, got[index], want[index])
		}
	}
}
