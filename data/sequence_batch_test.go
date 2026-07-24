package data_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_SequenceBatch_AccessorsReturnOwnedCopies(t *testing.T) {
	var (
		batches       []*data.SequenceBatch
		batch         *data.SequenceBatch
		firstInputs   *matrix.Matrix
		firstTargets  *matrix.Matrix
		firstLengths  *data.SequenceLengths
		secondInputs  *matrix.Matrix
		secondTargets *matrix.Matrix
		secondLengths []int
		values        []int
		err           error
	)

	batches, err = mustSequenceDatasetWithSamples(t, 2).Batches(2, nil)
	if err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}
	batch = batches[0]

	firstInputs, err = batch.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	firstTargets, err = batch.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	firstLengths, err = batch.Lengths()
	if err != nil {
		t.Fatalf("Lengths returned error: %v", err)
	}

	if err = firstInputs.Set(0, 0, 99); err != nil {
		t.Fatalf("inputs Set returned error: %v", err)
	}

	if err = firstTargets.Set(0, 0, 99); err != nil {
		t.Fatalf("targets Set returned error: %v", err)
	}

	values, err = firstLengths.Values()
	if err != nil {
		t.Fatalf("Values returned error: %v", err)
	}
	values[0] = 1

	secondInputs = mustMatrix(t, 2, 4, make([]float32, 8))
	secondTargets = mustMatrix(t, 2, 1, make([]float32, 2))
	secondLengths = make([]int, 2)
	if err = batch.InputsInto(secondInputs); err != nil {
		t.Fatalf("InputsInto returned error: %v", err)
	}

	if err = batch.TargetsInto(secondTargets); err != nil {
		t.Fatalf("TargetsInto returned error: %v", err)
	}

	if err = batch.LengthsInto(secondLengths); err != nil {
		t.Fatalf("LengthsInto returned error: %v", err)
	}

	requireSequenceAlignment(t, secondInputs, secondTargets, secondLengths)
	if batch.SampleCount() != 2 {
		t.Fatalf("SampleCount = %d, want 2", batch.SampleCount())
	}

	if batch.Steps() != 2 {
		t.Fatalf("Steps = %d, want 2", batch.Steps())
	}
}

func Test_SequenceDatasetAndBatch_DestinationAccessorsRejectWrongSizes(t *testing.T) {
	type testcase struct {
		name string
		run  func(*data.SequenceDataset, *data.SequenceBatch) error
	}

	var (
		dataset *data.SequenceDataset
		batches []*data.SequenceBatch
		batch   *data.SequenceBatch
		tests   []testcase
		err     error
	)

	dataset = mustSequenceDatasetWithSamples(t, 2)
	batches, err = dataset.Batches(2, nil)
	if err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}
	batch = batches[0]

	tests = []testcase{
		{
			name: "dataset inputs",
			run: func(dataset *data.SequenceDataset, _ *data.SequenceBatch) (err error) {
				err = dataset.InputsInto(mustMatrix(t, 1, 4, make([]float32, 4)))
				return err
			},
		},
		{
			name: "dataset targets",
			run: func(dataset *data.SequenceDataset, _ *data.SequenceBatch) (err error) {
				err = dataset.TargetsInto(mustMatrix(t, 1, 1, []float32{0}))
				return err
			},
		},
		{
			name: "dataset lengths",
			run: func(dataset *data.SequenceDataset, _ *data.SequenceBatch) (err error) {
				err = dataset.LengthsInto(make([]int, 1))
				return err
			},
		},
		{
			name: "batch inputs",
			run: func(_ *data.SequenceDataset, batch *data.SequenceBatch) (err error) {
				err = batch.InputsInto(mustMatrix(t, 1, 4, make([]float32, 4)))
				return err
			},
		},
		{
			name: "batch targets",
			run: func(_ *data.SequenceDataset, batch *data.SequenceBatch) (err error) {
				err = batch.TargetsInto(mustMatrix(t, 1, 1, []float32{0}))
				return err
			},
		},
		{
			name: "batch lengths",
			run: func(_ *data.SequenceDataset, batch *data.SequenceBatch) (err error) {
				err = batch.LengthsInto(make([]int, 1))
				return err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotErr error

			gotErr = tt.run(dataset, batch)
			if gotErr == nil {
				t.Fatal("destination accessor error = nil, want error")
			}
		})
	}
}

func Test_SequenceBatch_ValidatesNilAndZeroValues(t *testing.T) {
	type testcase struct {
		name  string
		batch *data.SequenceBatch
	}

	var (
		zero  data.SequenceBatch
		tests []testcase
	)
	tests = []testcase{
		{name: "nil", batch: nil},
		{name: "zero", batch: &zero},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				inputs *matrix.Matrix
				err    error
			)

			inputs, err = tt.batch.Inputs()
			if err == nil {
				t.Fatal("Inputs error = nil, want error")
			}

			if inputs != nil {
				t.Fatal("Inputs returned a matrix on error")
			}

			if tt.batch.SampleCount() != 0 || tt.batch.Steps() != 0 {
				t.Fatal("invalid batch dimensions are not zero")
			}
		})
	}
}
