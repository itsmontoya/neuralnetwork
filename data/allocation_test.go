package data_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var allocationDataBatches []*data.Batch
var allocationDataMatrix *matrix.Matrix
var allocationSequenceBatches []*data.SequenceBatch
var allocationSequenceLengths *data.SequenceLengths

func Test_DatasetBatchAllocationCeilings(t *testing.T) {
	var (
		dataset *data.Dataset
		random  *rand.Rand
		err     error
	)

	dataset = mustDatasetWithSamples(t, 4)
	requireMaxAllocs(t, "Dataset.Batches unshuffled", 12, func() {
		allocationDataBatches, err = dataset.Batches(2, nil)
		if err != nil {
			panic(err)
		}
	})

	random = rand.New(rand.NewSource(7))
	requireMaxAllocs(t, "Dataset.Batches shuffled", 12, func() {
		allocationDataBatches, err = dataset.Batches(2, random)
		if err != nil {
			panic(err)
		}
	})
}

func Test_DataCopyAccessorAllocationCeilings(t *testing.T) {
	var (
		dataset            *data.Dataset
		batches            []*data.Batch
		batch              *data.Batch
		indexes            []int
		inputsDestination  *matrix.Matrix
		targetsDestination *matrix.Matrix
		err                error
	)

	dataset = mustDatasetWithSamples(t, 4)
	batches, err = dataset.Batches(2, nil)
	if err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}

	batch = batches[0]
	indexes = []int{3, 1}
	inputsDestination = mustMatrix(t, 2, 2, []float32{0, 0, 0, 0})
	targetsDestination = mustMatrix(t, 2, 1, []float32{0, 0})
	requireMaxAllocs(t, "Dataset.SelectRowsInto", 0, func() {
		if err = dataset.SelectRowsInto(indexes, inputsDestination, targetsDestination); err != nil {
			panic(err)
		}
	})

	inputsDestination = mustMatrix(t, 4, 2, []float32{0, 0, 0, 0, 0, 0, 0, 0})
	targetsDestination = mustMatrix(t, 4, 1, []float32{0, 0, 0, 0})
	requireMaxAllocs(t, "Dataset.InputsInto", 0, func() {
		if err = dataset.InputsInto(inputsDestination); err != nil {
			panic(err)
		}
	})

	requireMaxAllocs(t, "Dataset.TargetsInto", 0, func() {
		if err = dataset.TargetsInto(targetsDestination); err != nil {
			panic(err)
		}
	})

	requireMaxAllocs(t, "Dataset.Inputs", 2, func() {
		allocationDataMatrix, err = dataset.Inputs()
		if err != nil {
			panic(err)
		}
	})

	requireMaxAllocs(t, "Dataset.Targets", 2, func() {
		allocationDataMatrix, err = dataset.Targets()
		if err != nil {
			panic(err)
		}
	})

	inputsDestination = mustMatrix(t, 2, 2, []float32{0, 0, 0, 0})
	targetsDestination = mustMatrix(t, 2, 1, []float32{0, 0})
	requireMaxAllocs(t, "Batch.InputsInto", 0, func() {
		if err = batch.InputsInto(inputsDestination); err != nil {
			panic(err)
		}
	})

	requireMaxAllocs(t, "Batch.TargetsInto", 0, func() {
		if err = batch.TargetsInto(targetsDestination); err != nil {
			panic(err)
		}
	})

	requireMaxAllocs(t, "Batch.Inputs", 2, func() {
		allocationDataMatrix, err = batch.Inputs()
		if err != nil {
			panic(err)
		}
	})

	requireMaxAllocs(t, "Batch.Targets", 2, func() {
		allocationDataMatrix, err = batch.Targets()
		if err != nil {
			panic(err)
		}
	})
}

func Test_SequenceDataCopyAccessorAllocationCeilings(t *testing.T) {
	var (
		dataset            *data.SequenceDataset
		batches            []*data.SequenceBatch
		batch              *data.SequenceBatch
		lengths            *data.SequenceLengths
		indexes            []int
		inputsDestination  *matrix.Matrix
		targetsDestination *matrix.Matrix
		lengthDestination  []int
		err                error
	)

	dataset = mustSequenceDatasetWithSamples(t, 4)
	if lengths, err = dataset.Lengths(); err != nil {
		t.Fatalf("Lengths returned error: %v", err)
	}

	if batches, err = dataset.Batches(2, nil); err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}
	batch = batches[0]
	indexes = []int{3, 1}
	lengthDestination = make([]int, 4)

	requireMaxAllocs(t, "SequenceLengths.ValuesInto", 0, func() {
		if err = lengths.ValuesInto(lengthDestination); err != nil {
			panic(err)
		}
	})

	lengthDestination = make([]int, 2)
	requireMaxAllocs(t, "SequenceLengths.SelectRowsInto", 0, func() {
		if err = lengths.SelectRowsInto(indexes, lengthDestination); err != nil {
			panic(err)
		}
	})

	inputsDestination = mustMatrix(t, 2, 4, make([]float32, 8))
	targetsDestination = mustMatrix(t, 2, 1, make([]float32, 2))
	requireMaxAllocs(t, "SequenceDataset.SelectRowsInto", 0, func() {
		if err = dataset.SelectRowsInto(
			indexes,
			inputsDestination,
			targetsDestination,
			lengthDestination,
		); err != nil {
			panic(err)
		}
	})

	inputsDestination = mustMatrix(t, 4, 4, make([]float32, 16))
	targetsDestination = mustMatrix(t, 4, 1, make([]float32, 4))
	lengthDestination = make([]int, 4)
	requireMaxAllocs(t, "SequenceDataset.InputsInto", 0, func() {
		if err = dataset.InputsInto(inputsDestination); err != nil {
			panic(err)
		}
	})

	requireMaxAllocs(t, "SequenceDataset.TargetsInto", 0, func() {
		if err = dataset.TargetsInto(targetsDestination); err != nil {
			panic(err)
		}
	})

	requireMaxAllocs(t, "SequenceDataset.LengthsInto", 0, func() {
		if err = dataset.LengthsInto(lengthDestination); err != nil {
			panic(err)
		}
	})

	inputsDestination = mustMatrix(t, 2, 4, make([]float32, 8))
	targetsDestination = mustMatrix(t, 2, 1, make([]float32, 2))
	lengthDestination = make([]int, 2)
	requireMaxAllocs(t, "SequenceBatch.InputsInto", 0, func() {
		if err = batch.InputsInto(inputsDestination); err != nil {
			panic(err)
		}
	})

	requireMaxAllocs(t, "SequenceBatch.TargetsInto", 0, func() {
		if err = batch.TargetsInto(targetsDestination); err != nil {
			panic(err)
		}
	})

	requireMaxAllocs(t, "SequenceBatch.LengthsInto", 0, func() {
		if err = batch.LengthsInto(lengthDestination); err != nil {
			panic(err)
		}
	})
}

func Test_SequenceDatasetBatchAllocationCeilings(t *testing.T) {
	var (
		dataset *data.SequenceDataset
		random  *rand.Rand
		err     error
	)

	dataset = mustSequenceDatasetWithSamples(t, 4)
	requireMaxAllocs(t, "SequenceDataset.Batches unshuffled", 18, func() {
		allocationSequenceBatches, err = dataset.Batches(2, nil)
		if err != nil {
			panic(err)
		}
	})

	random = rand.New(rand.NewSource(7))
	requireMaxAllocs(t, "SequenceDataset.Batches shuffled", 18, func() {
		allocationSequenceBatches, err = dataset.Batches(2, random)
		if err != nil {
			panic(err)
		}
	})

	requireMaxAllocs(t, "SequenceDataset.Lengths", 2, func() {
		allocationSequenceLengths, err = dataset.Lengths()
		if err != nil {
			panic(err)
		}
	})
}

func requireMaxAllocs(tb testing.TB, name string, max float64, run func()) {
	var got float64

	tb.Helper()

	got = testing.AllocsPerRun(100, run)
	if got > max {
		tb.Fatalf("%s allocations = %.0f, want <= %.0f", name, got, max)
	}
}
