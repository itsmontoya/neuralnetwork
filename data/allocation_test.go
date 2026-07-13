package data_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var allocationDataBatches []*data.Batch
var allocationDataMatrix *matrix.Matrix

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

func requireMaxAllocs(tb testing.TB, name string, max float64, run func()) {
	var got float64

	tb.Helper()

	got = testing.AllocsPerRun(100, run)
	if got > max {
		tb.Fatalf("%s allocations = %.0f, want <= %.0f", name, got, max)
	}
}
