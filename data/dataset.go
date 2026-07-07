// Package data provides in-memory supervised learning helpers.
package data

import (
	"fmt"
	"math/rand"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// NewDataset constructs an in-memory supervised dataset by copying inputs and targets.
func NewDataset(inputs, targets *matrix.Matrix) (out *Dataset, err error) {
	var (
		clonedInputs  *matrix.Matrix
		clonedTargets *matrix.Matrix
		d             Dataset
	)

	if err = validateMatrixPair("inputs", inputs, "targets", targets); err != nil {
		return nil, err
	}

	if clonedInputs, err = inputs.Clone(); err != nil {
		return nil, err
	}

	if clonedTargets, err = targets.Clone(); err != nil {
		return nil, err
	}

	d.inputs = clonedInputs
	d.targets = clonedTargets
	return &d, nil
}

// Dataset stores paired input and target matrices for supervised learning.
type Dataset struct {
	inputs  *matrix.Matrix
	targets *matrix.Matrix
}

// Inputs returns a copy of the dataset inputs.
func (d *Dataset) Inputs() (inputs *matrix.Matrix, err error) {
	if err = d.validate(); err != nil {
		return nil, err
	}

	inputs, err = d.inputs.Clone()
	return inputs, err
}

// Targets returns a copy of the dataset targets.
func (d *Dataset) Targets() (targets *matrix.Matrix, err error) {
	if err = d.validate(); err != nil {
		return nil, err
	}

	targets, err = d.targets.Clone()
	return targets, err
}

// SampleCount returns the number of paired samples in the dataset.
func (d *Dataset) SampleCount() (samples int) {
	if d == nil || d.inputs == nil {
		return 0
	}

	samples = d.inputs.Rows()
	return samples
}

// InputSize returns the number of input features per sample.
func (d *Dataset) InputSize() (features int) {
	if d == nil || d.inputs == nil {
		return 0
	}

	features = d.inputs.Cols()
	return features
}

// TargetSize returns the number of target values per sample.
func (d *Dataset) TargetSize() (values int) {
	if d == nil || d.targets == nil {
		return 0
	}

	values = d.targets.Cols()
	return values
}

// Batches returns fixed-size mini-batches.
//
// When random is not nil, rows are shuffled with the provided source before
// batching. A nil random source preserves dataset order. The final batch may
// contain fewer than batchSize samples.
func (d *Dataset) Batches(batchSize int, random *rand.Rand) (batches []*Batch, err error) {
	var (
		indexes      []int
		start        int
		end          int
		batchCount   int
		batchInputs  *matrix.Matrix
		batchTargets *matrix.Matrix
		batch        *Batch
	)

	if err = d.validate(); err != nil {
		return nil, err
	}

	if batchSize <= 0 {
		err = fmt.Errorf("data: batch size must be positive: batchSize=%d", batchSize)
		return nil, err
	}

	indexes = rowIndexes(d.inputs.Rows())
	shuffleIndexes(indexes, random)
	batchCount = (len(indexes) + batchSize - 1) / batchSize
	batches = make([]*Batch, 0, batchCount)

	for start = 0; start < len(indexes); start += batchSize {
		end = start + batchSize
		if end > len(indexes) {
			end = len(indexes)
		}

		if batchInputs, err = matrixRows(d.inputs, indexes[start:end]); err != nil {
			return nil, err
		}

		if batchTargets, err = matrixRows(d.targets, indexes[start:end]); err != nil {
			return nil, err
		}

		if batch, err = newBatch(batchInputs, batchTargets); err != nil {
			return nil, err
		}

		batches = append(batches, batch)
	}

	return batches, nil
}

// Split returns train and test datasets from a deterministic row split.
//
// testFraction must be greater than 0 and less than 1. The test sample count
// is floored from sampleCount*testFraction and must leave at least one sample
// in each split. When random is not nil, rows are shuffled with the provided
// source before splitting. A nil random source preserves dataset order.
func (d *Dataset) Split(testFraction float64, random *rand.Rand) (train, test *Dataset, err error) {
	var (
		sampleCount  int
		testCount    int
		trainCount   int
		indexes      []int
		trainInputs  *matrix.Matrix
		trainTargets *matrix.Matrix
		testInputs   *matrix.Matrix
		testTargets  *matrix.Matrix
		trainIndexes []int
		testIndexes  []int
	)

	if err = d.validate(); err != nil {
		return nil, nil, err
	}

	if testFraction <= 0 || testFraction >= 1 {
		err = fmt.Errorf("data: test fraction must be greater than 0 and less than 1: testFraction=%g", testFraction)
		return nil, nil, err
	}

	sampleCount = d.inputs.Rows()
	testCount = int(float64(sampleCount) * testFraction)
	trainCount = sampleCount - testCount
	if testCount == 0 || trainCount == 0 {
		err = fmt.Errorf("data: test fraction must produce non-empty splits: samples=%d testFraction=%g", sampleCount, testFraction)
		return nil, nil, err
	}

	indexes = rowIndexes(sampleCount)
	shuffleIndexes(indexes, random)
	trainIndexes = indexes[:trainCount]
	testIndexes = indexes[trainCount:]

	if trainInputs, err = matrixRows(d.inputs, trainIndexes); err != nil {
		return nil, nil, err
	}

	if trainTargets, err = matrixRows(d.targets, trainIndexes); err != nil {
		return nil, nil, err
	}

	if testInputs, err = matrixRows(d.inputs, testIndexes); err != nil {
		return nil, nil, err
	}

	if testTargets, err = matrixRows(d.targets, testIndexes); err != nil {
		return nil, nil, err
	}

	if train, err = NewDataset(trainInputs, trainTargets); err != nil {
		return nil, nil, err
	}

	if test, err = NewDataset(testInputs, testTargets); err != nil {
		return nil, nil, err
	}

	return train, test, nil
}

func (d *Dataset) validate() (err error) {
	if d == nil {
		err = fmt.Errorf("data: dataset is nil")
		return err
	}

	err = validateMatrixPair("inputs", d.inputs, "targets", d.targets)
	return err
}
