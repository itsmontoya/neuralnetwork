package data

import (
	"fmt"
	"math/rand"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// NewSequenceDataset constructs aligned sequence data by copying all values.
func NewSequenceDataset(
	inputs, targets *matrix.Matrix,
	lengths *SequenceLengths,
) (out *SequenceDataset, err error) {
	var (
		ownedInputs       *matrix.Matrix
		ownedTargets      *matrix.Matrix
		ownedLengthValues []int
		ownedLengths      *SequenceLengths
	)

	if err = validateSequenceData(
		"sequence dataset",
		"sequence dataset inputs",
		inputs,
		"sequence dataset targets",
		targets,
		lengths,
	); err != nil {
		return nil, err
	}

	if ownedInputs, err = inputs.Clone(); err != nil {
		err = fmt.Errorf("data: sequence dataset clone inputs: %w", err)
		return nil, err
	}

	if ownedTargets, err = targets.Clone(); err != nil {
		err = fmt.Errorf("data: sequence dataset clone targets: %w", err)
		return nil, err
	}

	if ownedLengthValues, err = lengths.Values(); err != nil {
		err = fmt.Errorf("data: sequence dataset copy lengths: %w", err)
		return nil, err
	}

	if ownedLengths, err = newSequenceLengths(lengths.Steps(), ownedLengthValues); err != nil {
		return nil, err
	}

	out, err = newSequenceDataset(ownedInputs, ownedTargets, ownedLengths)
	return out, err
}

// newSequenceDataset stores values that are already owned by the data package.
func newSequenceDataset(
	inputs, targets *matrix.Matrix,
	lengths *SequenceLengths,
) (out *SequenceDataset, err error) {
	var dataset SequenceDataset

	if err = validateSequenceData(
		"sequence dataset",
		"sequence dataset inputs",
		inputs,
		"sequence dataset targets",
		targets,
		lengths,
	); err != nil {
		return nil, err
	}

	dataset.inputs = inputs
	dataset.targets = targets
	dataset.lengths = lengths
	return &dataset, nil
}

// SequenceDataset stores aligned input, target, and logical-length values.
type SequenceDataset struct {
	inputs  *matrix.Matrix
	targets *matrix.Matrix
	lengths *SequenceLengths
}

// Validate reports whether the sequence dataset has valid aligned values.
func (d *SequenceDataset) Validate() (err error) {
	err = d.validate()
	return err
}

// Inputs returns a copy of the dataset inputs.
func (d *SequenceDataset) Inputs() (inputs *matrix.Matrix, err error) {
	if err = d.validate(); err != nil {
		return nil, err
	}

	if inputs, err = d.inputs.Clone(); err != nil {
		err = fmt.Errorf("data: sequence dataset copy inputs: %w", err)
		return nil, err
	}

	return inputs, nil
}

// Targets returns a copy of the dataset targets.
func (d *SequenceDataset) Targets() (targets *matrix.Matrix, err error) {
	if err = d.validate(); err != nil {
		return nil, err
	}

	if targets, err = d.targets.Clone(); err != nil {
		err = fmt.Errorf("data: sequence dataset copy targets: %w", err)
		return nil, err
	}

	return targets, nil
}

// Lengths returns an independent validated copy of the dataset lengths.
func (d *SequenceDataset) Lengths() (lengths *SequenceLengths, err error) {
	var values []int

	if err = d.validate(); err != nil {
		return nil, err
	}

	if values, err = d.lengths.Values(); err != nil {
		return nil, err
	}

	lengths, err = newSequenceLengths(d.lengths.Steps(), values)
	return lengths, err
}

// InputsInto copies dataset inputs into inputs.
//
// The destination must match the dataset input shape. A valid call fully
// overwrites the caller-owned destination without allocating or retaining it.
func (d *SequenceDataset) InputsInto(inputs *matrix.Matrix) (err error) {
	if err = d.validate(); err != nil {
		return err
	}

	if err = validateMatrixShape(
		"sequence dataset inputs destination",
		inputs,
		d.inputs.Rows(),
		d.inputs.Cols(),
	); err != nil {
		return err
	}

	if err = inputs.CopyFrom(d.inputs); err != nil {
		err = fmt.Errorf("data: copy sequence dataset inputs into destination: %w", err)
		return err
	}

	return nil
}

// TargetsInto copies dataset targets into targets.
//
// The destination must match the dataset target shape. A valid call fully
// overwrites the caller-owned destination without allocating or retaining it.
func (d *SequenceDataset) TargetsInto(targets *matrix.Matrix) (err error) {
	if err = d.validate(); err != nil {
		return err
	}

	if err = validateMatrixShape(
		"sequence dataset targets destination",
		targets,
		d.targets.Rows(),
		d.targets.Cols(),
	); err != nil {
		return err
	}

	if err = targets.CopyFrom(d.targets); err != nil {
		err = fmt.Errorf("data: copy sequence dataset targets into destination: %w", err)
		return err
	}

	return nil
}

// LengthsInto copies dataset logical lengths into lengths.
//
// The destination must match SampleCount. A valid call fully overwrites the
// caller-owned destination without allocating or retaining it.
func (d *SequenceDataset) LengthsInto(lengths []int) (err error) {
	if err = d.validate(); err != nil {
		return err
	}

	err = d.lengths.ValuesInto(lengths)
	return err
}

// SelectRowsInto copies aligned rows into caller-owned destinations.
//
// Rows are copied in index order, and repeated indexes duplicate rows. Valid
// calls fully overwrite all destinations without allocating or retaining them.
func (d *SequenceDataset) SelectRowsInto(
	indexes []int,
	inputs, targets *matrix.Matrix,
	lengths []int,
) (err error) {
	var sourceRow int

	if err = d.validate(); err != nil {
		return err
	}

	if len(indexes) == 0 {
		err = fmt.Errorf("data: sequence dataset row indexes are empty")
		return err
	}

	for _, sourceRow = range indexes {
		if sourceRow < 0 || sourceRow >= d.SampleCount() {
			err = fmt.Errorf(
				"data: sequence dataset row index out of range: row=%d rows=%d",
				sourceRow,
				d.SampleCount(),
			)
			return err
		}
	}

	if err = validateMatrixShape(
		"selected sequence inputs destination",
		inputs,
		len(indexes),
		d.inputs.Cols(),
	); err != nil {
		return err
	}

	if err = validateMatrixShape(
		"selected sequence targets destination",
		targets,
		len(indexes),
		d.targets.Cols(),
	); err != nil {
		return err
	}

	if inputs == targets {
		err = fmt.Errorf("data: selected sequence matrix destinations must not alias")
		return err
	}

	if len(lengths) != len(indexes) {
		err = fmt.Errorf(
			"data: selected sequence lengths destination length mismatch: got %d, want %d",
			len(lengths),
			len(indexes),
		)
		return err
	}

	if err = d.inputs.SelectRowsInto(indexes, inputs); err != nil {
		err = fmt.Errorf("data: select sequence input rows into destination: %w", err)
		return err
	}

	if err = d.targets.SelectRowsInto(indexes, targets); err != nil {
		err = fmt.Errorf("data: select sequence target rows into destination: %w", err)
		return err
	}

	err = d.lengths.SelectRowsInto(indexes, lengths)
	return err
}

// SampleCount returns the number of aligned samples.
func (d *SequenceDataset) SampleCount() (samples int) {
	if d == nil || d.inputs == nil {
		return 0
	}

	samples = d.inputs.Rows()
	return samples
}

// InputSize returns the flattened input width per sample.
func (d *SequenceDataset) InputSize() (features int) {
	if d == nil || d.inputs == nil {
		return 0
	}

	features = d.inputs.Cols()
	return features
}

// TargetSize returns the number of target values per sample.
func (d *SequenceDataset) TargetSize() (values int) {
	if d == nil || d.targets == nil {
		return 0
	}

	values = d.targets.Cols()
	return values
}

// Steps returns the physical sequence step count.
func (d *SequenceDataset) Steps() (steps int) {
	if d == nil || d.lengths == nil {
		return 0
	}

	steps = d.lengths.Steps()
	return steps
}

// Batches returns aligned sequence mini-batches.
//
// When random is not nil, rows are shuffled with the provided source before
// batching. A nil source preserves order. The final batch may be partial.
func (d *SequenceDataset) Batches(
	batchSize int,
	random *rand.Rand,
) (batches []*SequenceBatch, err error) {
	var (
		indexes      []int
		start        int
		end          int
		batchCount   int
		batchInputs  *matrix.Matrix
		batchTargets *matrix.Matrix
		batchLengths *SequenceLengths
		batch        *SequenceBatch
	)

	if err = d.validate(); err != nil {
		return nil, err
	}

	if batchSize <= 0 {
		err = fmt.Errorf("data: sequence dataset batch size must be positive: batchSize=%d", batchSize)
		return nil, err
	}

	indexes = rowIndexes(d.SampleCount())
	shuffleIndexes(indexes, random)
	batchCount = 1 + (len(indexes)-1)/batchSize
	batches = make([]*SequenceBatch, 0, batchCount)

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

		if batchLengths, err = d.lengths.selectRows(indexes[start:end]); err != nil {
			return nil, err
		}

		if batch, err = newSequenceBatch(batchInputs, batchTargets, batchLengths); err != nil {
			return nil, err
		}

		batches = append(batches, batch)
	}

	return batches, nil
}

// Split returns copied train and test sequence datasets.
//
// testFraction must be greater than 0 and less than 1 and must produce two
// non-empty splits. A non-nil random source shuffles rows before splitting.
func (d *SequenceDataset) Split(
	testFraction float32,
	random *rand.Rand,
) (train, test *SequenceDataset, err error) {
	var (
		sampleCount  int
		testCount    int
		trainCount   int
		indexes      []int
		trainInputs  *matrix.Matrix
		trainTargets *matrix.Matrix
		trainLengths *SequenceLengths
		testInputs   *matrix.Matrix
		testTargets  *matrix.Matrix
		testLengths  *SequenceLengths
		trainIndexes []int
		testIndexes  []int
	)

	if err = d.validate(); err != nil {
		return nil, nil, err
	}

	if !(testFraction > 0 && testFraction < 1) {
		err = fmt.Errorf(
			"data: sequence dataset test fraction must be greater than 0 and less than 1: testFraction=%g",
			testFraction,
		)
		return nil, nil, err
	}

	sampleCount = d.SampleCount()
	testCount = int(float32(sampleCount) * testFraction)
	trainCount = sampleCount - testCount
	if testCount == 0 || trainCount == 0 {
		err = fmt.Errorf(
			"data: sequence dataset test fraction must produce non-empty splits: samples=%d testFraction=%g",
			sampleCount,
			testFraction,
		)
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

	if trainLengths, err = d.lengths.selectRows(trainIndexes); err != nil {
		return nil, nil, err
	}

	if testInputs, err = matrixRows(d.inputs, testIndexes); err != nil {
		return nil, nil, err
	}

	if testTargets, err = matrixRows(d.targets, testIndexes); err != nil {
		return nil, nil, err
	}

	if testLengths, err = d.lengths.selectRows(testIndexes); err != nil {
		return nil, nil, err
	}

	if train, err = newSequenceDataset(trainInputs, trainTargets, trainLengths); err != nil {
		return nil, nil, err
	}

	if test, err = newSequenceDataset(testInputs, testTargets, testLengths); err != nil {
		return nil, nil, err
	}

	return train, test, nil
}

func (d *SequenceDataset) validate() (err error) {
	if d == nil {
		err = fmt.Errorf("data: sequence dataset is nil")
		return err
	}

	err = validateSequenceData(
		"sequence dataset",
		"sequence dataset inputs",
		d.inputs,
		"sequence dataset targets",
		d.targets,
		d.lengths,
	)
	return err
}

func validateSequenceData(
	name string,
	inputName string,
	inputs *matrix.Matrix,
	targetName string,
	targets *matrix.Matrix,
	lengths *SequenceLengths,
) (err error) {
	if err = validateMatrixPair(inputName, inputs, targetName, targets); err != nil {
		return err
	}

	if err = lengths.validate(); err != nil {
		err = fmt.Errorf("data: %s lengths are invalid: %w", name, err)
		return err
	}

	if lengths.SampleCount() != inputs.Rows() {
		err = fmt.Errorf(
			"data: %s sample count mismatch: inputs rows=%d, lengths=%d",
			name,
			inputs.Rows(),
			lengths.SampleCount(),
		)
		return err
	}

	if inputs.Cols()%lengths.Steps() != 0 {
		err = fmt.Errorf(
			"data: %s input width must be divisible by steps: inputCols=%d steps=%d",
			name,
			inputs.Cols(),
			lengths.Steps(),
		)
		return err
	}

	return nil
}
