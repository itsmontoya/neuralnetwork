package data

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// newSequenceBatch stores values that are already owned by the data package.
func newSequenceBatch(
	inputs, targets *matrix.Matrix,
	lengths *SequenceLengths,
) (out *SequenceBatch, err error) {
	var batch SequenceBatch

	if err = validateSequenceData(
		"sequence batch",
		"sequence batch inputs",
		inputs,
		"sequence batch targets",
		targets,
		lengths,
	); err != nil {
		return nil, err
	}

	batch.inputs = inputs
	batch.targets = targets
	batch.lengths = lengths
	return &batch, nil
}

// SequenceBatch contains aligned values for one sequence mini-batch.
type SequenceBatch struct {
	inputs  *matrix.Matrix
	targets *matrix.Matrix
	lengths *SequenceLengths
}

// WithView calls use with a temporary read-only-intent view of all aligned
// inputs, targets, and logical lengths. The batch owns the aliased storage. The
// view expires when use returns or panics and must not be retained, mutated, or
// used concurrently. The batch and overlapping aliases must remain unmodified
// until use returns. Concurrent or reentrant access is unsupported.
func (b *SequenceBatch) WithView(use SequenceDatasetViewFunc) (err error) {
	if err = b.validate(); err != nil {
		err = fmt.Errorf("data: sequence batch view owner is invalid: %w", err)
		return err
	}

	err = withSequenceDatasetRowView(
		"data: sequence batch view",
		b.inputs,
		b.targets,
		b.lengths,
		0,
		b.SampleCount(),
		false,
		use,
	)
	return err
}

// WithRowView calls use with a temporary read-only-intent view of aligned rows
// [start, end). The batch owns the aliased matrix and logical-length storage.
// The view expires when use returns or panics and must not be retained,
// mutated, or used concurrently. The batch and overlapping aliases must remain
// unmodified until use returns. Concurrent or reentrant access is unsupported.
func (b *SequenceBatch) WithRowView(
	start,
	end int,
	use SequenceDatasetViewFunc,
) (err error) {
	if err = b.validate(); err != nil {
		err = fmt.Errorf("data: sequence batch view owner is invalid: %w", err)
		return err
	}

	err = withSequenceDatasetRowView(
		"data: sequence batch view",
		b.inputs,
		b.targets,
		b.lengths,
		start,
		end,
		false,
		use,
	)
	return err
}

// WithSelectedRows calls use with aligned rows in index order. Contiguous rows
// alias batch storage. Other valid selections are rejected by ViewOnly or
// copied explicitly by ViewOrCopy. The temporary aligned view expires when use
// returns and must not be retained, mutated, or used concurrently. The batch
// and overlapping aliases must remain unmodified; concurrent or reentrant
// access is unsupported.
func (b *SequenceBatch) WithSelectedRows(
	indexes []int,
	policy ViewPolicy,
	use SequenceDatasetViewFunc,
) (err error) {
	var (
		start      int
		end        int
		contiguous bool
	)

	if err = b.validate(); err != nil {
		err = fmt.Errorf("data: sequence batch view owner is invalid: %w", err)
		return err
	}
	if start, end, contiguous, err = validateSelectedRows(
		"data: sequence batch view",
		indexes,
		b.SampleCount(),
	); err != nil {
		return err
	}
	if err = validateViewPolicy("data: sequence batch view", policy); err != nil {
		return err
	}
	if use == nil {
		err = fmt.Errorf("data: sequence batch view callback is nil")
		return err
	}
	if !contiguous && policy == ViewOnly {
		err = fmt.Errorf(
			"data: sequence batch view selection is non-contiguous under ViewOnly",
		)
		return err
	}
	if contiguous {
		err = withSequenceDatasetRowView(
			"data: sequence batch view",
			b.inputs,
			b.targets,
			b.lengths,
			start,
			end,
			false,
			use,
		)
		return err
	}

	err = withSelectedSequenceDatasetRows(
		"data: sequence batch view",
		b.inputs,
		b.targets,
		b.lengths,
		indexes,
		use,
	)
	return err
}

// Inputs returns a copy of the batch inputs.
func (b *SequenceBatch) Inputs() (inputs *matrix.Matrix, err error) {
	if err = b.validate(); err != nil {
		return nil, err
	}

	if inputs, err = b.inputs.Clone(); err != nil {
		err = fmt.Errorf("data: sequence batch copy inputs: %w", err)
		return nil, err
	}

	return inputs, nil
}

// Targets returns a copy of the batch targets.
func (b *SequenceBatch) Targets() (targets *matrix.Matrix, err error) {
	if err = b.validate(); err != nil {
		return nil, err
	}

	if targets, err = b.targets.Clone(); err != nil {
		err = fmt.Errorf("data: sequence batch copy targets: %w", err)
		return nil, err
	}

	return targets, nil
}

// Lengths returns an independent validated copy of the batch lengths.
func (b *SequenceBatch) Lengths() (lengths *SequenceLengths, err error) {
	var values []int

	if err = b.validate(); err != nil {
		return nil, err
	}

	if values, err = b.lengths.Values(); err != nil {
		return nil, err
	}

	lengths, err = newSequenceLengths(b.lengths.Steps(), values)
	return lengths, err
}

// InputsInto copies batch inputs into inputs.
//
// The destination must match the batch input shape. A valid call fully
// overwrites the caller-owned destination without allocating or retaining it.
func (b *SequenceBatch) InputsInto(inputs *matrix.Matrix) (err error) {
	if err = b.validate(); err != nil {
		return err
	}

	if err = validateMatrixShape(
		"sequence batch inputs destination",
		inputs,
		b.inputs.Rows(),
		b.inputs.Cols(),
	); err != nil {
		return err
	}

	if err = inputs.CopyFrom(b.inputs); err != nil {
		err = fmt.Errorf("data: copy sequence batch inputs into destination: %w", err)
		return err
	}

	return nil
}

// TargetsInto copies batch targets into targets.
//
// The destination must match the batch target shape. A valid call fully
// overwrites the caller-owned destination without allocating or retaining it.
func (b *SequenceBatch) TargetsInto(targets *matrix.Matrix) (err error) {
	if err = b.validate(); err != nil {
		return err
	}

	if err = validateMatrixShape(
		"sequence batch targets destination",
		targets,
		b.targets.Rows(),
		b.targets.Cols(),
	); err != nil {
		return err
	}

	if err = targets.CopyFrom(b.targets); err != nil {
		err = fmt.Errorf("data: copy sequence batch targets into destination: %w", err)
		return err
	}

	return nil
}

// LengthsInto copies batch logical lengths into lengths.
//
// The destination must match SampleCount. A valid call fully overwrites the
// caller-owned destination without allocating or retaining it.
func (b *SequenceBatch) LengthsInto(lengths []int) (err error) {
	if err = b.validate(); err != nil {
		return err
	}

	err = b.lengths.ValuesInto(lengths)
	return err
}

// SampleCount returns the number of aligned samples.
func (b *SequenceBatch) SampleCount() (samples int) {
	if b == nil || b.inputs == nil {
		return 0
	}

	samples = b.inputs.Rows()
	return samples
}

// Steps returns the physical sequence step count.
func (b *SequenceBatch) Steps() (steps int) {
	if b == nil || b.lengths == nil {
		return 0
	}

	steps = b.lengths.Steps()
	return steps
}

func (b *SequenceBatch) validate() (err error) {
	if b == nil {
		err = fmt.Errorf("data: sequence batch is nil")
		return err
	}

	err = validateSequenceData(
		"sequence batch",
		"sequence batch inputs",
		b.inputs,
		"sequence batch targets",
		b.targets,
		b.lengths,
	)
	return err
}
