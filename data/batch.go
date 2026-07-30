package data

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// newBatch stores matrices that are already owned by the data package.
func newBatch(inputs, targets *matrix.Matrix) (out *Batch, err error) {
	var b Batch

	if err = validateMatrixPair("batch inputs", inputs, "batch targets", targets); err != nil {
		return nil, err
	}

	b.inputs = inputs
	b.targets = targets
	return &b, nil
}

// Batch contains paired input and target matrices for one mini-batch.
type Batch struct {
	inputs  *matrix.Matrix
	targets *matrix.Matrix
}

// WithView calls use with a temporary read-only-intent view of all paired
// inputs and targets. The batch owns the aliased storage. The view expires when
// use returns or panics and must not be retained, mutated, or used
// concurrently. The batch and overlapping aliases must remain unmodified until
// use returns. Concurrent or reentrant access is unsupported.
func (b *Batch) WithView(use DatasetViewFunc) (err error) {
	if err = b.validate(); err != nil {
		err = fmt.Errorf("data: batch view owner is invalid: %w", err)
		return err
	}

	err = withDatasetRowView(
		"data: batch view",
		b.inputs,
		b.targets,
		0,
		b.SampleCount(),
		false,
		use,
	)
	return err
}

// WithRowView calls use with a temporary read-only-intent view of paired rows
// [start, end). The batch owns the aliased storage. The view expires when use
// returns or panics and must not be retained, mutated, or used concurrently.
// The batch and overlapping aliases must remain unmodified until use returns.
// Concurrent or reentrant access is unsupported.
func (b *Batch) WithRowView(start, end int, use DatasetViewFunc) (err error) {
	if err = b.validate(); err != nil {
		err = fmt.Errorf("data: batch view owner is invalid: %w", err)
		return err
	}

	err = withDatasetRowView(
		"data: batch view",
		b.inputs,
		b.targets,
		start,
		end,
		false,
		use,
	)
	return err
}

// WithSelectedRows calls use with paired rows in index order. Contiguous rows
// alias batch storage. Other valid selections are rejected by ViewOnly or
// copied explicitly by ViewOrCopy. The temporary view expires when use
// returns and must not be retained, mutated, or used concurrently. The batch
// and overlapping aliases must remain unmodified; concurrent or reentrant
// access is unsupported.
func (b *Batch) WithSelectedRows(
	indexes []int,
	policy ViewPolicy,
	use DatasetViewFunc,
) (err error) {
	var (
		start      int
		end        int
		contiguous bool
	)

	if err = b.validate(); err != nil {
		err = fmt.Errorf("data: batch view owner is invalid: %w", err)
		return err
	}
	if start, end, contiguous, err = validateSelectedRows(
		"data: batch view",
		indexes,
		b.SampleCount(),
	); err != nil {
		return err
	}
	if err = validateViewPolicy("data: batch view", policy); err != nil {
		return err
	}
	if use == nil {
		err = fmt.Errorf("data: batch view callback is nil")
		return err
	}
	if !contiguous && policy == ViewOnly {
		err = fmt.Errorf("data: batch view selection is non-contiguous under ViewOnly")
		return err
	}
	if contiguous {
		err = withDatasetRowView(
			"data: batch view",
			b.inputs,
			b.targets,
			start,
			end,
			false,
			use,
		)
		return err
	}

	err = withSelectedDatasetRows(
		"data: batch view",
		b.inputs,
		b.targets,
		indexes,
		use,
	)
	return err
}

// Inputs returns a copy of the batch inputs.
func (b *Batch) Inputs() (inputs *matrix.Matrix, err error) {
	if err = b.validate(); err != nil {
		return nil, err
	}

	inputs, err = b.inputs.Clone()
	return inputs, err
}

// Targets returns a copy of the batch targets.
func (b *Batch) Targets() (targets *matrix.Matrix, err error) {
	if err = b.validate(); err != nil {
		return nil, err
	}

	targets, err = b.targets.Clone()
	return targets, err
}

// InputsInto copies batch inputs into inputs.
//
// The destination must match the batch input shape. Values are copied, so
// mutating the caller-owned destination does not mutate the batch. Valid calls
// fully overwrite the destination without allocating or retaining it.
func (b *Batch) InputsInto(inputs *matrix.Matrix) (err error) {
	if err = b.validate(); err != nil {
		return err
	}

	if err = inputs.CopyFrom(b.inputs); err != nil {
		err = fmt.Errorf("data: copy batch inputs into destination: %w", err)
		return err
	}

	return nil
}

// TargetsInto copies batch targets into targets.
//
// The destination must match the batch target shape. Values are copied, so
// mutating the caller-owned destination does not mutate the batch. Valid calls
// fully overwrite the destination without allocating or retaining it.
func (b *Batch) TargetsInto(targets *matrix.Matrix) (err error) {
	if err = b.validate(); err != nil {
		return err
	}

	if err = targets.CopyFrom(b.targets); err != nil {
		err = fmt.Errorf("data: copy batch targets into destination: %w", err)
		return err
	}

	return nil
}

// SampleCount returns the number of paired samples in the batch.
func (b *Batch) SampleCount() (samples int) {
	if b == nil || b.inputs == nil {
		return 0
	}

	samples = b.inputs.Rows()
	return samples
}

func (b *Batch) validate() (err error) {
	if b == nil {
		err = nilBatchError()
		return err
	}

	err = validateMatrixPair("batch inputs", b.inputs, "batch targets", b.targets)
	return err
}
