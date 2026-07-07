package data

import "github.com/itsmontoya/neuralnetwork/matrix"

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
