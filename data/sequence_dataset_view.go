package data

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func newSequenceDatasetView(
	prefix string,
	inputs,
	targets *matrix.Matrix,
	steps int,
	lengths []int,
	copied bool,
) (out *SequenceDatasetView, err error) {
	var (
		state sequenceDatasetViewState
		view  SequenceDatasetView
	)

	if err = validateSequenceViewData(inputs, targets, steps, lengths); err != nil {
		err = fmt.Errorf("%s association is invalid: %w", prefix, err)
		return nil, err
	}

	state.prefix = prefix
	state.inputs = inputs
	state.targets = targets
	state.steps = steps
	state.lengths = lengths
	state.copied = copied
	state.active = true
	view.state = &state
	return &view, nil
}

// SequenceDatasetView is a temporary read-only-intent view of aligned input,
// target, and logical-length values. The source sequence dataset or batch owns
// whole and contiguous-window storage. Accessors do not copy. A view and all
// value copies of it expire together when the publishing callback exits and
// must not be retained, mutated, or used concurrently. Its owner and
// overlapping aliases must remain unmodified until that callback exits.
//
// A retained length slice cannot be revoked by Go after expiry. Retaining or
// using one after the callback is unsupported even though its slice header may
// remain accessible.
type SequenceDatasetView struct {
	state *sequenceDatasetViewState
}

type sequenceDatasetViewState struct {
	prefix  string
	inputs  *matrix.Matrix
	targets *matrix.Matrix
	steps   int
	lengths []int
	copied  bool
	active  bool
}

// Validate reports whether the sequence view is active, aligned, and usable.
func (v *SequenceDatasetView) Validate() (err error) {
	err = v.validate()
	return err
}

// Inputs returns the active temporary input matrix without copying.
//
// The returned matrix aliases owner storage and expires with the
// SequenceDatasetView. It must not be retained, mutated, or used concurrently.
func (v *SequenceDatasetView) Inputs() (inputs *matrix.Matrix, err error) {
	if err = v.validate(); err != nil {
		return nil, err
	}

	inputs = v.state.inputs
	return inputs, nil
}

// Targets returns the active temporary target matrix without copying.
//
// The returned matrix aliases owner storage and expires with the
// SequenceDatasetView. It must not be retained, mutated, or used concurrently.
func (v *SequenceDatasetView) Targets() (targets *matrix.Matrix, err error) {
	if err = v.validate(); err != nil {
		return nil, err
	}

	targets = v.state.targets
	return targets, nil
}

// Lengths returns the active temporary logical-length slice without copying.
//
// The returned slice aliases owner storage. It must not be retained, mutated,
// sent to another goroutine, or used after the publishing callback returns.
// Go cannot revoke a retained slice header after the view expires.
func (v *SequenceDatasetView) Lengths() (lengths []int, err error) {
	if err = v.validate(); err != nil {
		return nil, err
	}

	lengths = v.state.lengths
	return lengths, nil
}

// SampleCount returns the active view's aligned row count.
func (v *SequenceDatasetView) SampleCount() (samples int) {
	if v.validState() == nil {
		return 0
	}

	samples = v.state.inputs.Rows()
	return samples
}

// InputSize returns the active view's flattened input columns per sample.
func (v *SequenceDatasetView) InputSize() (features int) {
	if v.validState() == nil {
		return 0
	}

	features = v.state.inputs.Cols()
	return features
}

// TargetSize returns the active view's target columns per sample.
func (v *SequenceDatasetView) TargetSize() (values int) {
	if v.validState() == nil {
		return 0
	}

	values = v.state.targets.Cols()
	return values
}

// Steps returns the active view's physical sequence step count.
func (v *SequenceDatasetView) Steps() (steps int) {
	if v.validState() == nil {
		return 0
	}

	steps = v.state.steps
	return steps
}

// Copied reports whether an explicitly permitted selection fallback prepared
// owned matrix and length storage for this active view. Whole and contiguous
// views return false, as do nil, zero, and expired views.
func (v *SequenceDatasetView) Copied() (copied bool) {
	if v.validState() == nil {
		return false
	}

	copied = v.state.copied
	return copied
}

func (v *SequenceDatasetView) validate() (err error) {
	var prefix string

	prefix = "data: sequence dataset view"
	if v == nil {
		err = errors.New("data: sequence dataset view is nil")
		return err
	}
	if v.state == nil {
		err = errors.New("data: sequence dataset view is invalid")
		return err
	}
	if v.state.prefix != "" {
		prefix = v.state.prefix
	}
	if !v.state.active {
		err = fmt.Errorf("%s is expired", prefix)
		return err
	}
	if err = validateSequenceViewData(
		v.state.inputs,
		v.state.targets,
		v.state.steps,
		v.state.lengths,
	); err != nil {
		err = fmt.Errorf("%s association is invalid: %w", prefix, err)
		return err
	}

	return nil
}

func (v *SequenceDatasetView) validState() (state *sequenceDatasetViewState) {
	if v == nil || v.validate() != nil {
		return nil
	}

	state = v.state
	return state
}

func (v *SequenceDatasetView) expire() {
	if v == nil || v.state == nil {
		return
	}

	v.state.active = false
	v.state.inputs = nil
	v.state.targets = nil
	v.state.steps = 0
	v.state.lengths = nil
	v.state.copied = false
}

func withSequenceDatasetRowView(
	prefix string,
	inputs,
	targets *matrix.Matrix,
	lengths *SequenceLengths,
	start,
	end int,
	use SequenceDatasetViewFunc,
) (err error) {
	var lengthView []int

	if err = validateDataViewBounds(prefix, start, end, inputs.Rows()); err != nil {
		return err
	}
	if use == nil {
		err = fmt.Errorf("%s callback is nil", prefix)
		return err
	}

	lengthView = lengths.values[start:end]
	err = inputs.WithRowView(start, end, func(inputView *matrix.Matrix) (inputErr error) {
		inputErr = targets.WithRowView(start, end, func(targetView *matrix.Matrix) (targetErr error) {
			var view *SequenceDatasetView

			if view, targetErr = newSequenceDatasetView(
				prefix,
				inputView,
				targetView,
				lengths.Steps(),
				lengthView,
				false,
			); targetErr != nil {
				return targetErr
			}
			defer view.expire()
			if targetErr = use(view); targetErr != nil {
				targetErr = fmt.Errorf("%s callback: %w", prefix, targetErr)
				return targetErr
			}
			return nil
		})
		if inputErr != nil {
			inputErr = fmt.Errorf("%s targets: %w", prefix, inputErr)
			return inputErr
		}
		return nil
	})
	if err != nil {
		err = fmt.Errorf("%s inputs: %w", prefix, err)
		return err
	}
	return nil
}

func validateSequenceViewData(
	inputs,
	targets *matrix.Matrix,
	steps int,
	lengths []int,
) (err error) {
	if err = validateMatrixPair("view inputs", inputs, "view targets", targets); err != nil {
		return err
	}
	if err = validateSequenceLengthValues(steps, lengths); err != nil {
		return err
	}
	if len(lengths) != inputs.Rows() {
		err = fmt.Errorf(
			"data: sequence view sample count mismatch: inputs rows=%d, lengths=%d",
			inputs.Rows(),
			len(lengths),
		)
		return err
	}
	if inputs.Cols()%steps != 0 {
		err = fmt.Errorf(
			"data: sequence view input width must be divisible by steps: inputCols=%d steps=%d",
			inputs.Cols(),
			steps,
		)
		return err
	}

	return nil
}
