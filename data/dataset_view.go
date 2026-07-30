package data

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func newDatasetView(
	prefix string,
	inputs,
	targets *matrix.Matrix,
	copied bool,
) (out *DatasetView, err error) {
	var (
		state datasetViewState
		view  DatasetView
	)

	if err = validateMatrixPair("view inputs", inputs, "view targets", targets); err != nil {
		err = fmt.Errorf("%s association is invalid: %w", prefix, err)
		return nil, err
	}

	state.prefix = prefix
	state.inputs = inputs
	state.targets = targets
	state.copied = copied
	state.active = true
	view.state = &state
	return &view, nil
}

// DatasetView is a temporary read-only-intent view of paired input and target
// matrices. The source dataset or batch owns whole and contiguous-window
// storage. Matrix accessors do not copy. A view and all value copies of it
// expire together when the publishing callback exits and must not be retained,
// mutated, or used concurrently. Its owner and overlapping aliases must remain
// unmodified until that callback exits.
type DatasetView struct {
	state *datasetViewState
}

type datasetViewState struct {
	prefix  string
	inputs  *matrix.Matrix
	targets *matrix.Matrix
	copied  bool
	active  bool
}

// Validate reports whether the paired view is active, aligned, and usable.
func (v *DatasetView) Validate() (err error) {
	err = v.validate()
	return err
}

// Inputs returns the active temporary input matrix without copying.
//
// The returned matrix aliases owner storage and expires with the DatasetView.
// It must not be retained, mutated, or used concurrently.
func (v *DatasetView) Inputs() (inputs *matrix.Matrix, err error) {
	if err = v.validate(); err != nil {
		return nil, err
	}

	inputs = v.state.inputs
	return inputs, nil
}

// Targets returns the active temporary target matrix without copying.
//
// The returned matrix aliases owner storage and expires with the DatasetView.
// It must not be retained, mutated, or used concurrently.
func (v *DatasetView) Targets() (targets *matrix.Matrix, err error) {
	if err = v.validate(); err != nil {
		return nil, err
	}

	targets = v.state.targets
	return targets, nil
}

// SampleCount returns the active view's paired row count.
func (v *DatasetView) SampleCount() (samples int) {
	if v.validState() == nil {
		return 0
	}

	samples = v.state.inputs.Rows()
	return samples
}

// InputSize returns the active view's input columns per sample.
func (v *DatasetView) InputSize() (features int) {
	if v.validState() == nil {
		return 0
	}

	features = v.state.inputs.Cols()
	return features
}

// TargetSize returns the active view's target columns per sample.
func (v *DatasetView) TargetSize() (values int) {
	if v.validState() == nil {
		return 0
	}

	values = v.state.targets.Cols()
	return values
}

// Copied reports whether an explicitly permitted selection fallback prepared
// owned matrix storage for this active view. Whole and contiguous views return
// false, as do nil, zero, and expired views.
func (v *DatasetView) Copied() (copied bool) {
	if v.validState() == nil {
		return false
	}

	copied = v.state.copied
	return copied
}

func (v *DatasetView) validate() (err error) {
	var prefix string

	prefix = "data: dataset view"
	if v == nil {
		err = errors.New("data: dataset view is nil")
		return err
	}
	if v.state == nil {
		err = errors.New("data: dataset view is invalid")
		return err
	}
	if v.state.prefix != "" {
		prefix = v.state.prefix
	}
	if !v.state.active {
		err = fmt.Errorf("%s is expired", prefix)
		return err
	}
	if err = validateMatrixPair(
		"view inputs",
		v.state.inputs,
		"view targets",
		v.state.targets,
	); err != nil {
		err = fmt.Errorf("%s association is invalid: %w", prefix, err)
		return err
	}

	return nil
}

func (v *DatasetView) validState() (state *datasetViewState) {
	if v == nil || v.validate() != nil {
		return nil
	}

	state = v.state
	return state
}

func (v *DatasetView) expire() {
	if v == nil || v.state == nil {
		return
	}

	v.state.active = false
	v.state.inputs = nil
	v.state.targets = nil
	v.state.copied = false
}

func withDatasetRowView(
	prefix string,
	inputs,
	targets *matrix.Matrix,
	start,
	end int,
	use DatasetViewFunc,
) (err error) {
	if err = validateDataViewBounds(prefix, start, end, inputs.Rows()); err != nil {
		return err
	}
	if use == nil {
		err = fmt.Errorf("%s callback is nil", prefix)
		return err
	}

	err = inputs.WithRowView(start, end, func(inputView *matrix.Matrix) (inputErr error) {
		inputErr = targets.WithRowView(start, end, func(targetView *matrix.Matrix) (targetErr error) {
			var view *DatasetView

			if view, targetErr = newDatasetView(prefix, inputView, targetView, false); targetErr != nil {
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

func validateDataViewBounds(prefix string, start, end, rows int) (err error) {
	if start < 0 {
		err = fmt.Errorf("%s start must be non-negative: start=%d", prefix, start)
		return err
	}
	if end <= start {
		err = fmt.Errorf("%s must contain at least one row: start=%d end=%d", prefix, start, end)
		return err
	}
	if end > rows {
		err = fmt.Errorf("%s end is out of range: end=%d rows=%d", prefix, end, rows)
		return err
	}

	return nil
}
