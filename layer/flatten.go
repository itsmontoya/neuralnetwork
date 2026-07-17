package layer

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/scratch"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// NewFlatten constructs a validating spatial-to-dense adapter.
func NewFlatten(inputShape SpatialShape) (out *Flatten, err error) {
	if err = inputShape.validate(); err != nil {
		err = fmt.Errorf("layer: flatten input shape invalid: %w", err)
		return nil, err
	}

	var f Flatten
	f.inputShape = inputShape
	return &f, nil
}

// Flatten marks the boundary between flattened spatial values and dense
// features without changing channels-first value order.
type Flatten struct {
	inputShape           SpatialShape
	outputPool           scratch.MatrixPool
	outputScratch        *matrix.Matrix
	inputGradientPool    scratch.MatrixPool
	inputGradientScratch *matrix.Matrix
	forwardRows          int
	forwardCalled        bool
}

// Forward copies flattened spatial rows without reordering their values.
func (f *Flatten) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var rows int

	if err = f.validate(); err != nil {
		return nil, err
	}

	if rows, err = f.validateInput(input); err != nil {
		return nil, err
	}

	if f.outputScratch, _, err = f.outputPool.Get(rows, f.OutputSize()); err != nil {
		err = fmt.Errorf("layer: flatten allocate output: %w", err)
		return nil, err
	}

	if f.outputScratch == input {
		if f.outputScratch, err = matrix.New(rows, f.OutputSize()); err != nil {
			err = fmt.Errorf("layer: flatten allocate non-aliasing output: %w", err)
			return nil, err
		}
	}

	if err = f.outputScratch.CopyFrom(input); err != nil {
		err = fmt.Errorf("layer: flatten copy input: %w", err)
		return nil, err
	}

	f.forwardRows = rows
	f.forwardCalled = true
	output = f.outputScratch
	return output, nil
}

// Backward copies dense gradients back across the spatial boundary.
func (f *Flatten) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	if err = f.validate(); err != nil {
		return nil, err
	}

	if !f.forwardCalled {
		err = errors.New("layer: flatten backward called before forward")
		return nil, err
	}

	if err = f.validateOutputGradient(outputGradient); err != nil {
		return nil, err
	}

	if f.inputGradientScratch, _, err = f.inputGradientPool.Get(f.forwardRows, f.OutputSize()); err != nil {
		err = fmt.Errorf("layer: flatten allocate input gradient: %w", err)
		return nil, err
	}

	if f.inputGradientScratch == outputGradient {
		if f.inputGradientScratch, err = matrix.New(f.forwardRows, f.OutputSize()); err != nil {
			err = fmt.Errorf("layer: flatten allocate non-aliasing input gradient: %w", err)
			return nil, err
		}
	}

	if err = f.inputGradientScratch.CopyFrom(outputGradient); err != nil {
		err = fmt.Errorf("layer: flatten copy output gradient: %w", err)
		return nil, err
	}

	inputGradient = f.inputGradientScratch
	return inputGradient, nil
}

// InputShape returns the configured spatial input shape.
func (f *Flatten) InputShape() (shape SpatialShape) {
	if f == nil {
		return shape
	}

	shape = f.inputShape
	return shape
}

// OutputSize returns the flattened feature count per batch row.
func (f *Flatten) OutputSize() (size int) {
	if f == nil {
		return 0
	}

	size = f.inputShape.Size()
	return size
}

func (f *Flatten) validate() (err error) {
	if f == nil {
		err = errors.New("layer: flatten layer is nil")
		return err
	}

	if err = f.inputShape.validate(); err != nil {
		err = fmt.Errorf("layer: flatten input shape invalid: %w", err)
		return err
	}

	return nil
}

func (f *Flatten) validateInput(input *matrix.Matrix) (rows int, err error) {
	var cols int

	if input == nil {
		err = errors.New("layer: flatten input is nil")
		return 0, err
	}

	if err = input.Validate(); err != nil {
		err = fmt.Errorf("layer: flatten input invalid: %w", err)
		return 0, err
	}

	rows, cols = input.Shape()
	if cols != f.OutputSize() {
		err = fmt.Errorf(
			"layer: flatten input shape mismatch: got %dx%d, want batch rows x %d",
			rows,
			cols,
			f.OutputSize(),
		)
		return 0, err
	}

	return rows, nil
}

func (f *Flatten) validateOutputGradient(outputGradient *matrix.Matrix) (err error) {
	var (
		rows int
		cols int
	)

	if outputGradient == nil {
		err = errors.New("layer: flatten output gradient is nil")
		return err
	}

	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("layer: flatten output gradient invalid: %w", err)
		return err
	}

	rows, cols = outputGradient.Shape()
	if rows != f.forwardRows || cols != f.OutputSize() {
		err = fmt.Errorf(
			"layer: flatten output gradient shape mismatch: got %dx%d, want %dx%d",
			rows,
			cols,
			f.forwardRows,
			f.OutputSize(),
		)
		return err
	}

	return nil
}
