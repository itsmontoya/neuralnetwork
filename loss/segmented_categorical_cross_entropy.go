package loss

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// NewSegmentedCategoricalCrossEntropy constructs a loss with validated widths.
func NewSegmentedCategoricalCrossEntropy(
	widths ...int,
) (out *SegmentedCategoricalCrossEntropy, err error) {
	var (
		index  int
		width  int
		total  int
		maxInt int
		loss   SegmentedCategoricalCrossEntropy
	)

	if len(widths) == 0 {
		err = errors.New("loss: segmented categorical cross entropy requires at least one width")
		return nil, err
	}

	maxInt = int(^uint(0) >> 1)
	for index, width = range widths {
		if width <= 0 {
			err = fmt.Errorf(
				"loss: segmented categorical cross entropy width %d must be positive: width=%d",
				index,
				width,
			)
			return nil, err
		}
		if width > maxInt-total {
			err = errors.New("loss: segmented categorical cross entropy width total overflows int")
			return nil, err
		}
		total += width
	}

	loss.widths = append([]int(nil), widths...)
	loss.total = total
	return &loss, nil
}

// SegmentedCategoricalCrossEntropy computes the mean cross entropy over samples
// and independent one-hot categorical segments.
//
// Predictions and targets must have shape [batchSize, sum(widths)]. Every target
// segment must contain exactly one value equal to 1 and all remaining values
// equal to 0. This loss models independent categorical heads, not multi-label
// binary classification.
type SegmentedCategoricalCrossEntropy struct {
	widths []int
	total  int
}

// Value returns the mean categorical cross entropy over samples and segments.
func (s *SegmentedCategoricalCrossEntropy) Value(
	predictions,
	targets *matrix.Matrix,
) (value float32, err error) {
	var rows int

	if rows, err = s.validateInputs(predictions, targets); err != nil {
		return 0, err
	}
	if err = predictions.Pairwise(targets, func(row, col int, prediction, target float32) (err error) {
		if target == 1 {
			value -= f32.Log(clampPrediction(prediction))
		}
		return nil
	}); err != nil {
		return 0, err
	}

	value /= float32(rows) * float32(len(s.widths))
	return value, nil
}

// Gradient returns the prediction gradient of the sample-and-segment mean loss.
func (s *SegmentedCategoricalCrossEntropy) Gradient(
	predictions,
	targets *matrix.Matrix,
) (gradient *matrix.Matrix, err error) {
	var (
		rows int
		cols int
	)

	if rows, cols, err = s.validateInputShapes(predictions, targets); err != nil {
		return nil, err
	}
	if gradient, err = matrix.New(rows, cols); err != nil {
		return nil, err
	}
	if err = s.gradientInto(predictions, targets, gradient, rows); err != nil {
		return nil, err
	}

	return gradient, nil
}

// GradientInto writes the prediction gradient into destination.
// It follows DestinationGradient's destination and alias contract without
// allocating.
func (s *SegmentedCategoricalCrossEntropy) GradientInto(
	predictions,
	targets,
	destination *matrix.Matrix,
) (err error) {
	var rows int

	if rows, _, err = s.validateInputShapes(predictions, targets); err != nil {
		return err
	}
	err = s.gradientInto(predictions, targets, destination, rows)
	return err
}

// Widths returns a copy of the ordered segment widths.
func (s *SegmentedCategoricalCrossEntropy) Widths() (widths []int) {
	if s == nil {
		return nil
	}

	widths = append([]int(nil), s.widths...)
	return widths
}

func (s *SegmentedCategoricalCrossEntropy) gradientInto(
	predictions,
	targets,
	destination *matrix.Matrix,
	rows int,
) (err error) {
	var scale float32

	if err = s.validateTargets(targets); err != nil {
		return err
	}

	scale = 1 / (float32(rows) * float32(len(s.widths)))
	err = predictions.PairwiseInto(
		targets,
		destination,
		func(row, col int, prediction, target float32) (value float32, err error) {
			if target == 0 {
				return 0, nil
			}

			value = -target / clampPrediction(prediction) * scale
			return value, nil
		},
	)
	return err
}

func (s *SegmentedCategoricalCrossEntropy) validate() (err error) {
	if s == nil {
		err = errors.New("loss: segmented categorical cross entropy is nil")
		return err
	}
	if len(s.widths) == 0 || s.total <= 0 {
		err = errors.New("loss: segmented categorical cross entropy is not configured")
		return err
	}

	return nil
}

func (s *SegmentedCategoricalCrossEntropy) validateInputShapes(
	predictions,
	targets *matrix.Matrix,
) (rows, cols int, err error) {
	if err = s.validate(); err != nil {
		return 0, 0, err
	}
	if rows, cols, err = matrixShapePair(predictions, targets); err != nil {
		return 0, 0, err
	}
	if cols != s.total {
		err = fmt.Errorf(
			"loss: segmented categorical cross entropy column mismatch: got %d, want %d",
			cols,
			s.total,
		)
		return 0, 0, err
	}

	return rows, cols, nil
}

func (s *SegmentedCategoricalCrossEntropy) validateInputs(
	predictions,
	targets *matrix.Matrix,
) (rows int, err error) {
	if rows, _, err = s.validateInputShapes(predictions, targets); err != nil {
		return 0, err
	}
	if err = s.validateTargets(targets); err != nil {
		return 0, err
	}

	return rows, nil
}

func (s *SegmentedCategoricalCrossEntropy) validateTargets(targets *matrix.Matrix) (err error) {
	var (
		currentRow    int
		segment       int
		segmentEnd    int
		selectedCount int
	)

	currentRow = -1
	segmentEnd = s.widths[0]
	err = targets.Pairwise(targets, func(row, col int, left, right float32) (err error) {
		if row != currentRow {
			if currentRow >= 0 {
				if err = validateSegmentSelection(currentRow, segment, selectedCount); err != nil {
					return err
				}
			}
			currentRow = row
			segment = 0
			segmentEnd = s.widths[0]
			selectedCount = 0
		}

		if col == segmentEnd {
			if err = validateSegmentSelection(row, segment, selectedCount); err != nil {
				return err
			}
			segment++
			segmentEnd += s.widths[segment]
			selectedCount = 0
		}

		if left == 1 {
			selectedCount++
			if selectedCount > 1 {
				err = fmt.Errorf(
					"loss: segmented categorical target row %d segment %d contains multiple selected classes",
					row,
					segment,
				)
				return err
			}
			return nil
		}
		if left != 0 {
			err = fmt.Errorf(
				"loss: segmented categorical target row %d segment %d column %d must be 0 or 1: value=%g",
				row,
				segment,
				col,
				left,
			)
			return err
		}

		return nil
	})
	if err != nil {
		return err
	}

	err = validateSegmentSelection(currentRow, segment, selectedCount)
	return err
}

func validateSegmentSelection(row, segment, selectedCount int) (err error) {
	if selectedCount == 1 {
		return nil
	}

	err = fmt.Errorf(
		"loss: segmented categorical target row %d segment %d must contain exactly one class: selected=%d",
		row,
		segment,
		selectedCount,
	)
	return err
}
