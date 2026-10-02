package activation

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// NewSegmentedSoftmax constructs a SegmentedSoftmax with validated widths.
func NewSegmentedSoftmax(widths ...int) (out *SegmentedSoftmax, err error) {
	var (
		index  int
		width  int
		total  int
		maxInt int
		s      SegmentedSoftmax
	)

	if len(widths) == 0 {
		err = errors.New("activation: segmented softmax requires at least one width")
		return nil, err
	}

	maxInt = int(^uint(0) >> 1)
	for index, width = range widths {
		if width <= 0 {
			err = fmt.Errorf("activation: segmented softmax width %d must be positive: width=%d", index, width)
			return nil, err
		}
		if width > maxInt-total {
			err = errors.New("activation: segmented softmax width total overflows int")
			return nil, err
		}
		total += width
	}

	s.widths = append([]int(nil), widths...)
	s.total = total
	return &s, nil
}

// SegmentedSoftmax applies independent row-wise softmax operations to ordered
// column segments.
type SegmentedSoftmax struct {
	widths []int
	total  int
}

// Forward returns independently normalized probabilities for each row segment.
func (s *SegmentedSoftmax) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var rows int

	if rows, err = s.validateInput("input", input); err != nil {
		return nil, err
	}
	if output, err = matrix.New(rows, s.total); err != nil {
		return nil, err
	}
	if err = s.ForwardInto(input, output); err != nil {
		return nil, err
	}

	return output, nil
}

// Backward applies each segment's softmax Jacobian independently.
func (s *SegmentedSoftmax) Backward(
	input,
	outputGradient *matrix.Matrix,
) (inputGradient *matrix.Matrix, err error) {
	var rows int

	if rows, _, err = s.validatePair(input, outputGradient); err != nil {
		return nil, err
	}
	if inputGradient, err = matrix.New(rows, s.total); err != nil {
		return nil, err
	}
	if err = s.BackwardInto(input, outputGradient, inputGradient); err != nil {
		return nil, err
	}

	return inputGradient, nil
}

// ForwardInto writes independently normalized row segments into output.
// It follows DestinationActivation's destination and alias contract without
// allocating.
func (s *SegmentedSoftmax) ForwardInto(input, output *matrix.Matrix) (err error) {
	if _, err = s.validateInput("input", input); err != nil {
		return err
	}
	if err = input.SegmentedSoftmaxRowsInto(s.widths, output); err != nil {
		err = fmt.Errorf("activation: segmented softmax forward failed: %w", err)
		return err
	}

	return nil
}

// BackwardInto writes each segment's propagated output gradient into
// inputGradient. It follows DestinationActivation's destination and alias
// contract without allocating.
func (s *SegmentedSoftmax) BackwardInto(
	input,
	outputGradient,
	inputGradient *matrix.Matrix,
) (err error) {
	if _, _, err = s.validatePair(input, outputGradient); err != nil {
		return err
	}
	if err = input.SegmentedSoftmaxRowsBackwardInto(s.widths, outputGradient, inputGradient); err != nil {
		err = fmt.Errorf("activation: segmented softmax backward failed: %w", err)
		return err
	}

	return nil
}

// Widths returns a copy of the ordered segment widths.
func (s *SegmentedSoftmax) Widths() (widths []int) {
	if s == nil {
		return nil
	}

	widths = append([]int(nil), s.widths...)
	return widths
}

func (s *SegmentedSoftmax) validate() (err error) {
	if s == nil {
		err = errors.New("activation: segmented softmax is nil")
		return err
	}
	if len(s.widths) == 0 || s.total <= 0 {
		err = errors.New("activation: segmented softmax is not configured")
		return err
	}

	return nil
}

func (s *SegmentedSoftmax) validateInput(name string, input *matrix.Matrix) (rows int, err error) {
	var cols int

	if err = s.validate(); err != nil {
		return 0, err
	}
	if rows, cols, err = matrixShape(name, input); err != nil {
		return 0, err
	}
	if cols != s.total {
		err = fmt.Errorf(
			"activation: segmented softmax %s column mismatch: got %d, want %d",
			name,
			cols,
			s.total,
		)
		return 0, err
	}

	return rows, nil
}

func (s *SegmentedSoftmax) validatePair(
	input,
	outputGradient *matrix.Matrix,
) (rows, cols int, err error) {
	if rows, err = s.validateInput("input", input); err != nil {
		return 0, 0, err
	}
	if rows, cols, err = matrixPairShape(input, outputGradient); err != nil {
		return 0, 0, err
	}

	return rows, cols, nil
}
