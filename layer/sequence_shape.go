package layer

import "fmt"

// NewSequenceShape constructs a validated time-major sequence shape.
func NewSequenceShape(steps, featureSize int) (shape SequenceShape, err error) {
	var maxInt int

	if steps <= 0 {
		err = fmt.Errorf("layer: sequence shape steps must be positive: got=%d want>0", steps)
		return shape, err
	}

	if featureSize <= 0 {
		err = fmt.Errorf("layer: sequence shape feature size must be positive: got=%d want>0", featureSize)
		return shape, err
	}

	maxInt = int(^uint(0) >> 1)
	if steps > maxInt/featureSize {
		err = fmt.Errorf(
			"layer: sequence shape size overflows int: steps=%d featureSize=%d want<=%d",
			steps,
			featureSize,
			maxInt,
		)
		return shape, err
	}

	shape.steps = steps
	shape.featureSize = featureSize
	shape.size = steps * featureSize
	return shape, nil
}

// SequenceShape describes one time-major sequence without a batch dimension.
type SequenceShape struct {
	steps       int
	featureSize int
	size        int
}

// Steps returns the number of sequence steps.
func (s SequenceShape) Steps() (steps int) {
	steps = s.steps
	return steps
}

// FeatureSize returns the number of features at each step.
func (s SequenceShape) FeatureSize() (featureSize int) {
	featureSize = s.featureSize
	return featureSize
}

// Size returns the flattened time-major value count.
func (s SequenceShape) Size() (size int) {
	size = s.size
	return size
}

func (s SequenceShape) validate() (err error) {
	var expected SequenceShape

	if expected, err = NewSequenceShape(s.steps, s.featureSize); err != nil {
		return err
	}

	if s.size != expected.size {
		err = fmt.Errorf("layer: sequence shape size mismatch: got=%d want=%d", s.size, expected.size)
		return err
	}

	return nil
}
