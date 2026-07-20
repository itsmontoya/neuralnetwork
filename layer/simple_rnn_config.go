package layer

import "fmt"

// NewSimpleRNNConfig constructs validated simple recurrent layer geometry.
func NewSimpleRNNConfig(
	inputShape SequenceShape,
	hiddenSize int,
) (config SimpleRNNConfig, err error) {
	var outputShape SequenceShape

	if err = inputShape.validate(); err != nil {
		err = fmt.Errorf(
			"layer: simple rnn input shape invalid: steps=%d featureSize=%d: %w",
			inputShape.Steps(),
			inputShape.FeatureSize(),
			err,
		)
		return config, err
	}

	if hiddenSize <= 0 {
		err = fmt.Errorf("layer: simple rnn hidden size must be positive: got=%d want>0", hiddenSize)
		return config, err
	}

	if outputShape, err = NewSequenceShape(inputShape.Steps(), hiddenSize); err != nil {
		err = fmt.Errorf(
			"layer: simple rnn output shape invalid: steps=%d hiddenSize=%d: %w",
			inputShape.Steps(),
			hiddenSize,
			err,
		)
		return config, err
	}

	config.inputShape = inputShape
	config.outputShape = outputShape
	config.hiddenSize = hiddenSize
	return config, nil
}

// SimpleRNNConfig describes validated simple recurrent layer geometry.
type SimpleRNNConfig struct {
	inputShape  SequenceShape
	outputShape SequenceShape
	hiddenSize  int
}

// InputShape returns the configured input sequence shape.
func (c SimpleRNNConfig) InputShape() (shape SequenceShape) {
	shape = c.inputShape
	return shape
}

// OutputShape returns the derived output sequence shape.
func (c SimpleRNNConfig) OutputShape() (shape SequenceShape) {
	shape = c.outputShape
	return shape
}

// HiddenSize returns the number of recurrent hidden values at each step.
func (c SimpleRNNConfig) HiddenSize() (hiddenSize int) {
	hiddenSize = c.hiddenSize
	return hiddenSize
}

func (c SimpleRNNConfig) validate() (err error) {
	var expected SimpleRNNConfig

	if expected, err = NewSimpleRNNConfig(c.inputShape, c.hiddenSize); err != nil {
		return err
	}

	if c.outputShape != expected.outputShape {
		err = fmt.Errorf(
			"layer: simple rnn output shape mismatch: got=%dx%d want=%dx%d",
			c.outputShape.Steps(),
			c.outputShape.FeatureSize(),
			expected.outputShape.Steps(),
			expected.outputShape.FeatureSize(),
		)
		return err
	}

	return nil
}
