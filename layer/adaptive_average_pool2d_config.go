package layer

import "fmt"

// NewAdaptiveAveragePool2DConfig constructs validated adaptive average-pooling
// geometry with an explicit output height and width.
func NewAdaptiveAveragePool2DConfig(
	inputShape SpatialShape,
	outputHeight, outputWidth int,
) (config AdaptiveAveragePool2DConfig, err error) {
	var outputShape SpatialShape

	if err = inputShape.validate(); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d input shape invalid: %w", err)
		return config, err
	}
	if outputHeight <= 0 {
		err = fmt.Errorf("layer: adaptive average pool2d output height must be positive: outputHeight=%d", outputHeight)
		return config, err
	}
	if outputWidth <= 0 {
		err = fmt.Errorf("layer: adaptive average pool2d output width must be positive: outputWidth=%d", outputWidth)
		return config, err
	}
	if outputShape, err = NewSpatialShape(inputShape.Channels(), outputHeight, outputWidth); err != nil {
		err = fmt.Errorf("layer: adaptive average pool2d output shape invalid: %w", err)
		return config, err
	}

	config.inputShape = inputShape
	config.outputShape = outputShape
	return config, nil
}

// AdaptiveAveragePool2DConfig describes validated adaptive average-pooling
// geometry.
type AdaptiveAveragePool2DConfig struct {
	inputShape  SpatialShape
	outputShape SpatialShape
}

// InputShape returns the configured input shape.
func (c AdaptiveAveragePool2DConfig) InputShape() (shape SpatialShape) {
	shape = c.inputShape
	return shape
}

// OutputShape returns the configured output shape.
func (c AdaptiveAveragePool2DConfig) OutputShape() (shape SpatialShape) {
	shape = c.outputShape
	return shape
}

// OutputHeight returns the configured output height.
func (c AdaptiveAveragePool2DConfig) OutputHeight() (height int) {
	height = c.outputShape.Height()
	return height
}

// OutputWidth returns the configured output width.
func (c AdaptiveAveragePool2DConfig) OutputWidth() (width int) {
	width = c.outputShape.Width()
	return width
}

func (c AdaptiveAveragePool2DConfig) validate() (err error) {
	var expected AdaptiveAveragePool2DConfig

	if expected, err = NewAdaptiveAveragePool2DConfig(
		c.inputShape,
		c.outputShape.Height(),
		c.outputShape.Width(),
	); err != nil {
		return err
	}
	if c.outputShape != expected.outputShape {
		err = fmt.Errorf(
			"layer: adaptive average pool2d output shape mismatch: got=%dx%dx%d want=%dx%dx%d",
			c.outputShape.Channels(),
			c.outputShape.Height(),
			c.outputShape.Width(),
			expected.outputShape.Channels(),
			expected.outputShape.Height(),
			expected.outputShape.Width(),
		)
		return err
	}
	return nil
}
