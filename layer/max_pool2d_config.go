package layer

import "fmt"

// NewMaxPool2DConfig constructs validated two-dimensional max-pooling geometry.
func NewMaxPool2DConfig(
	inputShape SpatialShape,
	windowHeight, windowWidth int,
	strideHeight, strideWidth int,
) (config MaxPool2DConfig, err error) {
	var (
		outputShape  SpatialShape
		outputHeight int
		outputWidth  int
	)

	if err = inputShape.validate(); err != nil {
		err = fmt.Errorf("layer: max pool2d input shape invalid: %w", err)
		return config, err
	}

	if windowHeight <= 0 {
		err = fmt.Errorf("layer: max pool2d window height must be positive: windowHeight=%d", windowHeight)
		return config, err
	}

	if windowWidth <= 0 {
		err = fmt.Errorf("layer: max pool2d window width must be positive: windowWidth=%d", windowWidth)
		return config, err
	}

	if strideHeight <= 0 {
		err = fmt.Errorf("layer: max pool2d stride height must be positive: strideHeight=%d", strideHeight)
		return config, err
	}

	if strideWidth <= 0 {
		err = fmt.Errorf("layer: max pool2d stride width must be positive: strideWidth=%d", strideWidth)
		return config, err
	}

	if outputHeight, err = calculateSpatialOutputDimension(
		"max pool2d window height",
		inputShape.Height(),
		windowHeight,
		strideHeight,
		0,
	); err != nil {
		return config, err
	}

	if outputWidth, err = calculateSpatialOutputDimension(
		"max pool2d window width",
		inputShape.Width(),
		windowWidth,
		strideWidth,
		0,
	); err != nil {
		return config, err
	}

	if outputShape, err = NewSpatialShape(inputShape.Channels(), outputHeight, outputWidth); err != nil {
		err = fmt.Errorf("layer: max pool2d output shape invalid: %w", err)
		return config, err
	}

	config.inputShape = inputShape
	config.outputShape = outputShape
	config.windowHeight = windowHeight
	config.windowWidth = windowWidth
	config.strideHeight = strideHeight
	config.strideWidth = strideWidth
	return config, nil
}

// MaxPool2DConfig describes validated two-dimensional max-pooling geometry.
type MaxPool2DConfig struct {
	inputShape   SpatialShape
	outputShape  SpatialShape
	windowHeight int
	windowWidth  int
	strideHeight int
	strideWidth  int
}

// InputShape returns the configured input shape.
func (c MaxPool2DConfig) InputShape() (shape SpatialShape) {
	shape = c.inputShape
	return shape
}

// OutputShape returns the derived output shape.
func (c MaxPool2DConfig) OutputShape() (shape SpatialShape) {
	shape = c.outputShape
	return shape
}

// WindowHeight returns the pooling-window height.
func (c MaxPool2DConfig) WindowHeight() (height int) {
	height = c.windowHeight
	return height
}

// WindowWidth returns the pooling-window width.
func (c MaxPool2DConfig) WindowWidth() (width int) {
	width = c.windowWidth
	return width
}

// StrideHeight returns the vertical stride.
func (c MaxPool2DConfig) StrideHeight() (height int) {
	height = c.strideHeight
	return height
}

// StrideWidth returns the horizontal stride.
func (c MaxPool2DConfig) StrideWidth() (width int) {
	width = c.strideWidth
	return width
}

func (c MaxPool2DConfig) validate() (err error) {
	var expected MaxPool2DConfig

	if expected, err = NewMaxPool2DConfig(
		c.inputShape,
		c.windowHeight,
		c.windowWidth,
		c.strideHeight,
		c.strideWidth,
	); err != nil {
		return err
	}

	if c.outputShape != expected.outputShape {
		err = fmt.Errorf(
			"layer: max pool2d output shape mismatch: got=%dx%dx%d want=%dx%dx%d",
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
