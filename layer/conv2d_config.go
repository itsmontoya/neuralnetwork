package layer

import "fmt"

// NewConv2DConfig constructs validated two-dimensional convolution geometry.
func NewConv2DConfig(
	inputShape SpatialShape,
	outputChannels, kernelHeight, kernelWidth int,
	strideHeight, strideWidth int,
	paddingHeight, paddingWidth int,
) (config Conv2DConfig, err error) {
	var (
		outputShape  SpatialShape
		outputHeight int
		outputWidth  int
	)

	if err = inputShape.validate(); err != nil {
		err = fmt.Errorf("layer: conv2d input shape invalid: %w", err)
		return config, err
	}

	if outputChannels <= 0 {
		err = fmt.Errorf("layer: conv2d output channels must be positive: outputChannels=%d", outputChannels)
		return config, err
	}

	if kernelHeight <= 0 {
		err = fmt.Errorf("layer: conv2d kernel height must be positive: kernelHeight=%d", kernelHeight)
		return config, err
	}

	if kernelWidth <= 0 {
		err = fmt.Errorf("layer: conv2d kernel width must be positive: kernelWidth=%d", kernelWidth)
		return config, err
	}

	if strideHeight <= 0 {
		err = fmt.Errorf("layer: conv2d stride height must be positive: strideHeight=%d", strideHeight)
		return config, err
	}

	if strideWidth <= 0 {
		err = fmt.Errorf("layer: conv2d stride width must be positive: strideWidth=%d", strideWidth)
		return config, err
	}

	if paddingHeight < 0 {
		err = fmt.Errorf("layer: conv2d padding height must be non-negative: paddingHeight=%d", paddingHeight)
		return config, err
	}

	if paddingWidth < 0 {
		err = fmt.Errorf("layer: conv2d padding width must be non-negative: paddingWidth=%d", paddingWidth)
		return config, err
	}

	if _, err = checkedProduct3(
		"conv2d kernel size",
		inputShape.Channels(),
		kernelHeight,
		kernelWidth,
	); err != nil {
		return config, err
	}

	if outputHeight, err = calculateSpatialOutputDimension(
		"conv2d kernel height",
		inputShape.Height(),
		kernelHeight,
		strideHeight,
		paddingHeight,
	); err != nil {
		return config, err
	}

	if outputWidth, err = calculateSpatialOutputDimension(
		"conv2d kernel width",
		inputShape.Width(),
		kernelWidth,
		strideWidth,
		paddingWidth,
	); err != nil {
		return config, err
	}

	if outputShape, err = NewSpatialShape(outputChannels, outputHeight, outputWidth); err != nil {
		err = fmt.Errorf("layer: conv2d output shape invalid: %w", err)
		return config, err
	}

	config.inputShape = inputShape
	config.outputShape = outputShape
	config.outputChannels = outputChannels
	config.kernelHeight = kernelHeight
	config.kernelWidth = kernelWidth
	config.strideHeight = strideHeight
	config.strideWidth = strideWidth
	config.paddingHeight = paddingHeight
	config.paddingWidth = paddingWidth
	return config, nil
}

// Conv2DConfig describes validated two-dimensional convolution geometry.
type Conv2DConfig struct {
	inputShape     SpatialShape
	outputShape    SpatialShape
	outputChannels int
	kernelHeight   int
	kernelWidth    int
	strideHeight   int
	strideWidth    int
	paddingHeight  int
	paddingWidth   int
}

// InputShape returns the configured input shape.
func (c Conv2DConfig) InputShape() (shape SpatialShape) {
	shape = c.inputShape
	return shape
}

// OutputShape returns the derived output shape.
func (c Conv2DConfig) OutputShape() (shape SpatialShape) {
	shape = c.outputShape
	return shape
}

// OutputChannels returns the number of convolution filters.
func (c Conv2DConfig) OutputChannels() (channels int) {
	channels = c.outputChannels
	return channels
}

// KernelHeight returns the kernel height.
func (c Conv2DConfig) KernelHeight() (height int) {
	height = c.kernelHeight
	return height
}

// KernelWidth returns the kernel width.
func (c Conv2DConfig) KernelWidth() (width int) {
	width = c.kernelWidth
	return width
}

// StrideHeight returns the vertical stride.
func (c Conv2DConfig) StrideHeight() (height int) {
	height = c.strideHeight
	return height
}

// StrideWidth returns the horizontal stride.
func (c Conv2DConfig) StrideWidth() (width int) {
	width = c.strideWidth
	return width
}

// PaddingHeight returns the symmetric vertical padding.
func (c Conv2DConfig) PaddingHeight() (height int) {
	height = c.paddingHeight
	return height
}

// PaddingWidth returns the symmetric horizontal padding.
func (c Conv2DConfig) PaddingWidth() (width int) {
	width = c.paddingWidth
	return width
}

func (c Conv2DConfig) validate() (err error) {
	var expected Conv2DConfig

	if expected, err = NewConv2DConfig(
		c.inputShape,
		c.outputChannels,
		c.kernelHeight,
		c.kernelWidth,
		c.strideHeight,
		c.strideWidth,
		c.paddingHeight,
		c.paddingWidth,
	); err != nil {
		return err
	}

	if c.outputShape != expected.outputShape {
		err = fmt.Errorf(
			"layer: conv2d output shape mismatch: got=%dx%dx%d want=%dx%dx%d",
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
