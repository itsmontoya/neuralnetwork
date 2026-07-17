package layer

import "fmt"

// NewSpatialShape constructs a validated channels-first spatial shape.
func NewSpatialShape(channels, height, width int) (shape SpatialShape, err error) {
	var size int

	if channels <= 0 {
		err = fmt.Errorf("layer: spatial shape channels must be positive: channels=%d", channels)
		return shape, err
	}

	if height <= 0 {
		err = fmt.Errorf("layer: spatial shape height must be positive: height=%d", height)
		return shape, err
	}

	if width <= 0 {
		err = fmt.Errorf("layer: spatial shape width must be positive: width=%d", width)
		return shape, err
	}

	if size, err = checkedProduct3("spatial shape size", channels, height, width); err != nil {
		return shape, err
	}

	shape.channels = channels
	shape.height = height
	shape.width = width
	shape.size = size
	return shape, nil
}

// SpatialShape describes one channels-first spatial value without a batch dimension.
type SpatialShape struct {
	channels int
	height   int
	width    int
	size     int
}

// Channels returns the number of channels.
func (s SpatialShape) Channels() (channels int) {
	channels = s.channels
	return channels
}

// Height returns the spatial height.
func (s SpatialShape) Height() (height int) {
	height = s.height
	return height
}

// Width returns the spatial width.
func (s SpatialShape) Width() (width int) {
	width = s.width
	return width
}

// Size returns the flattened channels-first value count.
func (s SpatialShape) Size() (size int) {
	size = s.size
	return size
}

func (s SpatialShape) validate() (err error) {
	var expected SpatialShape

	if expected, err = NewSpatialShape(s.channels, s.height, s.width); err != nil {
		return err
	}

	if s.size != expected.size {
		err = fmt.Errorf("layer: spatial shape size mismatch: got=%d want=%d", s.size, expected.size)
		return err
	}

	return nil
}
