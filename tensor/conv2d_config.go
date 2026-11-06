package tensor

// Conv2DConfig specifies explicit NCHW cross-correlation geometry.
// Pads are top, left, bottom, right. Strides, dilations, and groups must be
// positive; pads must be nonnegative. Its zero value is invalid.
type Conv2DConfig struct {
	Strides   [2]int64
	Dilations [2]int64
	Pads      [4]int64
	Groups    int64
}
