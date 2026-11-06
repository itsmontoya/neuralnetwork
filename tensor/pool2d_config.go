package tensor

// Pool2DConfig specifies explicit NCHW pooling geometry without schema defaults.
// Pads are top, left, bottom, right. Kernel, strides, and dilations must be
// positive; pads must be nonnegative. Its zero value is invalid. Average pooling
// requires unit dilation; CountIncludePad has no effect on max pooling.
type Pool2DConfig struct {
	Kernel          [2]int64
	Strides         [2]int64
	Dilations       [2]int64
	Pads            [4]int64
	CeilMode        bool
	CountIncludePad bool
}
