package optimizer

// GradientClippingConfig configures opt-in value and global-norm clipping.
//
// A zero field disables that control. At least one control must be enabled.
type GradientClippingConfig struct {
	MaxValue float32
	MaxNorm  float32
}
