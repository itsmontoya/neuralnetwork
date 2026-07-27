package optimizer

// GradientClippingObservation describes the most recent completed clipping phase.
type GradientClippingObservation struct {
	ValueClipped        bool
	GlobalNorm          float64
	Scale               float64
	BaseUpdateCompleted bool
}
