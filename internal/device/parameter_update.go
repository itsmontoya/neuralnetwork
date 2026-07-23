package device

// ParameterUpdate identifies one opaque value and gradient pair for a private update.
type ParameterUpdate struct {
	Values   any
	Gradient any
}
