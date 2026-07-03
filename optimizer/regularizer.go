package optimizer

// Regularizer adds regularization terms to parameter gradients before updates.
type Regularizer interface {
	// Apply adds regularization gradients to parameters.
	Apply(parameters []*Parameter) (err error)
}
