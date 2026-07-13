package optimizer

// NewL2WeightDecay constructs L2 weight decay with the provided coefficient.
func NewL2WeightDecay(coefficient float32) (out *L2WeightDecay, err error) {
	if err = validateRegularizationCoefficient("l2 weight decay", coefficient); err != nil {
		return nil, err
	}

	var l L2WeightDecay
	l.coefficient = coefficient
	return &l, nil
}

// L2WeightDecay adds coefficient*value to each parameter gradient.
type L2WeightDecay struct {
	coefficient float32
}

// Apply adds L2 weight decay gradients to parameters.
func (l *L2WeightDecay) Apply(parameters []*Parameter) (err error) {
	if err = l.validate(); err != nil {
		return err
	}

	err = applyRegularizationGradient(parameters, l.gradient)
	return err
}

// Coefficient returns the L2 weight decay coefficient.
func (l *L2WeightDecay) Coefficient() (coefficient float32) {
	if l == nil {
		return 0
	}

	coefficient = l.coefficient
	return coefficient
}

// SetCoefficient updates the L2 weight decay coefficient.
func (l *L2WeightDecay) SetCoefficient(coefficient float32) (err error) {
	if l == nil {
		err = nilRegularizerError("l2 weight decay")
		return err
	}

	if err = validateRegularizationCoefficient("l2 weight decay", coefficient); err != nil {
		return err
	}

	l.coefficient = coefficient
	return nil
}

func (l *L2WeightDecay) gradient(value float32) (gradient float32) {
	gradient = l.coefficient * value
	return gradient
}

func (l *L2WeightDecay) validate() (err error) {
	if l == nil {
		err = nilRegularizerError("l2 weight decay")
		return err
	}

	err = validateRegularizationCoefficient("l2 weight decay", l.coefficient)
	return err
}
