package optimizer

// NewL1 constructs L1 regularization with the provided coefficient.
func NewL1(coefficient float64) (out *L1, err error) {
	if err = validateRegularizationCoefficient("l1", coefficient); err != nil {
		return nil, err
	}

	var l L1
	l.coefficient = coefficient
	return &l, nil
}

// L1 adds coefficient*sign(value) to each parameter gradient.
type L1 struct {
	coefficient float64
}

// Apply adds L1 regularization gradients to parameters.
func (l *L1) Apply(parameters []*Parameter) (err error) {
	if err = l.validate(); err != nil {
		return err
	}

	err = applyRegularizationGradient(parameters, l.gradient)
	return err
}

// Coefficient returns the L1 regularization coefficient.
func (l *L1) Coefficient() (coefficient float64) {
	if l == nil {
		return 0
	}

	coefficient = l.coefficient
	return coefficient
}

// SetCoefficient updates the L1 regularization coefficient.
func (l *L1) SetCoefficient(coefficient float64) (err error) {
	if l == nil {
		err = nilRegularizerError("l1")
		return err
	}

	if err = validateRegularizationCoefficient("l1", coefficient); err != nil {
		return err
	}

	l.coefficient = coefficient
	return nil
}

func (l *L1) gradient(value float64) (gradient float64) {
	if value > 0 {
		gradient = l.coefficient
		return gradient
	}

	if value < 0 {
		gradient = -l.coefficient
		return gradient
	}

	return 0
}

func (l *L1) validate() (err error) {
	if l == nil {
		err = nilRegularizerError("l1")
		return err
	}

	err = validateRegularizationCoefficient("l1", l.coefficient)
	return err
}
