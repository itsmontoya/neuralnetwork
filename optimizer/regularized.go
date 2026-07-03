package optimizer

import "errors"

// NewRegularized wraps an optimizer with one or more regularizers.
func NewRegularized(base Optimizer, regularizers ...Regularizer) (out *Regularized, err error) {
	if base == nil {
		err = errors.New("optimizer: base optimizer is nil")
		return nil, err
	}

	if err = validateRegularizers(regularizers); err != nil {
		return nil, err
	}

	var r Regularized
	r.base = base
	r.regularizers = append([]Regularizer(nil), regularizers...)
	return &r, nil
}

// Regularized applies regularizers before delegating to a base optimizer.
//
// The wrapped optimizer remains responsible for applying parameter updates and
// resetting gradients after a successful update.
type Regularized struct {
	base         Optimizer
	regularizers []Regularizer
}

// Update applies regularization gradients and then updates parameters.
func (r *Regularized) Update(parameters []*Parameter) (err error) {
	var regularizer Regularizer

	if err = r.validate(); err != nil {
		return err
	}

	if err = validateParameters(parameters); err != nil {
		return err
	}

	for _, regularizer = range r.regularizers {
		if err = regularizer.Apply(parameters); err != nil {
			return err
		}
	}

	err = r.base.Update(parameters)
	return err
}

// LearningRate returns the wrapped optimizer learning rate.
func (r *Regularized) LearningRate() (learningRate float64) {
	if r == nil || r.base == nil {
		return 0
	}

	learningRate = r.base.LearningRate()
	return learningRate
}

// SetLearningRate updates the wrapped optimizer learning rate.
func (r *Regularized) SetLearningRate(learningRate float64) (err error) {
	if r == nil {
		err = nilOptimizerError("regularized")
		return err
	}

	if r.base == nil {
		err = errors.New("optimizer: base optimizer is nil")
		return err
	}

	err = r.base.SetLearningRate(learningRate)
	return err
}

// Base returns the wrapped optimizer.
func (r *Regularized) Base() (base Optimizer) {
	if r == nil {
		return nil
	}

	base = r.base
	return base
}

// Regularizers returns the regularizers applied before each update.
func (r *Regularized) Regularizers() (regularizers []Regularizer) {
	if r == nil {
		return nil
	}

	regularizers = append([]Regularizer(nil), r.regularizers...)
	return regularizers
}

func (r *Regularized) validate() (err error) {
	if r == nil {
		err = nilOptimizerError("regularized")
		return err
	}

	if r.base == nil {
		err = errors.New("optimizer: base optimizer is nil")
		return err
	}

	err = validateRegularizers(r.regularizers)
	return err
}
