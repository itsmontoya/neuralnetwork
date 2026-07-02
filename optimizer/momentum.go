package optimizer

import "github.com/itsmontoya/neuralnetwork/matrix"

const defaultMomentumCoefficient = 0.9

// NewMomentum constructs Momentum with the default coefficient.
func NewMomentum(learningRate float64) (m *Momentum, err error) {
	m, err = NewMomentumWithCoefficient(learningRate, defaultMomentumCoefficient)
	return m, err
}

// NewMomentumWithCoefficient constructs Momentum with an explicit coefficient.
func NewMomentumWithCoefficient(learningRate, coefficient float64) (out *Momentum, err error) {
	if err = validateLearningRate(learningRate); err != nil {
		return nil, err
	}

	if err = validateUnitCoefficient("momentum", coefficient); err != nil {
		return nil, err
	}

	var m Momentum
	m.learningRate = learningRate
	m.coefficient = coefficient
	m.velocities = make(map[*Parameter]*matrix.Matrix)
	return &m, nil
}

// Momentum applies SGD updates with per-parameter velocity state.
//
// Update applies velocity = coefficient*velocity - learningRate*gradient and
// values += velocity for each parameter. Gradients are reset after a successful
// update.
type Momentum struct {
	learningRate float64
	coefficient  float64
	velocities   map[*Parameter]*matrix.Matrix
}

// Update applies one Momentum update to each parameter.
func (m *Momentum) Update(parameters []*Parameter) (err error) {
	var parameter *Parameter

	if err = m.validate(); err != nil {
		return err
	}

	if err = validateParameters(parameters); err != nil {
		return err
	}

	for _, parameter = range parameters {
		if err = m.updateParameter(parameter); err != nil {
			return err
		}
	}

	err = resetGradients(parameters)
	return err
}

// LearningRate returns the current learning rate.
func (m *Momentum) LearningRate() (learningRate float64) {
	if m == nil {
		return 0
	}

	learningRate = m.learningRate
	return learningRate
}

// SetLearningRate updates the learning rate.
func (m *Momentum) SetLearningRate(learningRate float64) (err error) {
	if m == nil {
		err = nilOptimizerError("momentum")
		return err
	}

	if err = validateLearningRate(learningRate); err != nil {
		return err
	}

	m.learningRate = learningRate
	return nil
}

// Coefficient returns the current momentum coefficient.
func (m *Momentum) Coefficient() (coefficient float64) {
	if m == nil {
		return 0
	}

	coefficient = m.coefficient
	return coefficient
}

// SetCoefficient updates the momentum coefficient.
func (m *Momentum) SetCoefficient(coefficient float64) (err error) {
	if m == nil {
		err = nilOptimizerError("momentum")
		return err
	}

	if err = validateUnitCoefficient("momentum", coefficient); err != nil {
		return err
	}

	m.coefficient = coefficient
	return nil
}

func (m *Momentum) updateParameter(parameter *Parameter) (err error) {
	var velocity *matrix.Matrix

	if velocity, err = m.velocityFor(parameter); err != nil {
		return err
	}

	if err = velocity.MultiplyScalarInPlace(m.coefficient); err != nil {
		return err
	}

	if err = velocity.AddScaledInPlace(parameter.Gradient(), -m.learningRate); err != nil {
		return err
	}

	err = parameter.Values().AddInPlace(velocity)
	return err
}

func (m *Momentum) velocityFor(parameter *Parameter) (velocity *matrix.Matrix, err error) {
	var (
		rows int
		cols int
	)

	if velocity = m.velocities[parameter]; velocity != nil {
		return velocity, nil
	}

	rows, cols = parameter.Values().Shape()
	if velocity, err = matrix.New(rows, cols); err != nil {
		return nil, err
	}

	m.velocities[parameter] = velocity
	return velocity, nil
}

func (m *Momentum) validate() (err error) {
	if m == nil {
		err = nilOptimizerError("momentum")
		return err
	}

	if err = validateLearningRate(m.learningRate); err != nil {
		return err
	}

	if err = validateUnitCoefficient("momentum", m.coefficient); err != nil {
		return err
	}

	if m.velocities == nil {
		m.velocities = make(map[*Parameter]*matrix.Matrix)
	}

	return nil
}
