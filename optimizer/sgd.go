package optimizer

import "github.com/itsmontoya/neuralnetwork/internal/device"

// NewSGD constructs stochastic gradient descent with the provided learning rate.
func NewSGD(learningRate float32) (out *SGD, err error) {
	if err = validateLearningRate(learningRate); err != nil {
		return nil, err
	}

	var s SGD
	s.learningRate = learningRate
	return &s, nil
}

// SGD applies plain stochastic gradient descent updates.
//
// Update applies values -= learningRate * gradient for each parameter and
// resets gradients after a successful update.
type SGD struct {
	learningRate float32
	updateBuffer []device.ParameterUpdate
}

// Update applies one SGD update to each parameter.
func (s *SGD) Update(parameters []*Parameter) (err error) {
	var (
		parameter *Parameter
		handled   bool
	)

	if err = s.validate(); err != nil {
		return err
	}

	if err = validateParameters(parameters); err != nil {
		return err
	}
	clear(s.updateBuffer)
	s.updateBuffer = s.updateBuffer[:0]
	for _, parameter = range parameters {
		s.updateBuffer = append(s.updateBuffer, device.ParameterUpdate{
			Values:   parameter.Values(),
			Gradient: parameter.Gradient(),
		})
	}
	if handled, err = device.SGD(s.updateBuffer, s.learningRate); err != nil {
		return err
	}
	if handled {
		return nil
	}

	for _, parameter = range parameters {
		if err = parameter.Values().AddScaledInPlace(parameter.Gradient(), -s.learningRate); err != nil {
			return err
		}
	}

	err = resetGradients(parameters)
	return err
}

// LearningRate returns the current learning rate.
func (s *SGD) LearningRate() (learningRate float32) {
	if s == nil {
		return 0
	}

	learningRate = s.learningRate
	return learningRate
}

// SetLearningRate updates the learning rate.
func (s *SGD) SetLearningRate(learningRate float32) (err error) {
	if s == nil {
		err = nilOptimizerError("sgd")
		return err
	}

	if err = validateLearningRate(learningRate); err != nil {
		return err
	}

	s.learningRate = learningRate
	return nil
}

func (s *SGD) validate() (err error) {
	if s == nil {
		err = nilOptimizerError("sgd")
		return err
	}

	err = validateLearningRate(s.learningRate)
	return err
}
