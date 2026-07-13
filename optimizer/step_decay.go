package optimizer

import "github.com/itsmontoya/neuralnetwork/internal/f32"

// NewStepDecay constructs a schedule that decays by factor every stepSize epochs.
func NewStepDecay(initialLearningRate, factor float32, stepSize int) (out *StepDecay, err error) {
	if err = validateLearningRate(initialLearningRate); err != nil {
		return nil, err
	}

	if err = validateDecayRate("step decay factor", factor); err != nil {
		return nil, err
	}

	if stepSize <= 0 {
		err = invalidStepSizeError(stepSize)
		return nil, err
	}

	var s StepDecay
	s.initialLearningRate = initialLearningRate
	s.factor = factor
	s.stepSize = stepSize
	return &s, nil
}

// StepDecay returns initialLearningRate*factor^k, where k advances every stepSize epochs.
type StepDecay struct {
	initialLearningRate float32
	factor              float32
	stepSize            int
}

// LearningRate returns the decayed learning rate for a one-based epoch.
func (s *StepDecay) LearningRate(epoch int) (learningRate float32, err error) {
	var exponent int

	if err = s.validate(); err != nil {
		return 0, err
	}

	if err = validateScheduleEpoch(epoch); err != nil {
		return 0, err
	}

	exponent = (epoch - 1) / s.stepSize
	learningRate = s.initialLearningRate * f32.Pow(s.factor, float32(exponent))
	if err = validateLearningRate(learningRate); err != nil {
		return 0, err
	}

	return learningRate, nil
}

// InitialLearningRate returns the epoch-one learning rate.
func (s *StepDecay) InitialLearningRate() (learningRate float32) {
	if s == nil {
		return 0
	}

	learningRate = s.initialLearningRate
	return learningRate
}

// Factor returns the multiplicative decay factor.
func (s *StepDecay) Factor() (factor float32) {
	if s == nil {
		return 0
	}

	factor = s.factor
	return factor
}

// StepSize returns the number of epochs between decays.
func (s *StepDecay) StepSize() (stepSize int) {
	if s == nil {
		return 0
	}

	stepSize = s.stepSize
	return stepSize
}

func (s *StepDecay) validate() (err error) {
	if s == nil {
		err = nilScheduleError("step decay")
		return err
	}

	if err = validateLearningRate(s.initialLearningRate); err != nil {
		return err
	}

	if err = validateDecayRate("step decay factor", s.factor); err != nil {
		return err
	}

	if s.stepSize <= 0 {
		err = invalidStepSizeError(s.stepSize)
		return err
	}

	return nil
}
