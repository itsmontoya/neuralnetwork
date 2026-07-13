package optimizer

import "github.com/itsmontoya/neuralnetwork/internal/f32"

// NewExponentialDecay constructs a schedule that decays every epoch by decayRate.
func NewExponentialDecay(initialLearningRate, decayRate float32) (out *ExponentialDecay, err error) {
	if err = validateLearningRate(initialLearningRate); err != nil {
		return nil, err
	}

	if err = validateDecayRate("exponential decay rate", decayRate); err != nil {
		return nil, err
	}

	var e ExponentialDecay
	e.initialLearningRate = initialLearningRate
	e.decayRate = decayRate
	return &e, nil
}

// ExponentialDecay returns initialLearningRate*decayRate^(epoch-1).
type ExponentialDecay struct {
	initialLearningRate float32
	decayRate           float32
}

// LearningRate returns the exponentially decayed learning rate for a one-based epoch.
func (e *ExponentialDecay) LearningRate(epoch int) (learningRate float32, err error) {
	if err = e.validate(); err != nil {
		return 0, err
	}

	if err = validateScheduleEpoch(epoch); err != nil {
		return 0, err
	}

	learningRate = e.initialLearningRate * f32.Pow(e.decayRate, float32(epoch-1))
	if err = validateLearningRate(learningRate); err != nil {
		return 0, err
	}

	return learningRate, nil
}

// InitialLearningRate returns the epoch-one learning rate.
func (e *ExponentialDecay) InitialLearningRate() (learningRate float32) {
	if e == nil {
		return 0
	}

	learningRate = e.initialLearningRate
	return learningRate
}

// DecayRate returns the per-epoch multiplicative decay rate.
func (e *ExponentialDecay) DecayRate() (decayRate float32) {
	if e == nil {
		return 0
	}

	decayRate = e.decayRate
	return decayRate
}

func (e *ExponentialDecay) validate() (err error) {
	if e == nil {
		err = nilScheduleError("exponential decay")
		return err
	}

	if err = validateLearningRate(e.initialLearningRate); err != nil {
		return err
	}

	err = validateDecayRate("exponential decay rate", e.decayRate)
	return err
}
