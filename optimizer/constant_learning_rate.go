package optimizer

// NewConstantLearningRate constructs a schedule that always returns learningRate.
func NewConstantLearningRate(learningRate float32) (out *ConstantLearningRate, err error) {
	if err = validateLearningRate(learningRate); err != nil {
		return nil, err
	}

	var c ConstantLearningRate
	c.learningRate = learningRate
	return &c, nil
}

// ConstantLearningRate returns the same learning rate for every epoch.
type ConstantLearningRate struct {
	learningRate float32
}

// LearningRate returns the configured learning rate for a one-based epoch.
func (c *ConstantLearningRate) LearningRate(epoch int) (learningRate float32, err error) {
	if err = c.validate(); err != nil {
		return 0, err
	}

	if err = validateScheduleEpoch(epoch); err != nil {
		return 0, err
	}

	learningRate = c.learningRate
	return learningRate, nil
}

// Rate returns the configured learning rate.
func (c *ConstantLearningRate) Rate() (learningRate float32) {
	if c == nil {
		return 0
	}

	learningRate = c.learningRate
	return learningRate
}

func (c *ConstantLearningRate) validate() (err error) {
	if c == nil {
		err = nilScheduleError("constant learning rate")
		return err
	}

	err = validateLearningRate(c.learningRate)
	return err
}
