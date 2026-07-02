package optimizer

import (
	"fmt"
	"math"
)

// LearningRateSchedule computes the optimizer learning rate for a Fit epoch.
type LearningRateSchedule interface {
	// LearningRate returns the learning rate for a one-based epoch.
	LearningRate(epoch int) (learningRate float64, err error)
}

func validateScheduleEpoch(epoch int) (err error) {
	if epoch <= 0 {
		err = fmt.Errorf("optimizer: schedule epoch must be positive: epoch=%d", epoch)
		return err
	}

	return nil
}

func validateDecayRate(name string, rate float64) (err error) {
	if rate <= 0 || rate > 1 || math.IsNaN(rate) || math.IsInf(rate, 0) {
		err = fmt.Errorf("optimizer: %s must be greater than 0 and less than or equal to 1: %s=%g", name, name, rate)
		return err
	}

	return nil
}

func nilScheduleError(name string) (err error) {
	err = fmt.Errorf("optimizer: %s schedule is nil", name)
	return err
}

func invalidStepSizeError(stepSize int) (err error) {
	err = fmt.Errorf("optimizer: step decay step size must be positive: stepSize=%d", stepSize)
	return err
}
