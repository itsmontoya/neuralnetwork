package model

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
)

// NewEarlyStopping constructs early stopping with patience and minDelta.
func NewEarlyStopping(patience int, minDelta float32) (out *EarlyStopping, err error) {
	if patience <= 0 {
		err = fmt.Errorf("model: early stopping patience must be positive: patience=%d", patience)
		return nil, err
	}

	if minDelta < 0 || f32.IsNaN(minDelta) || f32.IsInf(minDelta, 0) {
		err = fmt.Errorf("model: early stopping min delta must be non-negative and finite: minDelta=%g", minDelta)
		return nil, err
	}

	var e EarlyStopping
	e.patience = patience
	e.minDelta = minDelta
	return &e, nil
}

// EarlyStopping configures Fit to stop when monitored loss stops improving.
//
// Fit monitors validation loss when validation data is configured, otherwise it
// monitors training loss. An improvement must decrease the monitored loss by at
// least MinDelta to reset patience.
type EarlyStopping struct {
	patience int
	minDelta float32
}

// Patience returns the number of consecutive non-improving epochs before stopping.
func (e *EarlyStopping) Patience() (patience int) {
	if e == nil {
		return 0
	}

	patience = e.patience
	return patience
}

// MinDelta returns the minimum loss decrease required to count as improvement.
func (e *EarlyStopping) MinDelta() (minDelta float32) {
	if e == nil {
		return 0
	}

	minDelta = e.minDelta
	return minDelta
}

func (e *EarlyStopping) validate() (err error) {
	if e == nil {
		return nil
	}

	if e.patience <= 0 {
		err = fmt.Errorf("model: early stopping patience must be positive: patience=%d", e.patience)
		return err
	}

	if e.minDelta < 0 || f32.IsNaN(e.minDelta) || f32.IsInf(e.minDelta, 0) {
		err = fmt.Errorf("model: early stopping min delta must be non-negative and finite: minDelta=%g", e.minDelta)
		return err
	}

	return nil
}

type earlyStoppingState struct {
	config      *EarlyStopping
	bestLoss    float32
	initialized bool
	waitCount   int
}

func newEarlyStoppingState(config *EarlyStopping) (state earlyStoppingState) {
	state.config = config
	return state
}

func (s *earlyStoppingState) observe(metrics EpochMetrics) (stop bool) {
	var lossValue float32

	if s == nil || s.config == nil {
		return false
	}

	lossValue = monitoredLoss(metrics)
	if !s.initialized {
		s.bestLoss = lossValue
		s.initialized = true
		return false
	}

	if lossValue < s.bestLoss-s.config.minDelta {
		s.bestLoss = lossValue
		s.waitCount = 0
		return false
	}

	s.waitCount++
	stop = s.waitCount >= s.config.patience
	return stop
}

func monitoredLoss(metrics EpochMetrics) (lossValue float32) {
	if metrics.HasValidationLoss {
		lossValue = metrics.ValidationLoss
		return lossValue
	}

	lossValue = metrics.Loss
	return lossValue
}
