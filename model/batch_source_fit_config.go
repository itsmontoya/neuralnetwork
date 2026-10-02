package model

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

// BatchSourceFitConfig configures multi-epoch fitting from lazy batch sources.
type BatchSourceFitConfig struct {
	// Epochs is the number of complete training-source passes.
	Epochs int
	// Random is passed to the training source at each epoch reset.
	Random *rand.Rand
	// Optimizer updates trainable model parameters after each batch.
	Optimizer optimizer.Optimizer
	// LearningRateSchedule updates the optimizer learning rate before each epoch.
	LearningRateSchedule optimizer.LearningRateSchedule
	// EarlyStopping stops training when monitored loss stops improving.
	EarlyStopping *EarlyStopping
	// Loss evaluates predictions and supplies prediction gradients.
	Loss loss.Loss
	// ValidationSource is evaluated after each epoch when provided.
	ValidationSource BatchSource
	// Accuracy evaluates optional training and validation accuracy.
	Accuracy AccuracyFunc
	// Callback receives completed epoch metrics without library-owned printing.
	Callback FitCallback
}

func (c BatchSourceFitConfig) validate() (err error) {
	if c.Epochs <= 0 {
		err = fmt.Errorf("model: batch source fit epochs must be positive: epochs=%d", c.Epochs)
		return err
	}
	if c.Optimizer == nil {
		err = errors.New("model: batch source fit optimizer is nil")
		return err
	}
	if c.Loss == nil {
		err = errors.New("model: batch source fit loss is nil")
		return err
	}
	if err = c.EarlyStopping.validate(); err != nil {
		return err
	}
	return nil
}
