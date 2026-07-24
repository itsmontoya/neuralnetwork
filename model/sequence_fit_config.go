package model

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

// SequenceFitConfig configures length-aware multi-epoch training.
type SequenceFitConfig struct {
	// Epochs is the number of complete passes over the training dataset.
	Epochs int
	// BatchSize is the maximum number of samples in each training batch.
	BatchSize int
	// Shuffle enables per-epoch sample shuffling before batches are built.
	Shuffle bool
	// Random supplies deterministic shuffling when Shuffle is true.
	Random *rand.Rand
	// Optimizer updates trainable model parameters after each batch.
	Optimizer optimizer.Optimizer
	// LearningRateSchedule updates the optimizer learning rate before each epoch.
	LearningRateSchedule optimizer.LearningRateSchedule
	// EarlyStopping stops training when monitored loss stops improving.
	EarlyStopping *EarlyStopping
	// Loss evaluates predictions and supplies prediction gradients.
	Loss loss.Loss
	// ValidationData is evaluated with its aligned lengths after each epoch.
	ValidationData *data.SequenceDataset
	// Accuracy evaluates optional training and validation accuracy.
	Accuracy AccuracyFunc
	// Callback receives completed epoch metrics without library-owned printing.
	Callback FitCallback
}

func (c SequenceFitConfig) validate() (err error) {
	if c.Epochs <= 0 {
		err = fmt.Errorf("model: sequence fit epochs must be positive: epochs=%d", c.Epochs)
		return err
	}

	if c.BatchSize <= 0 {
		err = fmt.Errorf("model: sequence fit batch size must be positive: batchSize=%d", c.BatchSize)
		return err
	}

	if c.Optimizer == nil {
		err = errors.New("model: sequence fit optimizer is nil")
		return err
	}

	if c.Loss == nil {
		err = errors.New("model: sequence fit loss is nil")
		return err
	}

	if c.Shuffle && c.Random == nil {
		err = errors.New("model: sequence fit random source is nil when shuffle is enabled")
		return err
	}

	if err = c.EarlyStopping.validate(); err != nil {
		return err
	}

	return nil
}
