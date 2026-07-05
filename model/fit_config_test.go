package model

import (
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_FitConfigValidateRejectsInvalidEarlyStopping(t *testing.T) {
	type testcase struct {
		name          string
		earlyStopping *EarlyStopping
	}

	var tests []testcase

	tests = []testcase{
		{
			name:          "zero patience",
			earlyStopping: &EarlyStopping{},
		},
		{
			name: "negative min delta",
			earlyStopping: &EarlyStopping{
				patience: 1,
				minDelta: -0.1,
			},
		},
		{
			name: "nan min delta",
			earlyStopping: &EarlyStopping{
				patience: 1,
				minDelta: math.NaN(),
			},
		},
		{
			name: "infinite min delta",
			earlyStopping: &EarlyStopping{
				patience: 1,
				minDelta: math.Inf(1),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				sgd    *optimizer.SGD
				config FitConfig
				err    error
			)

			sgd, err = optimizer.NewSGD(0.1)
			if err != nil {
				t.Fatalf("NewSGD returned error: %v", err)
			}

			config = FitConfig{
				Epochs:        1,
				BatchSize:     1,
				Optimizer:     sgd,
				Loss:          loss.MeanSquaredError{},
				EarlyStopping: tt.earlyStopping,
			}

			err = config.validate()
			if err == nil {
				t.Fatal("validate error = nil, want error")
			}
		})
	}
}
