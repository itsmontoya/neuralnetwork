package model_test

import (
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_NewEarlyStopping(t *testing.T) {
	var (
		earlyStopping *model.EarlyStopping
		err           error
	)

	earlyStopping, err = model.NewEarlyStopping(3, 0.01)
	if err != nil {
		t.Fatalf("NewEarlyStopping returned error: %v", err)
	}

	if earlyStopping.Patience() != 3 {
		t.Fatalf("Patience = %d, want 3", earlyStopping.Patience())
	}

	testutil.RequireAlmostEqual(t, earlyStopping.MinDelta(), 0.01, epsilon)
}

func Test_NewEarlyStopping_ValidatesConfig(t *testing.T) {
	type testcase struct {
		name     string
		patience int
		minDelta float64
	}

	tests := []testcase{
		{
			name:     "zero patience",
			patience: 0,
			minDelta: 0,
		},
		{
			name:     "negative patience",
			patience: -1,
			minDelta: 0,
		},
		{
			name:     "negative min delta",
			patience: 1,
			minDelta: -0.1,
		},
		{
			name:     "nan min delta",
			patience: 1,
			minDelta: math.NaN(),
		},
		{
			name:     "infinite min delta",
			patience: 1,
			minDelta: math.Inf(1),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				earlyStopping *model.EarlyStopping
				err           error
			)

			earlyStopping, err = model.NewEarlyStopping(tt.patience, tt.minDelta)
			if err == nil {
				t.Fatal("NewEarlyStopping error = nil, want error")
			}

			if earlyStopping != nil {
				t.Fatal("NewEarlyStopping returned config on error")
			}
		})
	}
}
