package layer

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_SimpleRNN_ValidateRejectsInvalidParameterState(t *testing.T) {
	type testcase struct {
		name      string
		modify    func(recurrent *SimpleRNN)
		wantError string
	}

	tests := []testcase{
		{
			name: "nil input weights",
			modify: func(recurrent *SimpleRNN) {
				recurrent.inputWeights = nil
			},
			wantError: "input weights parameter is nil",
		},
		{
			name: "input weight shape mismatch",
			modify: func(recurrent *SimpleRNN) {
				recurrent.inputWeights = mustInternalSimpleRNNParameter(t, 1, 2)
			},
			wantError: "input weights shape mismatch",
		},
		{
			name: "recurrent weight shape mismatch",
			modify: func(recurrent *SimpleRNN) {
				recurrent.recurrentWeights = mustInternalSimpleRNNParameter(t, 1, 2)
			},
			wantError: "recurrent weights shape mismatch",
		},
		{
			name: "bias shape mismatch",
			modify: func(recurrent *SimpleRNN) {
				recurrent.biases = mustInternalSimpleRNNParameter(t, 2, 2)
			},
			wantError: "biases shape mismatch",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				recurrent *SimpleRNN
				err       error
			)

			recurrent = mustInternalSimpleRNN(t)
			tt.modify(recurrent)
			err = recurrent.validate()
			if err == nil {
				t.Fatal("validate error = nil, want error")
			}
			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("validate error = %q, want substring %q", err, tt.wantError)
			}
		})
	}
}

func mustInternalSimpleRNN(tb testing.TB) (recurrent *SimpleRNN) {
	var (
		config     SimpleRNNConfig
		inputShape SequenceShape
		err        error
	)

	tb.Helper()
	inputShape, err = NewSequenceShape(2, 2)
	if err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}
	config, err = NewSimpleRNNConfig(inputShape, 2)
	if err != nil {
		tb.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	recurrent, err = NewSimpleRNN(config, ZeroWeights, ZeroWeights)
	if err != nil {
		tb.Fatalf("NewSimpleRNN returned error: %v", err)
	}

	return recurrent
}

func mustInternalSimpleRNNParameter(tb testing.TB, rows, cols int) (parameter *optimizer.Parameter) {
	var (
		values *matrix.Matrix
		err    error
	)

	tb.Helper()
	values, err = matrix.New(rows, cols)
	if err != nil {
		tb.Fatalf("matrix.New returned error: %v", err)
	}
	parameter, err = optimizer.NewParameter(values)
	if err != nil {
		tb.Fatalf("NewParameter returned error: %v", err)
	}

	return parameter
}
