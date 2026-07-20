package layer

import (
	"strings"
	"testing"
)

func Test_SimpleRNNConfig_ValidateRejectsZeroValue(t *testing.T) {
	var (
		config SimpleRNNConfig
		err    error
	)

	err = config.validate()
	if err == nil {
		t.Fatal("validate error = nil, want error")
	}

	if !strings.HasPrefix(err.Error(), "layer: ") {
		t.Fatalf("validate error = %q, want layer context", err)
	}
}

func Test_SimpleRNNConfig_ValidateRejectsMismatchedOutputShape(t *testing.T) {
	var (
		config      SimpleRNNConfig
		err         error
		inputShape  SequenceShape
		outputShape SequenceShape
	)

	inputShape, err = NewSequenceShape(2, 3)
	if err != nil {
		t.Fatalf("NewSequenceShape returned error: %v", err)
	}

	outputShape, err = NewSequenceShape(2, 5)
	if err != nil {
		t.Fatalf("NewSequenceShape returned error: %v", err)
	}

	config.inputShape = inputShape
	config.outputShape = outputShape
	config.hiddenSize = 4
	err = config.validate()
	if err == nil {
		t.Fatal("validate error = nil, want error")
	}

	if !strings.Contains(err.Error(), "got=2x5 want=2x4") {
		t.Fatalf("validate error = %q, want got/want output dimensions", err)
	}
}
