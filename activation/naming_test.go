package activation_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
)

func Test_Name_BuiltInActivations(t *testing.T) {
	type testcase struct {
		name       string
		function   activation.Activation
		serialized string
	}

	tests := []testcase{
		{
			name:       "elu",
			function:   activation.ELU{},
			serialized: "elu",
		},
		{
			name:       "gelu",
			function:   activation.GELU{},
			serialized: "gelu",
		},
		{
			name:       "leaky relu",
			function:   activation.LeakyReLU{},
			serialized: "leaky_relu",
		},
		{
			name:       "linear",
			function:   activation.Linear{},
			serialized: "linear",
		},
		{
			name:       "relu",
			function:   activation.ReLU{},
			serialized: "relu",
		},
		{
			name:       "sigmoid",
			function:   activation.Sigmoid{},
			serialized: "sigmoid",
		},
		{
			name:       "softmax",
			function:   activation.Softmax{},
			serialized: "softmax",
		},
		{
			name:       "tanh",
			function:   activation.Tanh{},
			serialized: "tanh",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				name string
				err  error
			)

			name, err = activation.Name(tt.function)
			if err != nil {
				t.Fatalf("Name returned error: %v", err)
			}

			if name != tt.serialized {
				t.Fatalf("Name = %q, want %q", name, tt.serialized)
			}
		})
	}
}

func Test_FromName_BuiltInActivations(t *testing.T) {
	type testcase struct {
		name string
	}

	tests := []testcase{
		{name: "elu"},
		{name: "gelu"},
		{name: "leaky_relu"},
		{name: "linear"},
		{name: "relu"},
		{name: "sigmoid"},
		{name: "softmax"},
		{name: "tanh"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				function activation.Activation
				name     string
				err      error
			)

			function, err = activation.FromName(tt.name)
			if err != nil {
				t.Fatalf("FromName returned error: %v", err)
			}

			name, err = activation.Name(function)
			if err != nil {
				t.Fatalf("Name returned error: %v", err)
			}

			if name != tt.name {
				t.Fatalf("round-trip name = %q, want %q", name, tt.name)
			}
		})
	}
}

func Test_FromName_RejectsUnknownName(t *testing.T) {
	var (
		function activation.Activation
		err      error
	)

	function, err = activation.FromName("swish")
	if err == nil {
		t.Fatal("FromName error = nil, want error")
	}

	if function != nil {
		t.Fatal("FromName returned activation on error")
	}
}
