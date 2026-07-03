package activation

import (
	"errors"
	"fmt"
)

const (
	activationNameELU       = "elu"
	activationNameGELU      = "gelu"
	activationNameLeakyReLU = "leaky_relu"
	activationNameLinear    = "linear"
	activationNameReLU      = "relu"
	activationNameSigmoid   = "sigmoid"
	activationNameSoftmax   = "softmax"
	activationNameTanh      = "tanh"
)

// Name returns the stable serialization name for a built-in activation.
func Name(function Activation) (name string, err error) {
	if function == nil {
		err = errors.New("activation: activation function is nil")
		return "", err
	}

	switch current := function.(type) {
	case ELU:
		name = activationNameELU
	case *ELU:
		if current == nil {
			err = errors.New("activation: activation function is nil")
			return "", err
		}

		name = activationNameELU
	case GELU:
		name = activationNameGELU
	case *GELU:
		if current == nil {
			err = errors.New("activation: activation function is nil")
			return "", err
		}

		name = activationNameGELU
	case LeakyReLU:
		name = activationNameLeakyReLU
	case *LeakyReLU:
		if current == nil {
			err = errors.New("activation: activation function is nil")
			return "", err
		}

		name = activationNameLeakyReLU
	case Linear:
		name = activationNameLinear
	case *Linear:
		if current == nil {
			err = errors.New("activation: activation function is nil")
			return "", err
		}

		name = activationNameLinear
	case ReLU:
		name = activationNameReLU
	case *ReLU:
		if current == nil {
			err = errors.New("activation: activation function is nil")
			return "", err
		}

		name = activationNameReLU
	case Sigmoid:
		name = activationNameSigmoid
	case *Sigmoid:
		if current == nil {
			err = errors.New("activation: activation function is nil")
			return "", err
		}

		name = activationNameSigmoid
	case Softmax:
		name = activationNameSoftmax
	case *Softmax:
		if current == nil {
			err = errors.New("activation: activation function is nil")
			return "", err
		}

		name = activationNameSoftmax
	case Tanh:
		name = activationNameTanh
	case *Tanh:
		if current == nil {
			err = errors.New("activation: activation function is nil")
			return "", err
		}

		name = activationNameTanh
	default:
		err = fmt.Errorf("activation: unsupported activation type %T", function)
		return "", err
	}

	return name, nil
}

// FromName constructs a built-in activation from its stable serialization name.
func FromName(name string) (function Activation, err error) {
	switch name {
	case activationNameELU:
		function = ELU{}
	case activationNameGELU:
		function = GELU{}
	case activationNameLeakyReLU:
		function = LeakyReLU{}
	case activationNameLinear:
		function = Linear{}
	case activationNameReLU:
		function = ReLU{}
	case activationNameSigmoid:
		function = Sigmoid{}
	case activationNameSoftmax:
		function = Softmax{}
	case activationNameTanh:
		function = Tanh{}
	default:
		err = fmt.Errorf("activation: unknown activation name %q", name)
		return nil, err
	}

	return function, nil
}
