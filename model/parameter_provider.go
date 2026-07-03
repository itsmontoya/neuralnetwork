package model

import "github.com/itsmontoya/neuralnetwork/optimizer"

type parameterProvider interface {
	Parameters() (parameters []*optimizer.Parameter)
}
