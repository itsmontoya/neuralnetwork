package model

import "github.com/itsmontoya/neuralnetwork/optimizer"

type parameterAppender interface {
	AppendParameters(parameters []*optimizer.Parameter) (out []*optimizer.Parameter)
}
