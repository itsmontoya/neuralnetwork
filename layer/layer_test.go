package layer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_Layer_Interface(t *testing.T) {
	var _ layer.Layer = mockLayer{}
}

type mockLayer struct{}

func (m mockLayer) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	output = input
	return output, nil
}

func (m mockLayer) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	inputGradient = outputGradient
	return inputGradient, nil
}
