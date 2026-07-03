package loss_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_Loss_Interface(t *testing.T) {
	var _ loss.Loss = loss.MeanSquaredError{}
	var _ loss.Loss = loss.BinaryCrossEntropy{}
	var _ loss.Loss = loss.CategoricalCrossEntropy{}
	var _ loss.Loss = mockLoss{}
}

type mockLoss struct{}

func (m mockLoss) Value(predictions, targets *matrix.Matrix) (value float64, err error) {
	value = 0
	return value, nil
}

func (m mockLoss) Gradient(predictions, targets *matrix.Matrix) (gradient *matrix.Matrix, err error) {
	gradient, err = predictions.Clone()
	return gradient, err
}
