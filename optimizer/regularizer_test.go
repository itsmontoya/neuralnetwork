package optimizer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_Regularizer_Interface(t *testing.T) {
	var _ optimizer.Regularizer = (*optimizer.L1)(nil)
	var _ optimizer.Regularizer = (*optimizer.L2WeightDecay)(nil)
}
