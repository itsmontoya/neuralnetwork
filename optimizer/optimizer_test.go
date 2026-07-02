package optimizer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_Optimizer_Interface(t *testing.T) {
	var _ optimizer.Optimizer = (*optimizer.SGD)(nil)
	var _ optimizer.Optimizer = (*optimizer.Momentum)(nil)
	var _ optimizer.Optimizer = (*optimizer.Adam)(nil)
	var _ optimizer.Optimizer = &mockOptimizer{}
}

type mockOptimizer struct {
	learningRate float64
}

func (m *mockOptimizer) Update(parameters []*optimizer.Parameter) (err error) {
	return nil
}

func (m *mockOptimizer) LearningRate() (learningRate float64) {
	learningRate = m.learningRate
	return learningRate
}

func (m *mockOptimizer) SetLearningRate(learningRate float64) (err error) {
	m.learningRate = learningRate
	return nil
}
