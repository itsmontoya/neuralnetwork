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
	learningRate       float32
	updateFunc         func(parameters []*optimizer.Parameter) (err error)
	setLearningRateErr error
}

func (m *mockOptimizer) Update(parameters []*optimizer.Parameter) (err error) {
	if m.updateFunc != nil {
		err = m.updateFunc(parameters)
		return err
	}

	return nil
}

func (m *mockOptimizer) LearningRate() (learningRate float32) {
	learningRate = m.learningRate
	return learningRate
}

func (m *mockOptimizer) SetLearningRate(learningRate float32) (err error) {
	if m.setLearningRateErr != nil {
		err = m.setLearningRateErr
		return err
	}

	m.learningRate = learningRate
	return nil
}
