package optimizer

import "github.com/itsmontoya/neuralnetwork/matrix"

type adamState struct {
	firstMoment  *matrix.Matrix
	secondMoment *matrix.Matrix
	step         int
}
