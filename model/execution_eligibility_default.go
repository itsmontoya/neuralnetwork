//go:build !darwin || !cgo || !metal || purego

package model

import (
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func deviceExecutionEligible(
	[]layer.Layer,
	*matrix.Matrix,
) (eligible bool) {
	return false
}
