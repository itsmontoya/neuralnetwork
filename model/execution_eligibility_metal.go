//go:build darwin && cgo && metal && !purego

package model

import (
	"math"

	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func deviceExecutionEligible(
	layers []layer.Layer,
	value *matrix.Matrix,
) (eligible bool) {
	var (
		current    layer.Layer
		dense      *layer.Dense
		operations uint64
		ok         bool
	)

	if value == nil || value.Rows() <= 0 {
		return false
	}

	for _, current = range layers {
		if dense, ok = current.(*layer.Dense); !ok || dense == nil {
			continue
		}
		operations = denseMatMulOperations(
			value.Rows(),
			dense.InputSize(),
			dense.OutputSize(),
		)
		if device.MatMulEligible(operations) {
			return true
		}
	}

	return false
}

func denseMatMulOperations(rows, inputSize, outputSize int) (operations uint64) {
	var (
		rowCount    uint64
		inputCount  uint64
		outputCount uint64
	)

	if rows <= 0 || inputSize <= 0 || outputSize <= 0 {
		return 0
	}

	rowCount = uint64(rows)
	inputCount = uint64(inputSize)
	outputCount = uint64(outputSize)
	if rowCount > math.MaxUint64/inputCount {
		return math.MaxUint64
	}
	operations = rowCount * inputCount
	if operations > math.MaxUint64/outputCount {
		return math.MaxUint64
	}

	operations *= outputCount
	return operations
}
