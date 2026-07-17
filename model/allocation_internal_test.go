package model

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

var allocationParameters []*optimizer.Parameter

func Test_Sequential_RebuildParametersDoesNotAllocateAfterWarmUp(t *testing.T) {
	var (
		dense       *layer.Dense
		batchNorm   *layer.BatchNormalization
		network     *Sequential
		allocations float64
		err         error
	)

	dense, err = layer.NewDense(2, 2, layer.ZeroWeights)
	if err != nil {
		t.Fatalf("NewDense returned error: %v", err)
	}

	batchNorm, err = layer.NewBatchNormalization(2)
	if err != nil {
		t.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	network, err = NewSequential(dense, batchNorm)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	allocationParameters = network.rebuildParameters()
	allocations = testing.AllocsPerRun(100, func() {
		allocationParameters = network.rebuildParameters()
	})
	if allocations != 0 {
		t.Fatalf("rebuildParameters allocations = %g, want 0", allocations)
	}
}
