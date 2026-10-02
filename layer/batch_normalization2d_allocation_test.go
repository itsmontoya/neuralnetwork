package layer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_BatchNormalization2D_SteadyStateAllocations(t *testing.T) {
	var (
		batchNorm      *layer.BatchNormalization2D
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		err            error
	)

	batchNorm = mustBatchNormalization2D(t, 3, 2, 2, 0.9, 1e-5)
	input = allocationLayerMatrix(t, 8, 12)
	outputGradient = allocationLayerMatrix(t, 8, 12)
	if _, err = batchNorm.Forward(input); err != nil {
		t.Fatalf("warm-up Forward returned error: %v", err)
	}
	if _, err = batchNorm.Backward(outputGradient); err != nil {
		t.Fatalf("warm-up Backward returned error: %v", err)
	}
	if err = batchNorm.ResetGradients(); err != nil {
		t.Fatalf("ResetGradients returned error: %v", err)
	}

	requireMaxAllocs(t, "BatchNormalization2D Forward and Backward", 0, func() {
		if _, err = batchNorm.Forward(input); err != nil {
			panic(err)
		}
		if allocationLayerResult, err = batchNorm.Backward(outputGradient); err != nil {
			panic(err)
		}
	})
}
