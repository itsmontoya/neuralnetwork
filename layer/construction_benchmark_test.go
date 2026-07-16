package layer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
)

var benchmarkConstructedDense *layer.Dense
var benchmarkConstructedBatchNormalization *layer.BatchNormalization

func Benchmark_NewDense_ColdPath(b *testing.B) {
	var (
		dense *layer.Dense
		err   error
		index int
	)

	b.ReportAllocs()
	for index = 0; index < b.N; index++ {
		dense, err = layer.NewDense(32, 64, layer.ZeroWeights)
		if err != nil {
			b.Fatalf("NewDense returned error: %v", err)
		}
	}

	benchmarkConstructedDense = dense
}

func Benchmark_NewBatchNormalization_ColdPath(b *testing.B) {
	var (
		batchNormalization *layer.BatchNormalization
		err                error
		index              int
	)

	b.ReportAllocs()
	for index = 0; index < b.N; index++ {
		batchNormalization, err = layer.NewBatchNormalization(64)
		if err != nil {
			b.Fatalf("NewBatchNormalization returned error: %v", err)
		}
	}

	benchmarkConstructedBatchNormalization = batchNormalization
}
