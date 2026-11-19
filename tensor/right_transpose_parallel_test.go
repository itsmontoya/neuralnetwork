package tensor_test

import (
	"runtime"
	"slices"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_Gemm_RightTransposeSingleRowConcurrentRetry(t *testing.T) {
	const width = 2048
	var (
		previous                  = runtime.GOMAXPROCS(1)
		shape                     = []int64{1, width}
		config                    = tensor.GemmConfig{TransposeB: true, Alpha: 0.5, Beta: -2}
		inputValues               = rightTransposeBenchmarkValues(width)
		weightValues              = rightTransposeBenchmarkValues(width * width)
		left, right, bias, output *tensor.Tensor
		want, got                 []float32
		alias                     tensor.Tensor
		workers                   sync.WaitGroup
		err                       error
	)
	t.Cleanup(func() { runtime.GOMAXPROCS(previous) })
	if left, err = tensor.FromFloat32(shape, inputValues); err != nil {
		t.Fatal(err)
	}
	if right, err = tensor.FromFloat32([]int64{width, width}, weightValues); err != nil {
		t.Fatal(err)
	}
	if bias, err = tensor.FromFloat32([]int64{width}, inputValues); err != nil {
		t.Fatal(err)
	}
	if output, err = tensor.New(tensor.Float32, shape); err != nil {
		t.Fatal(err)
	}
	if err = tensor.GemmInto(left, right, bias, output, config); err != nil {
		t.Fatal(err)
	}
	if want, err = output.Float32Values(); err != nil {
		t.Fatal(err)
	}

	runtime.GOMAXPROCS(8)
	alias = *left
	if err = tensor.GemmInto(left, right, bias, &alias, config); err == nil {
		t.Fatal("GemmInto accepted a destination sharing the input storage")
	}
	if got, err = left.Float32Values(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, inputValues) {
		t.Fatal("rejected GEMM changed its aliased input")
	}
	for range 8 {
		workers.Go(func() {
			var (
				destination *tensor.Tensor
				values      []float32
				workerErr   error
			)
			if destination, workerErr = tensor.FromFloat32(shape, inputValues); workerErr != nil {
				t.Error(workerErr)
				return
			}
			if workerErr = tensor.GemmInto(left, right, bias, destination, config); workerErr != nil {
				t.Error(workerErr)
				return
			}
			if values, workerErr = destination.Float32Values(); workerErr != nil {
				t.Error(workerErr)
				return
			}
			if !slices.Equal(values, want) {
				t.Error("concurrent GEMM differs from the serial reference")
			}
		})
	}
	workers.Wait()
	if got, err = output.Float32Values(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Fatal("later GEMM changed the retained output")
	}
}
