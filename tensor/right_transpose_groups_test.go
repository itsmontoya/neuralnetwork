package tensor_test

import (
	"math"
	"runtime"
	"slices"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_Gemm_RightTransposeGroupsConcurrentRetry(t *testing.T) {
	var tests = []struct {
		name string
		rows int
	}{
		{name: "standalone pair", rows: 2},
		{name: "one remainder row", rows: 4},
		{name: "two remainder rows", rows: 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testGemmRightTransposeConcurrentRetry(t, tt.rows)
		})
	}
}

func testGemmRightTransposeConcurrentRetry(t *testing.T, rows int) {
	t.Helper()
	const width = 1031
	var (
		previous                   = runtime.GOMAXPROCS(1)
		shape                      = []int64{int64(rows), width}
		config                     = tensor.GemmConfig{TransposeB: true, Alpha: 0.5, Beta: -2}
		inputValues                = rightTransposeBenchmarkValues(rows * width)
		left, right, bias, badBias *tensor.Tensor
		retained, retry            *tensor.Tensor
		want, got                  []float32
		alias                      tensor.Tensor
		callers                    sync.WaitGroup
		err                        error
		equalBits                  = func(a, b float32) (equal bool) {
			return math.Float32bits(a) == math.Float32bits(b)
		}
	)
	t.Cleanup(func() { runtime.GOMAXPROCS(previous) })
	if left, err = tensor.FromFloat32(shape, inputValues); err != nil {
		t.Fatal(err)
	}
	if right, err = tensor.FromFloat32([]int64{width, width}, rightTransposeBenchmarkValues(width*width)); err != nil {
		t.Fatal(err)
	}
	if bias, err = tensor.FromFloat32([]int64{width}, inputValues[:width]); err != nil {
		t.Fatal(err)
	}
	if badBias, err = tensor.New(tensor.Float32, []int64{width - 1}); err != nil {
		t.Fatal(err)
	}
	if retained, err = tensor.New(tensor.Float32, shape); err != nil {
		t.Fatal(err)
	}
	if err = tensor.GemmInto(left, right, bias, retained, config); err != nil {
		t.Fatal(err)
	}
	if want, err = retained.Float32Values(); err != nil {
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
	if !slices.EqualFunc(got, inputValues, equalBits) {
		t.Fatal("rejected GEMM changed its aliased input")
	}
	if retry, err = tensor.FromFloat32(shape, inputValues); err != nil {
		t.Fatal(err)
	}
	if err = tensor.GemmInto(left, right, badBias, retry, config); err == nil {
		t.Fatal("GemmInto accepted an invalid bias")
	}
	if got, err = retry.Float32Values(); err != nil {
		t.Fatal(err)
	}
	if !slices.EqualFunc(got, inputValues, equalBits) {
		t.Fatal("rejected bias changed the dirty destination")
	}
	if err = tensor.GemmInto(left, right, bias, retry, config); err != nil {
		t.Fatal(err)
	}

	for range 4 {
		callers.Go(func() {
			var (
				destination *tensor.Tensor
				values      []float32
				callErr     error
			)
			if destination, callErr = tensor.FromFloat32(shape, inputValues); callErr != nil {
				t.Error(callErr)
				return
			}
			if callErr = tensor.GemmInto(left, right, bias, destination, config); callErr != nil {
				t.Error(callErr)
				return
			}
			if values, callErr = destination.Float32Values(); callErr != nil {
				t.Error(callErr)
				return
			}
			if !slices.EqualFunc(values, want, equalBits) {
				t.Error("concurrent GEMM differs from the serial result")
			}
		})
	}
	callers.Wait()
	for _, destination := range []*tensor.Tensor{retained, retry} {
		if got, err = destination.Float32Values(); err != nil {
			t.Fatal(err)
		}
		if !slices.EqualFunc(got, want, equalBits) {
			t.Fatal("retained GEMM result differs from the serial result")
		}
	}
}
