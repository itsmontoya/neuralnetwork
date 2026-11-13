package tensor

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"slices"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_Arithmetic_EqualShapeMatrixBits(t *testing.T) {
	var (
		operations = []struct {
			name   string
			run    func(*Tensor, *Tensor, *Tensor) error
			matrix func(*matrix.Matrix, *matrix.Matrix, *matrix.Matrix) error
		}{
			{name: "Add", run: AddInto, matrix: (*matrix.Matrix).AddInto},
			{name: "Sub", run: SubInto, matrix: (*matrix.Matrix).SubtractInto},
			{name: "Mul", run: MulInto, matrix: (*matrix.Matrix).MultiplyElementsInto},
		}
		shapes = [][]int64{
			nil, {0}, {2, 0, 3}, {1}, {1, 10}, {15}, {16}, {17}, {4096},
			{1, 1, 1, 1, 1, 1, 1, 1, 17},
		}
		bits = []uint32{
			0, 0x80000000, 0x3fc00000, 0xc0100000, 0x7f800000,
			0xff800000, 0x7fc12345, 0x00000001, 0x7f7fffff,
		}
	)
	for _, operation := range operations {
		for _, shape := range shapes {
			t.Run(fmt.Sprint(operation.name, shape), func(t *testing.T) {
				var (
					left, right, output, saved      *Tensor
					leftMatrix, rightMatrix, result *matrix.Matrix
					first, second                   *Tensor
					leftBefore, rightBefore, want   []float32
					outputStorage                   *float32
					shapeStorage                    *int64
					identity                        *byte
					err                             error
				)
				if left, err = New(Float32, shape); err != nil {
					t.Fatal(err)
				}
				if right, err = New(Float32, shape); err != nil {
					t.Fatal(err)
				}
				if output, err = New(Float32, shape); err != nil {
					t.Fatal(err)
				}
				for index := range left.floats {
					left.floats[index] = math.Float32frombits(bits[index%len(bits)])
					right.floats[index] = math.Float32frombits(bits[(index+3)%len(bits)])
				}
				outputStorage = slicesElement(output.floats)
				if len(output.shape) > 0 {
					shapeStorage = &output.shape[0]
				}
				identity = output.identity
				for call := range 3 {
					first, second = left, right
					if call == 1 {
						for index := range right.floats {
							right.floats[index] = float32(index%29-14) * 0.125
						}
					}
					if call == 2 {
						second = left
					}
					leftBefore = slices.Clone(left.floats)
					rightBefore = slices.Clone(right.floats)
					if len(output.floats) > 0 {
						if leftMatrix, err = matrix.FromSlice(1, len(first.floats), first.floats); err != nil {
							t.Fatal(err)
						}
						if rightMatrix, err = matrix.FromSlice(1, len(second.floats), second.floats); err != nil {
							t.Fatal(err)
						}
						if result, err = matrix.New(1, len(output.floats)); err != nil {
							t.Fatal(err)
						}
						if err = operation.matrix(leftMatrix, rightMatrix, result); err != nil {
							t.Fatal(err)
						}
						if want, err = result.Values(); err != nil {
							t.Fatal(err)
						}
					}
					for index := range output.floats {
						output.floats[index] = 99
					}
					if err = operation.run(first, second, output); err != nil {
						t.Fatal(err)
					}
					requireAdapterBits(t, output.floats, want)
					requireAdapterBits(t, left.floats, leftBefore)
					requireAdapterBits(t, right.floats, rightBefore)
					if output.identity != identity || slicesElement(output.floats) != outputStorage || !slices.Equal(output.shape, shape) {
						t.Fatal("operation replaced destination storage or metadata")
					}
					if len(output.shape) > 0 && &output.shape[0] != shapeStorage {
						t.Fatal("operation replaced destination shape storage")
					}
					if call == 0 {
						if saved, err = output.Clone(); err != nil {
							t.Fatal(err)
						}
					}
				}
				left, right, first, second = nil, nil, nil, nil
				runtime.GC()
				requireAdapterBits(t, output.floats, want)
				if len(output.floats) > 0 {
					var savedBefore = slices.Clone(saved.floats)
					output.floats[0] = 123
					requireAdapterBits(t, saved.floats, savedBefore)
				}
			})
		}
	}
}

func Test_MatrixBinaryInto_OperationError(t *testing.T) {
	var (
		left, right, output *Tensor
		before              = []float32{99, math.Float32frombits(0x80000000)}
		failure             = errors.New("arithmetic failed after writing private output")
		err                 error
	)
	if left, err = FromFloat32([]int64{2}, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	if right, err = left.Clone(); err != nil {
		t.Fatal(err)
	}
	if output, err = FromFloat32([]int64{2}, before); err != nil {
		t.Fatal(err)
	}
	err = matrixBinaryInto(left, right, output, func(a, b, destination *matrix.Matrix) (operationErr error) {
		if operationErr = a.AddInto(b, destination); operationErr != nil {
			return operationErr
		}
		return failure
	})
	if !errors.Is(err, failure) {
		t.Fatalf("error=%v, want wrapped operation failure", err)
	}
	requireAdapterBits(t, output.floats, before)
}

func Test_Arithmetic_EqualShapeConcurrentReaders(t *testing.T) {
	var operations = []struct {
		name string
		run  func(*Tensor, *Tensor, *Tensor) error
	}{
		{name: "Add", run: AddInto},
		{name: "Sub", run: SubInto},
		{name: "Mul", run: MulInto},
	}
	for _, operation := range operations {
		t.Run(operation.name, func(t *testing.T) {
			var (
				left, right, reference  *Tensor
				leftBefore, rightBefore []float32
				workers                 sync.WaitGroup
				err                     error
			)
			if left, err = New(Float32, []int64{1, 4096}); err != nil {
				t.Fatal(err)
			}
			if right, err = New(Float32, left.shape); err != nil {
				t.Fatal(err)
			}
			if reference, err = New(Float32, left.shape); err != nil {
				t.Fatal(err)
			}
			for index := range left.floats {
				left.floats[index] = float32((index+3)%29-14) * 0.125
				right.floats[index] = float32((index+7)%29-14) * 0.125
			}
			leftBefore, rightBefore = slices.Clone(left.floats), slices.Clone(right.floats)
			if err = operation.run(left, right, reference); err != nil {
				t.Fatal(err)
			}
			for range 8 {
				workers.Add(1)
				go func() {
					var (
						output    *Tensor
						workerErr error
					)
					defer workers.Done()
					if output, workerErr = New(Float32, reference.shape); workerErr != nil {
						t.Error(workerErr)
						return
					}
					for range 10 {
						if workerErr = operation.run(left, right, output); workerErr != nil {
							t.Error(workerErr)
							return
						}
						requireAdapterBits(t, output.floats, reference.floats)
					}
				}()
			}
			workers.Wait()
			requireAdapterBits(t, left.floats, leftBefore)
			requireAdapterBits(t, right.floats, rightBefore)
		})
	}
}

func requireAdapterBits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length=%d, want %d", len(got), len(want))
	}
	for index := range want {
		if math.Float32bits(got[index]) != math.Float32bits(want[index]) {
			t.Fatalf("value[%d]=%08x, want %08x", index, math.Float32bits(got[index]), math.Float32bits(want[index]))
		}
	}
}

func slicesElement(values []float32) (element *float32) {
	if len(values) == 0 {
		return nil
	}
	return &values[0]
}
