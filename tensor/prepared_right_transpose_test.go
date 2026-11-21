package tensor

import (
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/float32product"
)

func Test_Gemm_PreparedRightTranspose(t *testing.T) {
	var (
		previous = runtime.GOMAXPROCS(3)
		tests    = []struct {
			name              string
			rows, inner, cols int
			nonfinite         bool
		}{
			{name: "one row", rows: 1, inner: 64, cols: 8, nonfinite: true},
			{name: "two rows", rows: 2, inner: 64, cols: 17, nonfinite: true},
			{name: "three rows", rows: 3, inner: 64, cols: 17, nonfinite: true},
			{name: "group and one row", rows: 4, inner: 68, cols: 20, nonfinite: true},
			{name: "group and two rows", rows: 5, inner: 68, cols: 13, nonfinite: true},
			{name: "multiple groups", rows: 8, inner: 68, cols: 25},
			{name: "short contraction", rows: 1, inner: 60, cols: 12},
			{name: "uneven contraction", rows: 1, inner: 66, cols: 12},
			{name: "short output tile", rows: 1, inner: 64, cols: 7},
			{name: "uneven one row shard", rows: 1, inner: 2048, cols: 1047, nonfinite: true},
			{name: "uneven group shard", rows: 5, inner: 2048, cols: 775, nonfinite: true},
			{name: "long contraction", rows: 3, inner: 8192, cols: 17},
			{name: "empty contraction", rows: 2, cols: 9},
			{name: "empty output rows", inner: 64, cols: 8},
			{name: "empty output columns", rows: 2, inner: 64},
		}
	)
	t.Cleanup(func() { runtime.GOMAXPROCS(previous) })
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				random = rand.New(rand.NewPCG(9, 18))
				left   = make([]float32, tt.rows*tt.inner)
				right  = make([]float32, tt.cols*tt.inner)
				dirty  = make([]float32, tt.rows*tt.cols)
				bias   = make([]float32, tt.cols)
				shape  = []int64{int64(tt.cols), int64(tt.inner)}
			)
			for index := range left {
				left[index] = 2*random.Float32() - 1
			}
			for index := range right {
				right[index] = 2*random.Float32() - 1
			}
			for column := range bias {
				bias[column] = float32(column%7) / 5
				if tt.nonfinite {
					row := right[column*tt.inner:][:tt.inner]
					switch column % 8 {
					case 0:
						row[0] = math.Float32frombits(0xffc54321)
					case 1:
						row[1] = math.Float32frombits(0x7f854321)
					case 2:
						row[1] = float32(math.Inf(1))
					case 3:
						row[1] = float32(math.Inf(-1))
					case 4:
						clear(row)
						row[2] = math.Float32frombits(0x80000000)
					case 5:
						clear(row)
						row[3] = math.SmallestNonzeroFloat32
					case 6:
						row[0], row[1], row[2] = 1<<25, 1, -(1 << 25)
					}
				}
			}
			if tt.nonfinite {
				left[0], left[1], left[2], left[3] = 1, 1, 1, 1
				if tt.rows > 1 {
					left[tt.inner] = math.Float32frombits(0x7fc12345)
					left[tt.inner+1] = math.Float32frombits(0x7fa32123)
				}
			}
			for index := range dirty {
				dirty[index] = 123
			}
			var (
				input = newPreparedTestTensor(t, []int64{int64(tt.rows), int64(tt.inner)}, left)
				dense = newPreparedTestTensor(t, shape, right)
				add   = newPreparedTestTensor(t, []int64{int64(tt.cols)}, bias)
				want  = newPreparedTestTensor(t, []int64{int64(tt.rows), int64(tt.cols)}, dirty)
				got   = newPreparedTestTensor(t, want.Shape(), dirty)
			)
			weight, err := PrepareRightTranspose(shape, right)
			if err != nil {
				t.Fatal(err)
			}
			if len(weight.storage.floats) != len(right) {
				t.Fatal("prepared storage changed the weight count")
			}
			if weight.packed != (hostMatrixProduct && float32product.CanPackRightTranspose(tt.inner, tt.cols)) {
				t.Fatal("unexpected prepared layout")
			}
			for _, config := range []GemmConfig{
				{TransposeB: true, Alpha: 1},
				{TransposeB: true, Alpha: -0.75, Beta: 0.25},
			} {
				if err = GemmInto(input, dense, add, want, config); err != nil {
					t.Fatal(err)
				}
				if err = weight.GemmInto(input, add, got, config); err != nil {
					t.Fatal(err)
				}
				checkPreparedTestBits(t, got.floats, want.floats)
			}
		})
	}
}

func Test_Gemm_PreparedRightTransposeOwnership(t *testing.T) {
	var (
		shape  = []int64{8, 64}
		source = make([]float32, 8*64)
	)
	for index := range source {
		source[index] = float32(index-256) / 7
	}
	original := slices.Clone(source)
	weight, err := PrepareRightTranspose(shape, source)
	if err != nil {
		t.Fatal(err)
	}
	shape[0] = 999
	clear(source)
	copyShape := weight.Shape()
	copyShape[0] = 0
	if !slices.Equal(weight.Shape(), []int64{8, 64}) {
		t.Fatal("prepared weight shares caller shape storage")
	}
	dense, err := weight.Dense()
	if err != nil {
		t.Fatal(err)
	}
	checkPreparedTestBits(t, dense.floats, original)
	index, err := FromInt64([]int64{1}, []int64{-1})
	if err != nil {
		t.Fatal(err)
	}
	row := newPreparedTestTensor(t, []int64{1, 64}, make([]float32, 64))
	if err = GatherInto(dense, index, row, 0); err != nil {
		t.Fatal(err)
	}
	checkPreparedTestBits(t, row.floats, original[7*64:])

	for _, transposed := range []bool{false, true} {
		var (
			config = GemmConfig{TransposeA: transposed, TransposeB: transposed, Alpha: 1}
			shape  = []int64{1, 8}
		)
		if transposed {
			shape = []int64{64, 1}
		}
		left := newPreparedTestTensor(t, shape, original[:int(shape[0]*shape[1])])
		outShape, shapeErr := GemmShape(left.Shape(), dense.Shape(), config)
		if shapeErr != nil {
			t.Fatal(shapeErr)
		}
		want := newPreparedTestTensor(t, outShape, make([]float32, int(outShape[1])))
		got := newPreparedTestTensor(t, outShape, make([]float32, int(outShape[1])))
		if err = GemmInto(left, dense, nil, want, config); err != nil {
			t.Fatal(err)
		}
		if err = weight.GemmInto(left, nil, got, config); err != nil {
			t.Fatal(err)
		}
		checkPreparedTestBits(t, got.floats, want.floats)
	}
	clear(dense.floats)
	dense, err = weight.Dense()
	if err != nil {
		t.Fatal(err)
	}
	checkPreparedTestBits(t, dense.floats, original)
	for _, invalid := range [][]int64{{8, 64, 1}, {-1, 64}, {8, 63}} {
		failed, prepareErr := PrepareRightTranspose(invalid, original)
		if prepareErr == nil || failed != nil {
			t.Fatalf("accepted invalid preparation shape %v", invalid)
		}
		checkPreparedTestBits(t, dense.floats, original)
	}
}

func Test_Gemm_PreparedRightTransposeConcurrentRetry(t *testing.T) {
	const inner, cols = 2048, 512
	var (
		source  = make([]float32, inner*cols)
		config  = GemmConfig{TransposeB: true, Alpha: 1}
		inputs  [2]*Tensor
		outputs [2]*Tensor
		want    [2]*Tensor
		results [2]error
		group   sync.WaitGroup
	)
	for index := range source {
		source[index] = float32(index%47-23) / 17
	}
	weight, err := PrepareRightTranspose([]int64{cols, inner}, source)
	if err != nil {
		t.Fatal(err)
	}
	dense := newPreparedTestTensor(t, []int64{cols, inner}, source)
	for lane := range inputs {
		left := make([]float32, inner)
		for index := range left {
			left[index] = float32((index+lane)%13-6) / 11
		}
		inputs[lane] = newPreparedTestTensor(t, []int64{1, inner}, left)
		outputs[lane] = newPreparedTestTensor(t, []int64{1, cols}, make([]float32, cols))
		want[lane] = newPreparedTestTensor(t, []int64{1, cols}, make([]float32, cols))
		if err = GemmInto(inputs[lane], dense, nil, want[lane], config); err != nil {
			t.Fatal(err)
		}
	}
	for lane := range inputs {
		group.Go(func() {
			results[lane] = weight.GemmInto(inputs[lane], nil, outputs[lane], config)
		})
	}
	group.Wait()
	for lane, result := range results {
		if result != nil {
			t.Fatal(result)
		}
		checkPreparedTestBits(t, outputs[lane].floats, want[lane].floats)
		retained, valuesErr := outputs[lane].Float32Values()
		if valuesErr != nil {
			t.Fatal(valuesErr)
		}
		alias := *outputs[lane]
		if err = weight.GemmInto(inputs[lane], &alias, outputs[lane], config); err == nil {
			t.Fatal("accepted aliased bias with zero beta")
		}
		checkPreparedTestBits(t, outputs[lane].floats, retained)
		if err = weight.GemmInto(inputs[1-lane], nil, outputs[lane], config); err != nil {
			t.Fatal(err)
		}
		checkPreparedTestBits(t, outputs[lane].floats, want[1-lane].floats)
		checkPreparedTestBits(t, retained, want[lane].floats)
	}
}

func newPreparedTestTensor(t *testing.T, shape []int64, values []float32) (out *Tensor) {
	t.Helper()
	out, err := FromFloat32(shape, values)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func checkPreparedTestBits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d values, want %d", len(got), len(want))
	}
	for index, value := range want {
		if math.Float32bits(got[index]) != math.Float32bits(value) {
			t.Fatalf("value %d: got %08x, want %08x", index, math.Float32bits(got[index]), math.Float32bits(value))
		}
	}
}
