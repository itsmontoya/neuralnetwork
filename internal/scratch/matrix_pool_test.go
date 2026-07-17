package scratch_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/scratch"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var matrixPoolResult *matrix.Matrix

func Test_MatrixPool_ZeroValueAndDirtyReuse(t *testing.T) {
	var (
		pool   scratch.MatrixPool
		first  *matrix.Matrix
		second *matrix.Matrix
		value  float32
		reused bool
		err    error
	)

	first, reused, err = pool.Get(2, 3)
	if err != nil {
		t.Fatalf("first Get returned error: %v", err)
	}
	if reused {
		t.Fatal("first Get reused a matrix, want a miss")
	}
	if err = first.Fill(7); err != nil {
		t.Fatalf("Fill returned error: %v", err)
	}

	second, reused, err = pool.Get(2, 3)
	if err != nil {
		t.Fatalf("second Get returned error: %v", err)
	}
	if !reused {
		t.Fatal("second Get did not reuse the warmed matrix")
	}
	if second != first {
		t.Fatal("second Get returned a different matrix")
	}
	if value, err = second.At(1, 2); err != nil {
		t.Fatalf("At returned error: %v", err)
	}
	if value != 7 {
		t.Fatalf("reused matrix value = %g, want dirty value 7", value)
	}
}

func Test_MatrixPool_FourShapeWarmupAndNoAliasing(t *testing.T) {
	type shape struct {
		rows int
		cols int
	}

	var (
		pool     scratch.MatrixPool
		shapes   []shape
		matrices []*matrix.Matrix
		got      *matrix.Matrix
		value    float32
		reused   bool
		err      error
		index    int
	)

	shapes = []shape{
		{rows: 1, cols: 4},
		{rows: 2, cols: 3},
		{rows: 3, cols: 2},
		{rows: 4, cols: 1},
	}
	matrices = make([]*matrix.Matrix, len(shapes))
	for index = range shapes {
		matrices[index], reused, err = pool.Get(shapes[index].rows, shapes[index].cols)
		if err != nil {
			t.Fatalf("Get shape %dx%d returned error: %v", shapes[index].rows, shapes[index].cols, err)
		}
		if reused {
			t.Fatalf("Get shape %dx%d reused a matrix during warm-up", shapes[index].rows, shapes[index].cols)
		}
		if err = matrices[index].Set(0, 0, float32(index+1)); err != nil {
			t.Fatalf("Set shape %dx%d returned error: %v", shapes[index].rows, shapes[index].cols, err)
		}
	}

	for index = range shapes {
		got, reused, err = pool.Get(shapes[index].rows, shapes[index].cols)
		if err != nil {
			t.Fatalf("reused Get shape %dx%d returned error: %v", shapes[index].rows, shapes[index].cols, err)
		}
		if !reused {
			t.Fatalf("Get shape %dx%d missed after four-shape warm-up", shapes[index].rows, shapes[index].cols)
		}
		if got != matrices[index] {
			t.Fatalf("Get shape %dx%d returned an aliased matrix", shapes[index].rows, shapes[index].cols)
		}
		if value, err = got.At(0, 0); err != nil {
			t.Fatalf("At shape %dx%d returned error: %v", shapes[index].rows, shapes[index].cols, err)
		}
		if value != float32(index+1) {
			t.Fatalf("shape %dx%d value = %g, want %d", shapes[index].rows, shapes[index].cols, value, index+1)
		}
	}
}

func Test_MatrixPool_DeterministicEviction(t *testing.T) {
	var (
		pool        scratch.MatrixPool
		matrices    [5]*matrix.Matrix
		replacement *matrix.Matrix
		reused      bool
		err         error
		index       int
	)

	for index = 0; index < 4; index++ {
		matrices[index], reused, err = pool.Get(index+1, 1)
		if err != nil {
			t.Fatalf("warm-up Get length %d returned error: %v", index+1, err)
		}
		if reused {
			t.Fatalf("warm-up Get shape %dx1 reused a matrix", index+1)
		}
	}

	if _, reused, err = pool.Get(2, 1); err != nil {
		t.Fatalf("most-recent Get returned error: %v", err)
	}
	if !reused {
		t.Fatal("most-recent Get did not reuse shape 2x1")
	}

	matrices[4], reused, err = pool.Get(5, 1)
	if err != nil {
		t.Fatalf("fifth-shape Get returned error: %v", err)
	}
	if reused {
		t.Fatal("fifth-shape Get reused a matrix, want a miss")
	}

	for index = 1; index < len(matrices); index++ {
		if _, reused, err = pool.Get(index+1, 1); err != nil {
			t.Fatalf("retained Get shape %dx1 returned error: %v", index+1, err)
		}
		if !reused {
			t.Fatalf("shape %dx1 was evicted instead of least-recent shape 1x1", index+1)
		}
	}

	replacement, reused, err = pool.Get(1, 1)
	if err != nil {
		t.Fatalf("evicted-shape Get returned error: %v", err)
	}
	if reused {
		t.Fatal("evicted shape 1x1 was unexpectedly reused")
	}
	if replacement == matrices[0] {
		t.Fatal("evicted shape returned its previous matrix")
	}
}

func Test_MatrixPool_InvalidDimensions(t *testing.T) {
	type testcase struct {
		name string
		rows int
		cols int
	}

	var tests []testcase
	tests = []testcase{
		{name: "zero rows", rows: 0, cols: 1},
		{name: "zero columns", rows: 1, cols: 0},
		{name: "negative rows", rows: -1, cols: 1},
		{name: "negative columns", rows: 1, cols: -1},
		{name: "overflow", rows: int(^uint(0) >> 1), cols: 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				pool   scratch.MatrixPool
				got    *matrix.Matrix
				reused bool
				err    error
			)

			got, reused, err = pool.Get(tt.rows, tt.cols)
			if err == nil {
				t.Fatalf("Get(%d, %d) error is nil", tt.rows, tt.cols)
			}
			if got != nil {
				t.Fatalf("Get(%d, %d) returned a matrix", tt.rows, tt.cols)
			}
			if reused {
				t.Fatalf("Get(%d, %d) reported reuse", tt.rows, tt.cols)
			}
		})
	}
}

func Test_MatrixPool_Allocations(t *testing.T) {
	var (
		hitPool         scratch.MatrixPool
		missPool        scratch.MatrixPool
		got             *matrix.Matrix
		reused          bool
		err             error
		nextRows        int
		hitAllocations  float64
		missAllocations float64
	)

	if got, reused, err = hitPool.Get(8, 8); err != nil {
		t.Fatalf("warm-up Get returned error: %v", err)
	}
	if reused {
		t.Fatal("warm-up Get reused a matrix")
	}

	hitAllocations = testing.AllocsPerRun(100, func() {
		got, reused, err = hitPool.Get(8, 8)
		if err != nil {
			panic(err)
		}
		if !reused {
			panic("warmed MatrixPool Get missed")
		}
	})
	matrixPoolResult = got
	if hitAllocations != 0 {
		t.Fatalf("warmed Get allocations = %g, want 0", hitAllocations)
	}

	missAllocations = testing.AllocsPerRun(100, func() {
		nextRows = nextRows%5 + 1
		got, reused, err = missPool.Get(nextRows, 1)
		if err != nil {
			panic(err)
		}
		if reused {
			panic("five-shape MatrixPool Get hit")
		}
	})
	matrixPoolResult = got
	if missAllocations != 2 {
		t.Fatalf("miss Get allocations = %g, want 2", missAllocations)
	}
}
