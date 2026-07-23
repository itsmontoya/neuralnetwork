//go:build darwin && cgo && metal && !purego

package scratch_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
	"github.com/itsmontoya/neuralnetwork/internal/scratch"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_MatrixPool_EvictionReleasesResidency(t *testing.T) {
	var (
		pool        scratch.MatrixPool
		left        *matrix.Matrix
		right       *matrix.Matrix
		result      *matrix.Matrix
		next        *matrix.Matrix
		counters    metaltest.Counters
		available   bool
		reused      bool
		err         error
		shapeOffset int
	)

	if _, available, err = device.SharedRuntime(); err != nil {
		t.Fatalf("SharedRuntime returned error: %v", err)
	}
	if !available {
		t.Skip("Metal device unavailable")
	}
	left = matrixPoolMetalMatrix(t, 256, 128, 0.25)
	right = matrixPoolMetalMatrix(t, 128, 128, -0.5)
	if result, reused, err = pool.Get(256, 128); err != nil {
		t.Fatalf("Get result returned error: %v", err)
	}
	if reused {
		t.Fatal("first Get reused a matrix")
	}
	if err = left.MatMulInto(right, result); err != nil {
		t.Fatalf("MatMulInto returned error: %v", err)
	}

	for shapeOffset = 1; shapeOffset <= 3; shapeOffset++ {
		if _, reused, err = pool.Get(1, shapeOffset); err != nil {
			t.Fatalf("warm-up Get 1x%d returned error: %v", shapeOffset, err)
		}
		if reused {
			t.Fatalf("warm-up Get 1x%d reused a matrix", shapeOffset)
		}
	}

	metaltest.Enable()
	defer metaltest.Disable()
	if _, reused, err = pool.Get(1, 4); err != nil {
		t.Fatalf("evicting Get returned error: %v", err)
	}
	if reused {
		t.Fatal("evicting Get reused a matrix")
	}
	counters = metaltest.Snapshot()
	if counters.ResultDownloads != 1 {
		t.Fatalf("eviction downloads = %d, want 1", counters.ResultDownloads)
	}

	metaltest.Reset()
	if next, err = matrix.New(256, 128); err != nil {
		t.Fatalf("New next returned error: %v", err)
	}
	if err = result.MatMulInto(right, next); err != nil {
		t.Fatalf("MatMulInto evicted result returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.BufferCreations != 2 || counters.InputUploads != 1 {
		t.Fatalf("post-eviction counters = %+v, want detached input re-upload and destination staging", counters)
	}
}

func matrixPoolMetalMatrix(tb testing.TB, rows, cols int, offset float32) (out *matrix.Matrix) {
	tb.Helper()

	var (
		values []float32
		index  int
		err    error
	)

	values = make([]float32, rows*cols)
	for index = range values {
		values[index] = offset + float32(index%23)/23
	}
	if out, err = matrix.FromSlice(rows, cols, values); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}
	return out
}
