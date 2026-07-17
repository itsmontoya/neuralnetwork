package scratch_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/scratch"
)

var float32PoolResult []float32

func Test_Float32Pool_ZeroValueAndDirtyReuse(t *testing.T) {
	var (
		pool   scratch.Float32Pool
		first  []float32
		second []float32
		reused bool
		err    error
	)

	first, reused, err = pool.Get(3)
	if err != nil {
		t.Fatalf("first Get returned error: %v", err)
	}
	if reused {
		t.Fatal("first Get reused a buffer, want a miss")
	}
	first[2] = 7

	second, reused, err = pool.Get(3)
	if err != nil {
		t.Fatalf("second Get returned error: %v", err)
	}
	if !reused {
		t.Fatal("second Get did not reuse the warmed buffer")
	}
	if &second[0] != &first[0] {
		t.Fatal("second Get returned a different buffer")
	}
	if second[2] != 7 {
		t.Fatalf("reused buffer value = %g, want dirty value 7", second[2])
	}
}

func Test_Float32Pool_FourLengthWarmupAndNoAliasing(t *testing.T) {
	var (
		pool    scratch.Float32Pool
		buffers [4][]float32
		got     []float32
		reused  bool
		err     error
		index   int
	)

	for index = range buffers {
		buffers[index], reused, err = pool.Get(index + 1)
		if err != nil {
			t.Fatalf("Get length %d returned error: %v", index+1, err)
		}
		if reused {
			t.Fatalf("Get length %d reused a buffer during warm-up", index+1)
		}
		buffers[index][0] = float32(index + 1)
	}

	for index = range buffers {
		got, reused, err = pool.Get(index + 1)
		if err != nil {
			t.Fatalf("reused Get length %d returned error: %v", index+1, err)
		}
		if !reused {
			t.Fatalf("Get length %d missed after four-length warm-up", index+1)
		}
		if &got[0] != &buffers[index][0] {
			t.Fatalf("Get length %d returned an aliased buffer", index+1)
		}
		if got[0] != float32(index+1) {
			t.Fatalf("length %d value = %g, want %d", index+1, got[0], index+1)
		}
	}
}

func Test_Float32Pool_DeterministicEviction(t *testing.T) {
	var (
		pool        scratch.Float32Pool
		buffers     [5][]float32
		replacement []float32
		reused      bool
		err         error
		index       int
	)

	for index = 0; index < 4; index++ {
		buffers[index], reused, err = pool.Get(index + 1)
		if err != nil {
			t.Fatalf("warm-up Get length %d returned error: %v", index+1, err)
		}
		if reused {
			t.Fatalf("warm-up Get length %d reused a buffer", index+1)
		}
	}

	if _, reused, err = pool.Get(2); err != nil {
		t.Fatalf("most-recent Get returned error: %v", err)
	}
	if !reused {
		t.Fatal("most-recent Get did not reuse length 2")
	}

	buffers[4], reused, err = pool.Get(5)
	if err != nil {
		t.Fatalf("fifth-length Get returned error: %v", err)
	}
	if reused {
		t.Fatal("fifth-length Get reused a buffer, want a miss")
	}

	for index = 1; index < len(buffers); index++ {
		if _, reused, err = pool.Get(index + 1); err != nil {
			t.Fatalf("retained Get length %d returned error: %v", index+1, err)
		}
		if !reused {
			t.Fatalf("length %d was evicted instead of least-recent length 1", index+1)
		}
	}

	replacement, reused, err = pool.Get(1)
	if err != nil {
		t.Fatalf("evicted-length Get returned error: %v", err)
	}
	if reused {
		t.Fatal("evicted length 1 was unexpectedly reused")
	}
	if &replacement[0] == &buffers[0][0] {
		t.Fatal("evicted length returned its previous buffer")
	}
}

func Test_Float32Pool_InvalidLength(t *testing.T) {
	var (
		pool   scratch.Float32Pool
		got    []float32
		reused bool
		err    error
	)

	got, reused, err = pool.Get(-1)
	if err == nil {
		t.Fatal("Get(-1) error is nil")
	}
	if got != nil {
		t.Fatal("Get(-1) returned a buffer")
	}
	if reused {
		t.Fatal("Get(-1) reported reuse")
	}
}

func Test_Float32Pool_ZeroLength(t *testing.T) {
	var (
		pool   scratch.Float32Pool
		got    []float32
		reused bool
		err    error
	)

	got, reused, err = pool.Get(0)
	if err != nil {
		t.Fatalf("first Get(0) returned error: %v", err)
	}
	if reused {
		t.Fatal("first Get(0) reused a buffer")
	}
	if len(got) != 0 {
		t.Fatalf("first Get(0) length = %d, want 0", len(got))
	}

	got, reused, err = pool.Get(0)
	if err != nil {
		t.Fatalf("second Get(0) returned error: %v", err)
	}
	if !reused {
		t.Fatal("second Get(0) did not reuse the zero-length buffer")
	}
}

func Test_Float32Pool_Allocations(t *testing.T) {
	var (
		hitPool         scratch.Float32Pool
		missPool        scratch.Float32Pool
		got             []float32
		reused          bool
		err             error
		nextLength      int
		hitAllocations  float64
		missAllocations float64
	)

	if got, reused, err = hitPool.Get(64); err != nil {
		t.Fatalf("warm-up Get returned error: %v", err)
	}
	if reused {
		t.Fatal("warm-up Get reused a buffer")
	}

	hitAllocations = testing.AllocsPerRun(100, func() {
		got, reused, err = hitPool.Get(64)
		if err != nil {
			panic(err)
		}
		if !reused {
			panic("warmed Float32Pool Get missed")
		}
	})
	float32PoolResult = got
	if hitAllocations != 0 {
		t.Fatalf("warmed Get allocations = %g, want 0", hitAllocations)
	}

	missAllocations = testing.AllocsPerRun(100, func() {
		nextLength = nextLength%5 + 1
		got, reused, err = missPool.Get(nextLength)
		if err != nil {
			panic(err)
		}
		if reused {
			panic("five-length Float32Pool Get hit")
		}
	})
	float32PoolResult = got
	if missAllocations != 1 {
		t.Fatalf("miss Get allocations = %g, want 1", missAllocations)
	}
}
