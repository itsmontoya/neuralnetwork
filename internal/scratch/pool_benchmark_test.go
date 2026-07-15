package scratch_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/scratch"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var benchmarkMatrixPool scratch.MatrixPool
var benchmarkFloat32Pool scratch.Float32Pool
var benchmarkMatrixPoolResult *matrix.Matrix
var benchmarkFloat32PoolResult []float32

func Benchmark_MatrixPool_WarmedHit(b *testing.B) {
	var (
		pool   scratch.MatrixPool
		result *matrix.Matrix
		reused bool
		err    error
		index  int
	)

	if result, reused, err = pool.Get(128, 64); err != nil {
		b.Fatalf("warm-up Get returned error: %v", err)
	}
	if reused {
		b.Fatal("warm-up Get reused a matrix")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for index = 0; index < b.N; index++ {
		result, reused, err = pool.Get(128, 64)
		if err != nil {
			b.Fatalf("Get returned error: %v", err)
		}
		if !reused {
			b.Fatal("warmed Get missed")
		}
	}

	benchmarkMatrixPoolResult = result
}

func Benchmark_MatrixPool_Miss(b *testing.B) {
	var (
		pool   scratch.MatrixPool
		result *matrix.Matrix
		reused bool
		err    error
		index  int
		rows   int
	)

	b.ReportAllocs()
	for index = 0; index < b.N; index++ {
		rows = index%5 + 1
		result, reused, err = pool.Get(rows, 1)
		if err != nil {
			b.Fatalf("Get returned error: %v", err)
		}
		if reused {
			b.Fatal("five-shape Get hit")
		}
	}

	benchmarkMatrixPoolResult = result
}

func Benchmark_Float32Pool_WarmedHit(b *testing.B) {
	var (
		pool   scratch.Float32Pool
		result []float32
		reused bool
		err    error
		index  int
	)

	if result, reused, err = pool.Get(8192); err != nil {
		b.Fatalf("warm-up Get returned error: %v", err)
	}
	if reused {
		b.Fatal("warm-up Get reused a buffer")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for index = 0; index < b.N; index++ {
		result, reused, err = pool.Get(8192)
		if err != nil {
			b.Fatalf("Get returned error: %v", err)
		}
		if !reused {
			b.Fatal("warmed Get missed")
		}
	}

	benchmarkFloat32PoolResult = result
}

func Benchmark_Float32Pool_Miss(b *testing.B) {
	var (
		pool   scratch.Float32Pool
		result []float32
		reused bool
		err    error
		index  int
		length int
	)

	b.ReportAllocs()
	for index = 0; index < b.N; index++ {
		length = index%5 + 1
		result, reused, err = pool.Get(length)
		if err != nil {
			b.Fatalf("Get returned error: %v", err)
		}
		if reused {
			b.Fatal("five-length Get hit")
		}
	}

	benchmarkFloat32PoolResult = result
}

func Benchmark_MatrixPool_FourShapeRetention(b *testing.B) {
	var (
		pool   scratch.MatrixPool
		rows   []int
		result *matrix.Matrix
		reused bool
		err    error
		index  int
		shape  int
	)

	rows = []int{128, 17, 1024, 257}
	b.ReportAllocs()
	b.ReportMetric(365056, "retained-data-B")
	for index = 0; index < b.N; index++ {
		pool = scratch.MatrixPool{}
		for shape = range rows {
			result, reused, err = pool.Get(rows[shape], 64)
			if err != nil {
				b.Fatalf("Get returned error: %v", err)
			}
			if reused {
				b.Fatal("four-shape warm-up reused a matrix")
			}
		}
		benchmarkMatrixPool = pool
	}

	benchmarkMatrixPoolResult = result
}

func Benchmark_Float32Pool_FourLengthRetention(b *testing.B) {
	var (
		pool    scratch.Float32Pool
		lengths []int
		result  []float32
		reused  bool
		err     error
		index   int
		length  int
	)

	lengths = []int{128 * 64, 17 * 64, 1024 * 64, 257 * 64}
	b.ReportAllocs()
	b.ReportMetric(365056, "retained-data-B")
	for index = 0; index < b.N; index++ {
		pool = scratch.Float32Pool{}
		for length = range lengths {
			result, reused, err = pool.Get(lengths[length])
			if err != nil {
				b.Fatalf("Get returned error: %v", err)
			}
			if reused {
				b.Fatal("four-length warm-up reused a buffer")
			}
		}
		benchmarkFloat32Pool = pool
	}

	benchmarkFloat32PoolResult = result
}
