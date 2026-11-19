package float32product

import (
	"runtime"
	"sync"
)

const (
	rightTransposeColumnTile       = 4
	rightTransposeWeightsPerWorker = 1 << 19 // 2 MiB of FP32 weights
	rightTransposeMaxWorkers       = 8
)

func rightTransposeOneRowWorkerCount(inner, cols int) (workers int) {
	var tiles = cols / rightTransposeColumnTile
	// Count only complete tiles toward the weight budget. Small calls avoid
	// the scheduler query, and a single scheduler thread uses the serial kernel.
	workers = min(rightTransposeMaxWorkers, tiles, tiles*rightTransposeColumnTile*inner/rightTransposeWeightsPerWorker)
	if workers < 2 {
		return 1
	}
	return min(workers, runtime.GOMAXPROCS(0))
}

func rightTransposeOneRowParallel(left, right, output []float32, workers int) {
	var (
		inner          = len(left)
		tiles          = len(output) / rightTransposeColumnTile
		tilesPerWorker = tiles / workers
		extraTiles     = tiles % workers
		start          int
		group          sync.WaitGroup
	)
	group.Add(workers - 1)
	for worker := 0; worker < workers-1; worker++ {
		end := start + tilesPerWorker*rightTransposeColumnTile
		if worker < extraTiles {
			end += rightTransposeColumnTile
		}
		weights := right[start*inner : end*inner]
		result := output[start:end]
		go func() {
			rightTransposeOneRow(left, weights, result)
			group.Done()
		}()
		start = end
	}

	// The caller computes the last range, including any scalar column tail.
	rightTransposeOneRow(left, right[start*inner:], output[start:])
	group.Wait()
}
