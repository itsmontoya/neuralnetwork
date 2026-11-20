package float32product

import (
	"runtime"
	"sync"
)

const (
	rightTransposeColumnTile            = 4
	rightTransposeWeightsPerWorker      = 1 << 19 // 2 MiB of FP32 weights
	rightTransposeMaxWorkers            = 8
	rightTransposeGroupWeightsPerWorker = 1 << 18 // 1 MiB of FP32 weight visits
	rightTransposeGroupMaxWorkers       = 8
)

func dispatchRightTransposeGroups(left, right, output []float32, groups, inner, cols int) {
	var workers = rightTransposeGroupWorkerCount(groups, inner, cols)
	if workers > 1 {
		rightTransposeGroupsParallel(left, right, output, groups, inner, cols, workers)
		return
	}
	rightTransposeGroups(left, right, output, groups, inner, cols, cols)
}

func rightTransposeGroupWorkerCount(groups, inner, cols int) (workers int) {
	var (
		tiles = cols / rightTransposeColumnTile
		// Round the per-group budget up to avoid multiplying the weight
		// footprint by the number of groups.
		weightsPerGroup = 1 + (rightTransposeGroupWeightsPerWorker-1)/groups
	)
	workers = min(rightTransposeGroupMaxWorkers, tiles, tiles*rightTransposeColumnTile*inner/weightsPerGroup)
	if workers < 2 {
		return 1
	}
	return min(workers, runtime.GOMAXPROCS(0))
}

func rightTransposeGroupsParallel(left, right, output []float32, groups, inner, cols, workers int) {
	var (
		tiles          = cols / rightTransposeColumnTile
		tilesPerWorker = tiles / workers
		extraTiles     = tiles % workers
		start          int
		group          sync.WaitGroup
	)
	group.Add(workers - 1)
	for worker := 0; worker < workers-1; worker++ {
		var end = start + tilesPerWorker*rightTransposeColumnTile
		if worker < extraTiles {
			end += rightTransposeColumnTile
		}
		var (
			weights = right[start*inner : end*inner]
			result  = output[start:]
			width   = end - start
		)
		go func() {
			rightTransposeGroups(left, weights, result, groups, inner, cols, width)
			group.Done()
		}()
		start = end
	}

	// The caller owns the final tile range and the scalar column tail.
	rightTransposeGroups(left, right[start*inner:], output[start:], groups, inner, cols, cols-start)
	group.Wait()
}

func dispatchRightTransposeOneRow(left, right, output []float32) {
	var workers = rightTransposeOneRowWorkerCount(len(left), len(output))
	if workers > 1 {
		rightTransposeOneRowParallel(left, right, output, workers)
		return
	}
	rightTransposeOneRow(left, right, output)
}

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
