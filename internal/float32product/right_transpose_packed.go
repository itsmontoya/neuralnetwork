package float32product

import "sync"

const packedRightColumns = 8

// CanPackRightTranspose reports whether a right operand can use packed tiles.
func CanPackRightTranspose(inner, cols int) (ok bool) {
	return packedRightARM64 && inner >= 64 && inner%4 == 0 && cols >= packedRightColumns
}

// PackRightTranspose copies complete output tiles in contraction-major order.
// The final partial tile stays row-major and needs no padding. The contraction
// length must meet the packed kernel's requirements.
func PackRightTranspose(dense, packed []float32, inner, cols int) {
	var column int
	for ; column+packedRightColumns <= cols; column += packedRightColumns {
		var offset = column * inner
		packRightTransposeTile(dense[offset:], packed[offset:], inner)
	}
	copy(packed[column*inner:], dense[column*inner:])
}

// UnpackRightTranspose copies the requested contiguous output rows to dense.
// The destination length determines the number of rows.
func UnpackRightTranspose(packed, dense []float32, inner, first int) {
	var (
		fullCols = len(packed) / inner / packedRightColumns * packedRightColumns
		count    = len(dense) / inner
		start    = first / packedRightColumns * packedRightColumns
		end      = first + count
	)
	for ; start < min(end, fullCols); start += packedRightColumns {
		tile := packed[start*inner:][:packedRightColumns*inner]
		for column := max(start, first); column < min(start+packedRightColumns, end); column++ {
			row := dense[(column-first)*inner:][:inner]
			lane := column - start
			for contraction := range row {
				row[contraction] = tile[contraction*packedRightColumns+lane]
			}
		}
	}
	if end > fullCols {
		start = max(first, fullCols)
		copy(dense[(start-first)*inner:], packed[start*inner:end*inner])
	}
}

// RightTransposePacked multiplies a dense left block by a packed right operand.
// Its worker count and four-column shard boundaries match RightTranspose.
func RightTransposePacked(left, packed, output []float32, rows, inner, cols int) {
	var workers int
	if rows == 1 {
		workers = rightTransposeOneRowWorkerCount(inner, cols)
	} else {
		workers = rightTransposeRowsWorkerCount(rows, inner, cols)
	}
	if workers == 1 {
		rightTransposePackedRange(left, packed, output, rows, inner, cols, 0, cols)
		return
	}

	var (
		tiles          = cols / rightTransposeColumnTile
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
		first := start
		go func() {
			rightTransposePackedRange(left, packed, output, rows, inner, cols, first, end)
			group.Done()
		}()
		start = end
	}
	rightTransposePackedRange(left, packed, output, rows, inner, cols, start, cols)
	group.Wait()
}
