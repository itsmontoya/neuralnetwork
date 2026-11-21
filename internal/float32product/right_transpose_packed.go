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
// It uses the dense worker budget, with shards aligned to complete packed tiles.
func RightTransposePacked(left, packed, output []float32, rows, inner, cols int) {
	var workers int
	if rows == 1 {
		workers = rightTransposeOneRowWorkerCount(inner, cols)
	} else {
		workers = rightTransposeRowsWorkerCount(rows, inner, cols)
	}
	var tiles = cols / packedRightColumns
	workers = min(workers, tiles)
	if workers <= 1 {
		rightTransposePackedRange(left, packed, output, rows, inner, cols, 0, cols)
		return
	}

	var (
		tilesPerWorker = tiles / workers
		extraTiles     = tiles % workers
		start          int
		group          sync.WaitGroup
	)
	group.Add(workers - 1)
	for worker := 0; worker < workers-1; worker++ {
		end := start + tilesPerWorker*packedRightColumns
		if worker < extraTiles {
			end += packedRightColumns
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

// rightTransposePackedRange overwrites the requested columns of each output
// row. The dense tail begins after the last complete packed tile.
func rightTransposePackedRange(left, packed, output []float32, rows, inner, stride, start, end int) {
	var (
		packedEnd = min(end, stride/packedRightColumns*packedRightColumns)
		tileStart = start / packedRightColumns * packedRightColumns
	)
	if start < packedEnd && start != tileStart {
		var tileEnd = min(tileStart+packedRightColumns, packedEnd)
		rightTransposePackedPartialTile(left, packed[tileStart*inner:], output[start:], rows, inner, stride, start-tileStart, tileEnd-tileStart)
		start = tileEnd
	}
	for ; start+packedRightColumns <= packedEnd; start += packedRightColumns {
		var row int
		for ; row+3 <= rows; row += 3 {
			rightTransposePackedTile(left[row*inner:], packed[start*inner:], output[row*stride+start:], inner, stride, 3)
		}
		if row < rows {
			rightTransposePackedTile(left[row*inner:], packed[start*inner:], output[row*stride+start:], inner, stride, rows-row)
		}
	}
	if start < packedEnd {
		rightTransposePackedPartialTile(left, packed[start*inner:], output[start:], rows, inner, stride, 0, packedEnd-start)
		start = packedEnd
	}
	if start < end {
		rightTransposeRows(left, packed[start*inner:], output[start:], rows, inner, stride, end-start)
	}
}

func rightTransposePackedPartialTile(left, packed, output []float32, rows, inner, stride, first, end int) {
	// Use the same arithmetic as a complete tile and write only the lanes
	// owned by this worker. Scratch holds at most one group's outputs (96 bytes).
	var scratch [3 * packedRightColumns]float32
	for row := 0; row < rows; row += 3 {
		var groupRows = min(3, rows-row)
		rightTransposePackedTile(left[row*inner:], packed, scratch[:], inner, packedRightColumns, groupRows)
		for groupRow := range groupRows {
			var offset = groupRow * packedRightColumns
			copy(output[(row+groupRow)*stride:][:end-first], scratch[offset+first:offset+end])
		}
	}
}
