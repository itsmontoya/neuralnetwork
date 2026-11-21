//go:build arm64 && !purego && !race

package float32product

const packedRightARM64 = true

func packRightTransposeTile(dense, packed []float32, inner int) {
	packRightTransposeTileARM64(&dense[0], &packed[0], inner)
}

// packRightTransposeTileARM64 transposes eight weight rows into one tile.
// The positive contraction length must be divisible by four.
//
//go:noescape
func packRightTransposeTileARM64(dense, packed *float32, inner int)

func rightTransposePackedRange(left, packed, output []float32, rows, inner, stride, start, end int) {
	if start%packedRightColumns != 0 {
		// An eight-column dense kernel tile can cross two prepared tiles.
		// Restore that shard's original layout to keep its exact kernel cuts.
		weights := make([]float32, (end-start)*inner)
		UnpackRightTranspose(packed, weights, inner, start)
		rightTransposeRows(left, weights, output[start:], rows, inner, stride, end-start)
		return
	}

	fullEnd := end / packedRightColumns * packedRightColumns
	for column := start; column < fullEnd; column += packedRightColumns {
		weights := &packed[column*inner]
		var row int
		for ; row+3 <= rows; row += 3 {
			rightTransposePackedTileARM64(&left[row*inner], weights, &output[row*stride+column], inner, stride, 3)
		}
		if row < rows {
			rightTransposePackedTileARM64(&left[row*inner], weights, &output[row*stride+column], inner, stride, rows-row)
		}
	}
	if fullEnd == end {
		return
	}
	if end == stride {
		rightTransposeRows(left, packed[fullEnd*inner:], output[fullEnd:], rows, inner, stride, end-fullEnd)
		return
	}
	weights := make([]float32, (end-fullEnd)*inner)
	UnpackRightTranspose(packed, weights, inner, fullEnd)
	rightTransposeRows(left, weights, output[fullEnd:], rows, inner, stride, end-fullEnd)
}

// rightTransposePackedTileARM64 overwrites one to three rows of eight outputs.
// Every K contains eight adjacent weights and the caller supplies K % 4 == 0.
//
//go:noescape
func rightTransposePackedTileARM64(left, weights, output *float32, inner, stride, rows int)
