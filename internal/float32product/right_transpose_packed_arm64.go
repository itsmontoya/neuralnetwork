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

func rightTransposePackedTile(left, packed, output []float32, inner, stride, rows int) {
	rightTransposePackedTileARM64(&left[0], &packed[0], &output[0], inner, stride, rows)
}

// rightTransposePackedTileARM64 overwrites one to three rows of eight outputs.
// Every K contains eight adjacent weights and the caller supplies K % 4 == 0.
//
//go:noescape
func rightTransposePackedTileARM64(left, weights, output *float32, inner, stride, rows int)
