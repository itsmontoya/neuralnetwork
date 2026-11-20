//go:build arm64 && !purego && !race

package float32product

const (
	rightTransposeOneRowARM64Columns = 8
	rightTransposeOneRowARM64MinK    = 64
)

func rightTransposeOneRow(left, right, output []float32) {
	inner := len(left)
	if inner < rightTransposeOneRowARM64MinK || inner%4 != 0 {
		rightTransposeOneRowGo(left, right, output)
		return
	}

	var column int
	for ; column+rightTransposeOneRowARM64Columns <= len(output); column += rightTransposeOneRowARM64Columns {
		rightTransposeOneRowTileARM64(left, &right[column*inner], &output[column])
	}

	if column < len(output) {
		rightTransposeOneRowGo(left, right[column*inner:], output[column:])
	}
}

// rightTransposeOneRowTileARM64 reads eight consecutive weight rows of len(left)
// elements and overwrites eight outputs. The caller supplies a positive
// contraction divisible by four and validated, disjoint storage.
//
//go:noescape
func rightTransposeOneRowTileARM64(left []float32, weights, output *float32)
