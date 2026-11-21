//go:build !arm64 || purego || race

package float32product

const packedRightARM64 = false

func packRightTransposeTile(dense, packed []float32, inner int) {
	for contraction := range inner {
		for lane := range packedRightColumns {
			packed[contraction*packedRightColumns+lane] = dense[lane*inner+contraction]
		}
	}
}

func rightTransposePackedRange(left, packed, output []float32, rows, inner, stride, start, end int) {
	weights := make([]float32, (end-start)*inner)
	UnpackRightTranspose(packed, weights, inner, start)
	rightTransposeRows(left, weights, output[start:], rows, inner, stride, end-start)
}
