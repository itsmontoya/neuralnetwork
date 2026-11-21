//go:build !arm64 || purego || race

package float32product

const packedRightARM64 = false

func rightTransposePackedRange(left, packed, output []float32, rows, inner, stride, start, end int) {
	weights := make([]float32, (end-start)*inner)
	UnpackRightTranspose(packed, weights, inner, start)
	rightTransposeRows(left, weights, output[start:], rows, inner, stride, end-start)
}
