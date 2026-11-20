//go:build !arm64 || purego || race

package float32product

func rightTransposeOneRow(left, right, output []float32) {
	rightTransposeOneRowGo(left, right, output)
}
