//go:build !metal || purego || !darwin || !cgo

package matrix

func matMulInto(left, right, result *Matrix) {
	matMulIntoPure(left, right, result)
}

func matMulLeftTransposeInto(left, right, result *Matrix) {
	matMulLeftTransposeIntoPure(left, right, result)
}

func matMulRightTransposeInto(left, right, result *Matrix) {
	matMulRightTransposeIntoPure(left, right, result)
}
