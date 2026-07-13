//go:build darwin && cgo && metal && !purego

package matrix

func matMulInto(left, right, result *Matrix) {
	if metalRunMatMul(left, right, result, metalMatMulStandard) {
		return
	}

	matMulIntoPure(left, right, result)
}

func matMulLeftTransposeInto(left, right, result *Matrix) {
	if metalRunMatMul(left, right, result, metalMatMulLeftTranspose) {
		return
	}

	matMulLeftTransposeIntoPure(left, right, result)
}

func matMulRightTransposeInto(left, right, result *Matrix) {
	if metalRunMatMul(left, right, result, metalMatMulRightTranspose) {
		return
	}

	matMulRightTransposeIntoPure(left, right, result)
}
