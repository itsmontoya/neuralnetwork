//go:build darwin && cgo && metal && !purego

package matrix

func matMulInto(left, right, result *Matrix) (err error) {
	err = metalRunMatMul(left, right, result, metalMatMulStandard)
	return err
}

func matMulLeftTransposeInto(left, right, result *Matrix) (err error) {
	err = metalRunMatMul(left, right, result, metalMatMulLeftTranspose)
	return err
}

func matMulRightTransposeInto(left, right, result *Matrix) (err error) {
	err = metalRunMatMul(left, right, result, metalMatMulRightTranspose)
	return err
}
