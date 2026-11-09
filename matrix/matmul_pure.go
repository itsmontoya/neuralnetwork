package matrix

import "github.com/itsmontoya/neuralnetwork/internal/float32product"

func matMulIntoPure(left, right, result *Matrix) {
	float32product.Standard(left.data, right.data, result.data, left.rows, left.cols, right.cols)
}

func matMulLeftTransposeIntoPure(left, right, result *Matrix) {
	float32product.LeftTranspose(left.data, right.data, result.data, left.cols, left.rows, right.cols)
}

func matMulRightTransposeIntoPure(left, right, result *Matrix) {
	float32product.RightTranspose(left.data, right.data, result.data, left.rows, left.cols, right.rows)
}
