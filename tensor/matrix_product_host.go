//go:build !metal || purego || !darwin || !cgo

package tensor

import "github.com/itsmontoya/neuralnetwork/internal/float32product"

// matrixProductValues stages the product so a failed call cannot partially
// change its destination. The validated inputs remain private and read-only.
func matrixProductValues(left, right []float32, leftShape, rightShape []int64, transposeA, transposeB bool) (values []float32, err error) {
	rows, cols := leftShape[0], rightShape[1]
	if transposeA {
		rows = leftShape[1]
	}

	if transposeB {
		cols = rightShape[0]
	}

	values = make([]float32, int(rows*cols))
	err = matrixProductInto(left, right, leftShape, rightShape, transposeA, transposeB, values)
	return values, err
}

// matrixProductInto reuses a disjoint block of the call-owned product workspace.
// Shape, storage bounds, and alias validation belong to MatMulInto/GemmInto.
func matrixProductInto(left, right []float32, leftShape, rightShape []int64, transposeA, transposeB bool, destination []float32) (err error) {
	rows, inner, cols := int(leftShape[0]), int(leftShape[1]), int(rightShape[1])
	if transposeA {
		rows, inner = inner, rows
	}

	if transposeB {
		cols = int(rightShape[0])
	}

	switch {
	case transposeA && transposeB:
		float32product.BothTranspose(left, right, destination, rows, inner, cols)
	case transposeA:
		float32product.LeftTranspose(left, right, destination, rows, inner, cols)
	case transposeB:
		float32product.RightTranspose(left, right, destination, rows, inner, cols)
	default:
		float32product.Standard(left, right, destination, rows, inner, cols)
	}
	return nil
}
