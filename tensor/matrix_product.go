//go:build metal && !purego && darwin && cgo

package tensor

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// matrixProductValues copies validated nonempty blocks through existing matrix
// kernels. No tensor destination is changed if an adapter operation fails.
func matrixProductValues(left, right []float32, leftShape, rightShape []int64, transposeA, transposeB bool) (values []float32, err error) {
	var (
		leftMatrix, rightMatrix, output *matrix.Matrix
		rows, cols                      int64
	)

	if leftMatrix, err = matrix.FromSlice(int(leftShape[0]), int(leftShape[1]), left); err != nil {
		return nil, fmt.Errorf("tensor: left product matrix adapter: %w", err)
	}
	if rightMatrix, err = matrix.FromSlice(int(rightShape[0]), int(rightShape[1]), right); err != nil {
		return nil, fmt.Errorf("tensor: right product matrix adapter: %w", err)
	}
	rows, cols = leftShape[0], rightShape[1]
	if transposeA {
		rows = leftShape[1]
	}
	if transposeB {
		cols = rightShape[0]
	}
	if output, err = matrix.New(int(rows), int(cols)); err != nil {
		return nil, fmt.Errorf("tensor: product output matrix adapter: %w", err)
	}
	if transposeA && transposeB {
		if rightMatrix, err = rightMatrix.Transpose(); err != nil {
			return nil, fmt.Errorf("tensor: transposed product matrix adapter: %w", err)
		}
		err = leftMatrix.MatMulLeftTransposeInto(rightMatrix, output)
	} else if transposeA {
		err = leftMatrix.MatMulLeftTransposeInto(rightMatrix, output)
	} else if transposeB {
		err = leftMatrix.MatMulRightTransposeInto(rightMatrix, output)
	} else {
		err = leftMatrix.MatMulInto(rightMatrix, output)
	}
	if err != nil {
		return nil, fmt.Errorf("tensor: matrix product: %w", err)
	}
	if values, err = output.Values(); err != nil {
		return nil, fmt.Errorf("tensor: matrix product values: %w", err)
	}
	return values, nil
}

func matrixProductInto(left, right []float32, leftShape, rightShape []int64, transposeA, transposeB bool, destination []float32) (err error) {
	var values []float32
	if values, err = matrixProductValues(left, right, leftShape, rightShape, transposeA, transposeB); err != nil {
		return err
	}

	copy(destination, values)
	return nil
}
