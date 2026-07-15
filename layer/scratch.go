package layer

import "github.com/itsmontoya/neuralnetwork/matrix"

func matrixScratch(current *matrix.Matrix, rows, cols int) (scratch *matrix.Matrix, err error) {
	if current != nil && current.Rows() == rows && current.Cols() == cols {
		scratch = current
		return scratch, nil
	}

	scratch, err = matrix.New(rows, cols)
	return scratch, err
}
