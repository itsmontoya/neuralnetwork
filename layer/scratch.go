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

func floatScratch(current []float64, length int) (scratch []float64) {
	if len(current) == length {
		scratch = current
		return scratch
	}

	scratch = make([]float64, length)
	return scratch
}
