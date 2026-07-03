package data

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func validateMatrixPair(leftName string, left *matrix.Matrix, rightName string, right *matrix.Matrix) (err error) {
	var (
		leftRows  int
		rightRows int
	)

	if err = validateMatrix(leftName, left); err != nil {
		return err
	}

	if err = validateMatrix(rightName, right); err != nil {
		return err
	}

	leftRows = left.Rows()
	rightRows = right.Rows()
	if leftRows != rightRows {
		err = fmt.Errorf("data: sample count mismatch: %s rows=%d, %s rows=%d", leftName, leftRows, rightName, rightRows)
		return err
	}

	return nil
}

func validateMatrix(name string, m *matrix.Matrix) (err error) {
	if m == nil {
		err = fmt.Errorf("data: %s matrix is nil", name)
		return err
	}

	if err = m.Validate(); err != nil {
		err = fmt.Errorf("data: %s matrix is invalid: %w", name, err)
		return err
	}

	return nil
}

func matrixRows(source *matrix.Matrix, indexes []int) (result *matrix.Matrix, err error) {
	var (
		sourceValues []float64
		resultValues []float64
		sourceRows   int
		cols         int
		index        int
		row          int
		sourceStart  int
		resultStart  int
	)

	if len(indexes) == 0 {
		err = errors.New("data: row indexes are empty")
		return nil, err
	}

	if sourceValues, err = source.Values(); err != nil {
		err = fmt.Errorf("data: source matrix is invalid: %w", err)
		return nil, err
	}

	sourceRows = source.Rows()
	cols = source.Cols()
	resultValues = make([]float64, len(indexes)*cols)

	for index, row = range indexes {
		if row < 0 || row >= sourceRows {
			err = fmt.Errorf("data: row index out of range: row=%d rows=%d", row, sourceRows)
			return nil, err
		}

		sourceStart = row * cols
		resultStart = index * cols
		copy(resultValues[resultStart:resultStart+cols], sourceValues[sourceStart:sourceStart+cols])
	}

	result, err = matrix.FromSlice(len(indexes), cols, resultValues)
	return result, err
}

func rowIndexes(count int) (indexes []int) {
	var index int

	indexes = make([]int, count)
	for index = range indexes {
		indexes[index] = index
	}

	return indexes
}

func shuffleIndexes(indexes []int, random *rand.Rand) {
	if random == nil {
		return
	}

	random.Shuffle(len(indexes), func(left, right int) {
		indexes[left], indexes[right] = indexes[right], indexes[left]
	})
}

func nilBatchError() (err error) {
	err = errors.New("data: batch is nil")
	return err
}
