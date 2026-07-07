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
	if result, err = source.SelectRows(indexes); err != nil {
		err = fmt.Errorf("data: select matrix rows: %w", err)
		return nil, err
	}

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
