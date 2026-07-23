//go:build !darwin || !cgo || !metal || purego

package matrix

func addRowVectorInPlaceDevice(_ *Matrix, _ *Matrix) (handled bool, err error) {
	return false, nil
}

func reluForwardDevice(_ *Matrix, _ *Matrix) (handled bool, err error) {
	return false, nil
}

func addIntoDevice(_ *Matrix, _ *Matrix, _ *Matrix) (handled bool, err error) {
	return false, nil
}

func reluBackwardDevice(_ *Matrix, _ *Matrix, _ *Matrix) (handled bool, err error) {
	return false, nil
}

func softmaxRowsIntoDevice(_ *Matrix, _ *Matrix) (handled bool, err error) {
	return false, nil
}

func softmaxRowsBackwardIntoDevice(_ *Matrix, _ *Matrix, _ *Matrix) (handled bool, err error) {
	return false, nil
}

func columnSumsIntoDevice(_ *Matrix, _ *Matrix) (handled bool, err error) {
	return false, nil
}

func accumulateColumnSumsIntoDevice(_ *Matrix, _ *Matrix) (handled bool, err error) {
	return false, nil
}

func resetDevice(_ *Matrix) (handled bool, err error) {
	return false, nil
}
