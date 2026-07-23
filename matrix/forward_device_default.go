//go:build !darwin || !cgo || !metal || purego

package matrix

func addRowVectorInPlaceDevice(_ *Matrix, _ *Matrix) (handled bool, err error) {
	return false, nil
}

func reluForwardDevice(_ *Matrix, _ *Matrix) (handled bool, err error) {
	return false, nil
}

func softmaxRowsIntoDevice(_ *Matrix, _ *Matrix) (handled bool, err error) {
	return false, nil
}
