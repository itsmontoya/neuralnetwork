//go:build !darwin || !cgo || !metal || purego

package matrix

import "github.com/itsmontoya/neuralnetwork/internal/device"

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

func categoricalCrossEntropyValueDevice(
	_ *Matrix,
	_ *Matrix,
	_ float32,
) (value float32, handled bool, err error) {
	return 0, false, nil
}

func categoricalCrossEntropyGradientDevice(
	_ *Matrix,
	_ *Matrix,
	_ *Matrix,
	_ float32,
) (handled bool, err error) {
	return false, nil
}

func sgdDevice(_ []device.ParameterUpdate, _ float32) (handled bool, err error) {
	return false, nil
}
