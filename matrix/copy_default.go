//go:build !metal || purego || !darwin || !cgo

package matrix

func copyMatrix(source, destination *Matrix) (err error) {
	err = copyMatrixHost(source, destination)
	return err
}
