//go:build !metal || purego || !darwin || !cgo

package matrix

func matMulInto(left, right, result *Matrix) (err error) {
	if err = left.ensureHostCurrent(); err != nil {
		return err
	}
	if err = right.ensureHostCurrent(); err != nil {
		return err
	}
	if err = result.markHostWrite(); err != nil {
		return err
	}
	matMulIntoPure(left, right, result)
	return nil
}

func matMulLeftTransposeInto(left, right, result *Matrix) (err error) {
	if err = left.ensureHostCurrent(); err != nil {
		return err
	}
	if err = right.ensureHostCurrent(); err != nil {
		return err
	}
	if err = result.markHostWrite(); err != nil {
		return err
	}
	matMulLeftTransposeIntoPure(left, right, result)
	return nil
}

func matMulRightTransposeInto(left, right, result *Matrix) (err error) {
	if err = left.ensureHostCurrent(); err != nil {
		return err
	}
	if err = right.ensureHostCurrent(); err != nil {
		return err
	}
	if err = result.markHostWrite(); err != nil {
		return err
	}
	matMulRightTransposeIntoPure(left, right, result)
	return nil
}
