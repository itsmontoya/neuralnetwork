//go:build !darwin || !cgo || !metal || purego

package device

type unavailableBackend struct{}

func newPlatformBackend() (runtimeBackend backend) {
	runtimeBackend = unavailableBackend{}
	return runtimeBackend
}

func (unavailableBackend) available() (available bool, err error) {
	return false, nil
}

func (unavailableBackend) newBuffer(uint64) (handle any, err error) {
	return nil, ErrUnavailable
}

func (unavailableBackend) upload(any, []float32) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) download(any, []float32) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) releaseBuffer(any) {}

func (unavailableBackend) newScope() (handle any, err error) {
	return nil, ErrUnavailable
}

func (unavailableBackend) encodeCopy(any, any, any, uint64) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) encodeFill(any, any, float32, uint64) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) encodeAddRowVector(any, any, any, uint32, uint32) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) encodeAddScaled(any, any, any, any, float32, uint32) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) encodeReLU(any, any, any, uint32) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) encodeReLUBackward(any, any, any, any, uint32) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) encodeSoftmaxRows(any, any, any, uint32, uint32) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) encodeSoftmaxRowsBackward(any, any, any, any, uint32, uint32) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) encodeColumnSums(any, any, any, uint32, uint32, bool) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) encodeCategoricalCrossEntropy(
	any,
	any,
	any,
	any,
	uint32,
	uint32,
	float32,
) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) encodeCategoricalCrossEntropyGradient(
	any,
	any,
	any,
	any,
	uint32,
	uint32,
	float32,
) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) encodeMatMul(
	any,
	any,
	any,
	any,
	matMulDimensions,
	Operation,
) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) commit(any) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) completed(any) (complete bool, err error) {
	return false, ErrUnavailable
}

func (unavailableBackend) wait(any) (err error) {
	return ErrUnavailable
}

func (unavailableBackend) releaseScope(any) {}

func (unavailableBackend) resourceSnapshot() (snapshot ResourceSnapshot) {
	return snapshot
}

func (unavailableBackend) resetResourcePeaks() (err error) {
	return nil
}
