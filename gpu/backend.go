package gpu

import "github.com/itsmontoya/neuralnetwork/internal/device"

type backend interface {
	newBuffer(count uint64) (buffer allocation, err error)
	newScope() (scope commandBuffer, err error)
}

type allocation interface {
	Upload(values []float32) (err error)
	Download(destination []float32) (err error)
	Release()
}

type commandBuffer interface {
	encodeFill(destination allocation, value float32) (err error)
	encodeCopy(source, destination allocation) (err error)
	encodeMatMul(left, right, destination allocation, m, n, k uint32, rightTransposed bool) (err error)
	Commit() (err error)
	Wait() (err error)
	Release() (err error)
}

type metalBackend struct {
	runtime *device.Runtime
}

func (m metalBackend) newBuffer(count uint64) (buffer allocation, err error) {
	var native *device.Buffer
	if native, err = m.runtime.NewBuffer(count); err != nil {
		return nil, err
	}
	return native, nil
}

func (m metalBackend) newScope() (scope commandBuffer, err error) {
	var native *device.Scope
	if native, err = m.runtime.NewScope(); err != nil {
		return nil, err
	}
	return metalCommandBuffer{Scope: native}, nil
}

type metalCommandBuffer struct {
	*device.Scope
}

func (m metalCommandBuffer) encodeFill(destination allocation, value float32) (err error) {
	return m.EncodeFill(destination.(*device.Buffer), value)
}

func (m metalCommandBuffer) encodeCopy(source, destination allocation) (err error) {
	return m.EncodeCopy(source.(*device.Buffer), destination.(*device.Buffer))
}

func (m metalCommandBuffer) encodeMatMul(left, right, destination allocation, rows, cols, inner uint32, rightTransposed bool) (err error) {
	var (
		operation = device.OperationTensorMatMul
		rightRows = inner
		rightCols = cols
	)
	if rightTransposed {
		operation = device.OperationTensorMatMulRightTranspose
		rightRows, rightCols = rightCols, rightRows
	}
	return m.EncodeMatMul(left.(*device.Buffer), right.(*device.Buffer), destination.(*device.Buffer),
		rows, inner, rightRows, rightCols, rows, cols, operation)
}
