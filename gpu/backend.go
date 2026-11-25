package gpu

import "github.com/itsmontoya/neuralnetwork/internal/device"

type backend interface {
	newBuffer(count, bytes uint64, dtype DType) (buffer allocation, err error)
	newScope() (scope commandBuffer, err error)
}

type allocation interface {
	Upload(values []float32) (err error)
	Download(destination []float32) (err error)
	UploadUint16(values []uint16) (err error)
	DownloadUint16(destination []uint16) (err error)
	UploadBytes(values []byte) (err error)
	DownloadBytes(destination []byte) (err error)
	Release()
}

type commandBuffer interface {
	encodeFill(destination allocation, value float32) (err error)
	encodeCopy(source, destination allocation) (err error)
	encodeGatherRows(source, destination allocation, columns uint64, rows []int64, sourceDType, destinationDType DType) (err error)
	encodeMatMul(left, right, destination allocation, m, n, k uint32, rightTransposed bool, rightDType DType) (err error)
	encodeElementwise(left, right, destination allocation, operation device.ElementwiseOperation) (err error)
	encodeKVAppend(history, keys, values, target allocation, plane, prefix uint32, cacheDType DType) (err error)
	encodeAttentionMath(operation device.AttentionMathOperation, first, second, target, extra allocation, params [6]uint32, threads uint32) (err error)
	Commit() (err error)
	Wait() (err error)
	Release() (err error)
}

type metalBackend struct {
	runtime *device.Runtime
}

func (m metalBackend) newBuffer(count, bytes uint64, dtype DType) (buffer allocation, err error) {
	var native *device.Buffer
	if dtype == Float32 {
		if native, err = m.runtime.NewBuffer(count); err != nil {
			return nil, err
		}
		return native, nil
	}
	if native, err = m.runtime.NewByteBuffer(bytes); err != nil {
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

func (m metalCommandBuffer) encodeGatherRows(source, destination allocation, columns uint64, rows []int64, sourceDType, destinationDType DType) (err error) {
	if sourceDType == destinationDType {
		return m.EncodeGatherRows(source.(*device.Buffer), destination.(*device.Buffer), columns, sourceDType.Bytes(), rows)
	}
	var nativeDType uint32 = 1
	if sourceDType == BFloat16 {
		nativeDType = 2
	}
	return m.EncodeGatherRowsMixed(source.(*device.Buffer), destination.(*device.Buffer), columns, rows, nativeDType)
}

func (m metalCommandBuffer) encodeElementwise(left, right, destination allocation, operation device.ElementwiseOperation) (err error) {
	var second *device.Buffer
	if right != nil {
		second = right.(*device.Buffer)
	}
	return m.EncodeElementwise(left.(*device.Buffer), second, destination.(*device.Buffer), operation)
}

func (m metalCommandBuffer) encodeMatMul(left, right, destination allocation, rows, cols, inner uint32, rightTransposed bool, rightDType DType) (err error) {
	var (
		operation device.Operation
		rightRows = inner
		rightCols = cols
	)
	if rightTransposed {
		rightRows, rightCols = rightCols, rightRows
		switch rightDType {
		case Float16:
			operation = device.OperationTensorMatMulRightTransposeFloat16
		case BFloat16:
			operation = device.OperationTensorMatMulRightTransposeBFloat16
		case Q8_0:
			operation = device.OperationTensorMatMulRightTransposeQ8_0
		case Q4_0:
			operation = device.OperationTensorMatMulRightTransposeQ4_0
		default:
			operation = device.OperationTensorMatMulRightTranspose
		}
	} else {
		switch rightDType {
		case Float16:
			operation = device.OperationTensorMatMulFloat16
		case BFloat16:
			operation = device.OperationTensorMatMulBFloat16
		default:
			operation = device.OperationTensorMatMul
		}
	}
	return m.EncodeMatMul(left.(*device.Buffer), right.(*device.Buffer), destination.(*device.Buffer),
		rows, inner, rightRows, rightCols, rows, cols, operation)
}

func (m metalCommandBuffer) encodeKVAppend(history, keys, values, target allocation, plane, prefix uint32, cacheDType DType) (err error) {
	var previous *device.Buffer
	if history != nil {
		previous = history.(*device.Buffer)
	}
	if cacheDType == Float16 {
		return m.EncodeKVAppendFloat16(previous, keys.(*device.Buffer), values.(*device.Buffer), target.(*device.Buffer), plane, prefix)
	}
	return m.EncodeKVAppend(previous, keys.(*device.Buffer), values.(*device.Buffer), target.(*device.Buffer), plane, prefix)
}

func (m metalCommandBuffer) encodeAttentionMath(operation device.AttentionMathOperation, first, second, target, extra allocation, params [6]uint32, threads uint32) (err error) {
	var additional *device.Buffer
	if extra != nil {
		additional = extra.(*device.Buffer)
	}
	return m.EncodeAttentionMath(operation, first.(*device.Buffer), second.(*device.Buffer), target.(*device.Buffer), additional, params, threads)
}
