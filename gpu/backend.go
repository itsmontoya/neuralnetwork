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
	encodeScale(input, destination allocation, scale float32) (err error)
	encodeBroadcast(left, right, destination allocation, operation device.BroadcastOperation, spec device.BroadcastSpec) (err error)
	encodeSoftmax(input, destination allocation, outer, axisSize, inner uint32) (err error)
	encodeLayerNorm(input, weight, bias, destination allocation, rows, width uint32, epsilon float32, weightDType, biasDType DType) (err error)
	encodeBatchNorm(input, scale, bias, mean, variance, destination allocation, batch, channels, spatial uint32, epsilon float32) (err error)
	encodePool2D(input, destination allocation, spec device.Pool2DSpec) (err error)
	encodeConv2D(input, weights, bias, destination allocation, spec device.Conv2DSpec) (err error)
	encodeTranspose(input, destination allocation, spec device.TransposeSpec) (err error)
	encodeMatMulLeftTranspose(left, right, destination allocation, m, n, k uint32, rightDType DType) (err error)
	encodeBatchedMatMul(left, right, destination allocation, spec device.BatchedMatMulSpec) (err error)
	encodeKVAppend(keys, values, target allocation, plane, prefix uint32, cacheDType DType) (err error)
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
		return m.EncodeGatherRows(source.(*device.Buffer), destination.(*device.Buffer), columns/sourceDType.BlockSize(), sourceDType.BlockBytes(), rows)
	}
	var nativeDType uint32
	switch sourceDType {
	case Float16:
		nativeDType = 1
	case BFloat16:
		nativeDType = 2
	case Q8_0:
		nativeDType = 3
	case Q4_0:
		nativeDType = 4
	case Q4_1:
		nativeDType = 5
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

func (m metalCommandBuffer) encodeScale(input, destination allocation, scale float32) (err error) {
	return m.EncodeTensorScale(input.(*device.Buffer), destination.(*device.Buffer), scale)
}

func (m metalCommandBuffer) encodeBroadcast(left, right, destination allocation, operation device.BroadcastOperation, spec device.BroadcastSpec) (err error) {
	return m.EncodeTensorBroadcast(left.(*device.Buffer), right.(*device.Buffer), destination.(*device.Buffer), operation, spec)
}

func (m metalCommandBuffer) encodeSoftmax(input, destination allocation, outer, axisSize, inner uint32) (err error) {
	return m.EncodeTensorSoftmax(input.(*device.Buffer), destination.(*device.Buffer), outer, axisSize, inner)
}

func (m metalCommandBuffer) encodeLayerNorm(input, weight, bias, destination allocation, rows, width uint32, epsilon float32, weightDType, biasDType DType) (err error) {
	var biasBuffer *device.Buffer
	if bias != nil {
		biasBuffer = bias.(*device.Buffer)
	}
	return m.EncodeTensorLayerNorm(
		input.(*device.Buffer),
		weight.(*device.Buffer),
		biasBuffer,
		destination.(*device.Buffer),
		rows,
		width,
		epsilon,
		layerNormDTypeTag(weightDType),
		layerNormDTypeTag(biasDType),
	)
}

func layerNormDTypeTag(dtype DType) (tag uint32) {
	switch dtype {
	case Float16:
		return 1
	case BFloat16:
		return 2
	default:
		return 0
	}
}

func (m metalCommandBuffer) encodeBatchNorm(input, scale, bias, mean, variance, destination allocation, batch, channels, spatial uint32, epsilon float32) (err error) {
	return m.EncodeTensorBatchNorm(
		input.(*device.Buffer),
		scale.(*device.Buffer),
		bias.(*device.Buffer),
		mean.(*device.Buffer),
		variance.(*device.Buffer),
		destination.(*device.Buffer),
		batch,
		channels,
		spatial,
		epsilon,
	)
}

func (m metalCommandBuffer) encodePool2D(input, destination allocation, spec device.Pool2DSpec) (err error) {
	return m.EncodeTensorPool2D(input.(*device.Buffer), destination.(*device.Buffer), spec)
}

func (m metalCommandBuffer) encodeConv2D(input, weights, bias, destination allocation, spec device.Conv2DSpec) (err error) {
	var biasBuffer *device.Buffer
	if bias != nil {
		biasBuffer = bias.(*device.Buffer)
	}
	return m.EncodeTensorConv2D(
		input.(*device.Buffer),
		weights.(*device.Buffer),
		biasBuffer,
		destination.(*device.Buffer),
		spec,
	)
}

func (m metalCommandBuffer) encodeTranspose(input, destination allocation, spec device.TransposeSpec) (err error) {
	return m.EncodeTensorTranspose(input.(*device.Buffer), destination.(*device.Buffer), spec)
}

func (m metalCommandBuffer) encodeMatMulLeftTranspose(left, right, destination allocation, rows, cols, inner uint32, rightDType DType) (err error) {
	return m.EncodeTensorMatMulLeftTranspose(
		left.(*device.Buffer),
		right.(*device.Buffer),
		destination.(*device.Buffer),
		rows,
		cols,
		inner,
		layerNormDTypeTag(rightDType),
	)
}

func (m metalCommandBuffer) encodeBatchedMatMul(left, right, destination allocation, spec device.BatchedMatMulSpec) (err error) {
	return m.EncodeTensorBatchedMatMul(left.(*device.Buffer), right.(*device.Buffer), destination.(*device.Buffer), spec)
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
		case Q4_1:
			operation = device.OperationTensorMatMulRightTransposeQ4_1
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

func (m metalCommandBuffer) encodeKVAppend(keys, values, target allocation, plane, prefix uint32, cacheDType DType) (err error) {
	if cacheDType == Float16 {
		return m.EncodeKVAppendFloat16(keys.(*device.Buffer), values.(*device.Buffer), target.(*device.Buffer), plane, prefix)
	}
	return m.EncodeKVAppend(keys.(*device.Buffer), values.(*device.Buffer), target.(*device.Buffer), plane, prefix)
}

func (m metalCommandBuffer) encodeAttentionMath(operation device.AttentionMathOperation, first, second, target, extra allocation, params [6]uint32, threads uint32) (err error) {
	var additional *device.Buffer
	if extra != nil {
		additional = extra.(*device.Buffer)
	}
	return m.EncodeAttentionMath(operation, first.(*device.Buffer), second.(*device.Buffer), target.(*device.Buffer), additional, params, threads)
}
