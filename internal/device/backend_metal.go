//go:build darwin && cgo && metal && !purego

package device

/*
#cgo LDFLAGS: -framework Foundation -framework Metal
#include <stdlib.h>
#include "metal_backend.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

type metalBackend struct{}

func newPlatformBackend() (runtimeBackend backend) {
	runtimeBackend = &metalBackend{}
	return runtimeBackend
}

func (m *metalBackend) available() (available bool, err error) {
	var status C.int

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_runtime_available()
	switch status {
	case C.NNMetalStatusSuccess:
		return true, nil
	case C.NNMetalStatusUnavailable:
		return false, nil
	default:
		err = m.lastError("initialize Metal runtime")
		return false, err
	}
}

func (m *metalBackend) maxBufferLength() (bytes uint64) {
	return uint64(C.nn_metal_max_buffer_length())
}

func (m *metalBackend) newBuffer(bytes uint64) (handle any, err error) {
	var buffer C.NNMetalBuffer

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	buffer = C.nn_metal_buffer_new(C.uint64_t(bytes))
	if buffer == nil {
		err = m.lastError("allocate Metal buffer")
		return nil, err
	}

	handle = unsafe.Pointer(buffer)
	return handle, nil
}

func (m *metalBackend) upload(handle any, values []float32) (err error) {
	var (
		buffer C.NNMetalBuffer
		status C.int
	)

	if buffer, err = m.bufferHandle(handle); err != nil {
		return err
	}
	if len(values) == 0 {
		err = errors.New("metal: upload buffer: values are empty")
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_buffer_upload(
		buffer,
		(*C.float)(unsafe.Pointer(&values[0])),
		C.uint64_t(len(values)),
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("upload Metal buffer")
		return err
	}

	return nil
}

func (m *metalBackend) download(handle any, values []float32) (err error) {
	var (
		buffer C.NNMetalBuffer
		status C.int
	)

	if buffer, err = m.bufferHandle(handle); err != nil {
		return err
	}
	if len(values) == 0 {
		err = errors.New("metal: download buffer: destination is empty")
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_buffer_download(
		buffer,
		(*C.float)(unsafe.Pointer(&values[0])),
		C.uint64_t(len(values)),
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("download Metal buffer")
		return err
	}

	return nil
}

func (m *metalBackend) uploadBytes(handle any, values []byte) (err error) {
	var (
		buffer C.NNMetalBuffer
		status C.int
	)

	if buffer, err = m.bufferHandle(handle); err != nil {
		return err
	}
	if len(values) == 0 {
		err = errors.New("metal: upload buffer bytes: values are empty")
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_buffer_upload_bytes(
		buffer,
		unsafe.Pointer(&values[0]),
		C.uint64_t(len(values)),
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("upload Metal buffer bytes")
		return err
	}

	return nil
}

func (m *metalBackend) downloadBytes(handle any, values []byte) (err error) {
	var (
		buffer C.NNMetalBuffer
		status C.int
	)

	if buffer, err = m.bufferHandle(handle); err != nil {
		return err
	}
	if len(values) == 0 {
		err = errors.New("metal: download buffer bytes: destination is empty")
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_buffer_download_bytes(
		buffer,
		unsafe.Pointer(&values[0]),
		C.uint64_t(len(values)),
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("download Metal buffer bytes")
		return err
	}

	return nil
}

func (m *metalBackend) uploadUint16(handle any, values []uint16) (err error) {
	var (
		buffer C.NNMetalBuffer
		status C.int
	)

	if buffer, err = m.bufferHandle(handle); err != nil {
		return err
	}
	if len(values) == 0 {
		err = errors.New("metal: upload uint16 buffer: values are empty")
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_buffer_upload_bytes(
		buffer,
		unsafe.Pointer(&values[0]),
		C.uint64_t(len(values))*2,
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("upload Metal uint16 buffer")
		return err
	}

	return nil
}

func (m *metalBackend) downloadUint16(handle any, values []uint16) (err error) {
	var (
		buffer C.NNMetalBuffer
		status C.int
	)

	if buffer, err = m.bufferHandle(handle); err != nil {
		return err
	}
	if len(values) == 0 {
		err = errors.New("metal: download uint16 buffer: destination is empty")
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_buffer_download_bytes(
		buffer,
		unsafe.Pointer(&values[0]),
		C.uint64_t(len(values))*2,
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("download Metal uint16 buffer")
		return err
	}

	return nil
}

func (m *metalBackend) releaseBuffer(handle any) {
	var (
		buffer C.NNMetalBuffer
		err    error
	)

	if buffer, err = m.bufferHandle(handle); err != nil {
		return
	}

	C.nn_metal_buffer_release(buffer)
}

func (m *metalBackend) newScope() (handle any, err error) {
	var scope C.NNMetalScope

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	scope = C.nn_metal_scope_new()
	if scope == nil {
		err = m.lastError("create Metal command scope")
		return nil, err
	}

	handle = unsafe.Pointer(scope)
	return handle, nil
}

func (m *metalBackend) encodeCopy(scope, source, destination any, bytes uint64) (err error) {
	var (
		scopeHandle       C.NNMetalScope
		sourceHandle      C.NNMetalBuffer
		destinationHandle C.NNMetalBuffer
		status            C.int
	)

	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if sourceHandle, err = m.bufferHandle(source); err != nil {
		return err
	}
	if destinationHandle, err = m.bufferHandle(destination); err != nil {
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_encode_copy(
		scopeHandle,
		sourceHandle,
		destinationHandle,
		C.uint64_t(bytes),
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("encode Metal buffer copy")
		return err
	}

	return nil
}

func (m *metalBackend) encodeGatherRows(scope, source, destination any, rowBytes uint64, rows []int64) (err error) {
	var (
		scopeHandle       C.NNMetalScope
		sourceHandle      C.NNMetalBuffer
		destinationHandle C.NNMetalBuffer
	)
	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if sourceHandle, err = m.bufferHandle(source); err != nil {
		return err
	}
	if destinationHandle, err = m.bufferHandle(destination); err != nil {
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var status = C.nn_metal_scope_encode_gather_rows(scopeHandle, sourceHandle, destinationHandle,
		C.uint64_t(rowBytes), (*C.int64_t)(unsafe.Pointer(unsafe.SliceData(rows))), C.uint64_t(len(rows)))
	if status != C.NNMetalStatusSuccess {
		return m.lastError("encode Metal row gather")
	}
	return nil
}

func (m *metalBackend) encodeGatherRowsMixed(scope, source, destination any, columns uint64, rows []int64, sourceDType uint32) (err error) {
	var (
		scopeHandle       C.NNMetalScope
		sourceHandle      C.NNMetalBuffer
		destinationHandle C.NNMetalBuffer
	)
	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if sourceHandle, err = m.bufferHandle(source); err != nil {
		return err
	}
	if destinationHandle, err = m.bufferHandle(destination); err != nil {
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var status = C.nn_metal_scope_encode_gather_rows_mixed(scopeHandle, sourceHandle, destinationHandle,
		C.uint64_t(columns), (*C.int64_t)(unsafe.Pointer(unsafe.SliceData(rows))), C.uint64_t(len(rows)), C.uint32_t(sourceDType))
	if status != C.NNMetalStatusSuccess {
		return m.lastError("encode Metal mixed row gather")
	}
	return nil
}

func (m *metalBackend) encodeFill(scope, buffer any, value float32, count uint64) (err error) {
	var (
		scopeHandle  C.NNMetalScope
		bufferHandle C.NNMetalBuffer
		status       C.int
	)

	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if bufferHandle, err = m.bufferHandle(buffer); err != nil {
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_encode_fill(
		scopeHandle,
		bufferHandle,
		C.float(value),
		C.uint64_t(count),
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("encode Metal buffer fill")
		return err
	}

	return nil
}

func (m *metalBackend) encodeElementwise(scope, left, right, result any, count uint32, operation ElementwiseOperation) (err error) {
	var (
		scopeHandle  C.NNMetalScope
		leftHandle   C.NNMetalBuffer
		rightHandle  C.NNMetalBuffer
		resultHandle C.NNMetalBuffer
		status       C.int
	)
	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if leftHandle, err = m.bufferHandle(left); err != nil {
		return err
	}
	if right != nil {
		if rightHandle, err = m.bufferHandle(right); err != nil {
			return err
		}
	}
	if resultHandle, err = m.bufferHandle(result); err != nil {
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_encode_tensor_elementwise(scopeHandle, leftHandle, rightHandle, resultHandle,
		C.uint32_t(count), C.uint32_t(operation))
	if status != C.NNMetalStatusSuccess {
		return m.lastError("encode Metal tensor elementwise operation")
	}
	return nil
}

func (m *metalBackend) encodeKVAppend(scope, history, keys, values, target any, plane, prefix uint32) (err error) {
	var scopeHandle C.NNMetalScope
	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	var handles [4]C.NNMetalBuffer
	for index, buffer := range [4]any{history, keys, values, target} {
		if index == 0 && history == nil {
			continue
		}
		if handles[index], err = m.bufferHandle(buffer); err != nil {
			return err
		}
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var status = C.nn_metal_scope_encode_kv_append(scopeHandle, handles[0], handles[1], handles[2], handles[3],
		C.uint32_t(plane), C.uint32_t(prefix))
	if status != C.NNMetalStatusSuccess {
		return m.lastError("encode Metal KV append")
	}
	return nil
}

func (m *metalBackend) encodeKVAppendFloat16(scope, history, keys, values, target any, plane, prefix uint32) (err error) {
	var scopeHandle C.NNMetalScope
	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	var handles [4]C.NNMetalBuffer
	for index, buffer := range [4]any{history, keys, values, target} {
		if index == 0 && history == nil {
			continue
		}
		if handles[index], err = m.bufferHandle(buffer); err != nil {
			return err
		}
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var status = C.nn_metal_scope_encode_kv_append_f16(scopeHandle, handles[0], handles[1], handles[2], handles[3],
		C.uint32_t(plane), C.uint32_t(prefix))
	if status != C.NNMetalStatusSuccess {
		return m.lastError("encode Metal FP16 KV append")
	}
	return nil
}

func (m *metalBackend) encodeAttentionMath(scope any, operation AttentionMathOperation, buffers [4]any, params [6]uint32, threads uint32) (err error) {
	var scopeHandle C.NNMetalScope
	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	var handles [4]C.NNMetalBuffer
	for index, buffer := range buffers {
		if index == 3 && buffer == nil {
			break
		}
		if handles[index], err = m.bufferHandle(buffer); err != nil {
			return err
		}
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var status = C.nn_metal_scope_encode_attention_math(scopeHandle, C.uint32_t(operation), handles[0], handles[1], handles[2], handles[3],
		(*C.uint32_t)(unsafe.Pointer(&params[0])), C.uint32_t(threads))
	if status != C.NNMetalStatusSuccess {
		return m.lastError("encode Metal attention math")
	}
	return nil
}

func (m *metalBackend) encodeAddRowVector(
	scope,
	values,
	rowVector any,
	rows,
	cols uint32,
) (err error) {
	var (
		scopeHandle     C.NNMetalScope
		valuesHandle    C.NNMetalBuffer
		rowVectorHandle C.NNMetalBuffer
		status          C.int
	)

	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if valuesHandle, err = m.bufferHandle(values); err != nil {
		return err
	}
	if rowVectorHandle, err = m.bufferHandle(rowVector); err != nil {
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_encode_add_row_vector(
		scopeHandle,
		valuesHandle,
		rowVectorHandle,
		C.uint32_t(rows),
		C.uint32_t(cols),
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("encode Metal row-vector addition")
		return err
	}

	return nil
}

func (m *metalBackend) encodeAddScaled(
	scope,
	left,
	right,
	result any,
	scale float32,
	count uint32,
) (err error) {
	var (
		scopeHandle  C.NNMetalScope
		leftHandle   C.NNMetalBuffer
		rightHandle  C.NNMetalBuffer
		resultHandle C.NNMetalBuffer
		status       C.int
	)

	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if leftHandle, err = m.bufferHandle(left); err != nil {
		return err
	}
	if rightHandle, err = m.bufferHandle(right); err != nil {
		return err
	}
	if resultHandle, err = m.bufferHandle(result); err != nil {
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_encode_add_scaled(
		scopeHandle,
		leftHandle,
		rightHandle,
		resultHandle,
		C.float(scale),
		C.uint32_t(count),
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("encode Metal scaled addition")
		return err
	}

	return nil
}

func (m *metalBackend) encodeReLU(scope, input, result any, count uint32) (err error) {
	var (
		scopeHandle  C.NNMetalScope
		inputHandle  C.NNMetalBuffer
		resultHandle C.NNMetalBuffer
		status       C.int
	)

	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if inputHandle, err = m.bufferHandle(input); err != nil {
		return err
	}
	if resultHandle, err = m.bufferHandle(result); err != nil {
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_encode_relu(
		scopeHandle,
		inputHandle,
		resultHandle,
		C.uint32_t(count),
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("encode Metal ReLU")
		return err
	}

	return nil
}

func (m *metalBackend) encodeReLUBackward(
	scope,
	input,
	outputGradient,
	result any,
	count uint32,
) (err error) {
	var (
		scopeHandle          C.NNMetalScope
		inputHandle          C.NNMetalBuffer
		outputGradientHandle C.NNMetalBuffer
		resultHandle         C.NNMetalBuffer
		status               C.int
	)

	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if inputHandle, err = m.bufferHandle(input); err != nil {
		return err
	}
	if outputGradientHandle, err = m.bufferHandle(outputGradient); err != nil {
		return err
	}
	if resultHandle, err = m.bufferHandle(result); err != nil {
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_encode_relu_backward(
		scopeHandle,
		inputHandle,
		outputGradientHandle,
		resultHandle,
		C.uint32_t(count),
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("encode Metal ReLU backward")
		return err
	}

	return nil
}

func (m *metalBackend) encodeSoftmaxRows(
	scope,
	input,
	result any,
	rows,
	cols uint32,
) (err error) {
	var (
		scopeHandle  C.NNMetalScope
		inputHandle  C.NNMetalBuffer
		resultHandle C.NNMetalBuffer
		status       C.int
	)

	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if inputHandle, err = m.bufferHandle(input); err != nil {
		return err
	}
	if resultHandle, err = m.bufferHandle(result); err != nil {
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_encode_softmax_rows(
		scopeHandle,
		inputHandle,
		resultHandle,
		C.uint32_t(rows),
		C.uint32_t(cols),
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("encode Metal row-wise Softmax")
		return err
	}

	return nil
}

func (m *metalBackend) encodeSoftmaxRowsBackward(
	scope,
	input,
	outputGradient,
	result any,
	rows,
	cols uint32,
) (err error) {
	var (
		scopeHandle          C.NNMetalScope
		inputHandle          C.NNMetalBuffer
		outputGradientHandle C.NNMetalBuffer
		resultHandle         C.NNMetalBuffer
		status               C.int
	)

	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if inputHandle, err = m.bufferHandle(input); err != nil {
		return err
	}
	if outputGradientHandle, err = m.bufferHandle(outputGradient); err != nil {
		return err
	}
	if resultHandle, err = m.bufferHandle(result); err != nil {
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_encode_softmax_rows_backward(
		scopeHandle,
		inputHandle,
		outputGradientHandle,
		resultHandle,
		C.uint32_t(rows),
		C.uint32_t(cols),
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("encode Metal row-wise Softmax backward")
		return err
	}

	return nil
}

func (m *metalBackend) encodeColumnSums(
	scope,
	input,
	result any,
	rows,
	cols uint32,
	accumulate bool,
) (err error) {
	var (
		scopeHandle     C.NNMetalScope
		inputHandle     C.NNMetalBuffer
		resultHandle    C.NNMetalBuffer
		accumulateValue C.uint32_t
		status          C.int
	)

	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if inputHandle, err = m.bufferHandle(input); err != nil {
		return err
	}
	if resultHandle, err = m.bufferHandle(result); err != nil {
		return err
	}
	if accumulate {
		accumulateValue = 1
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_encode_column_sums(
		scopeHandle,
		inputHandle,
		resultHandle,
		C.uint32_t(rows),
		C.uint32_t(cols),
		accumulateValue,
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("encode Metal column sums")
		return err
	}

	return nil
}

func (m *metalBackend) encodeCategoricalCrossEntropy(
	scope,
	predictions,
	targets,
	result any,
	rows,
	cols uint32,
	epsilon float32,
) (err error) {
	var (
		scopeHandle      C.NNMetalScope
		predictionHandle C.NNMetalBuffer
		targetHandle     C.NNMetalBuffer
		resultHandle     C.NNMetalBuffer
		status           C.int
	)

	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if predictionHandle, err = m.bufferHandle(predictions); err != nil {
		return err
	}
	if targetHandle, err = m.bufferHandle(targets); err != nil {
		return err
	}
	if resultHandle, err = m.bufferHandle(result); err != nil {
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_encode_categorical_cross_entropy(
		scopeHandle,
		predictionHandle,
		targetHandle,
		resultHandle,
		C.uint32_t(rows),
		C.uint32_t(cols),
		C.float(epsilon),
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("encode Metal categorical cross entropy")
		return err
	}

	return nil
}

func (m *metalBackend) encodeCategoricalCrossEntropyGradient(
	scope,
	predictions,
	targets,
	result any,
	rows,
	cols uint32,
	epsilon float32,
) (err error) {
	var (
		scopeHandle      C.NNMetalScope
		predictionHandle C.NNMetalBuffer
		targetHandle     C.NNMetalBuffer
		resultHandle     C.NNMetalBuffer
		status           C.int
	)

	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if predictionHandle, err = m.bufferHandle(predictions); err != nil {
		return err
	}
	if targetHandle, err = m.bufferHandle(targets); err != nil {
		return err
	}
	if resultHandle, err = m.bufferHandle(result); err != nil {
		return err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_encode_categorical_cross_entropy_gradient(
		scopeHandle,
		predictionHandle,
		targetHandle,
		resultHandle,
		C.uint32_t(rows),
		C.uint32_t(cols),
		C.float(epsilon),
	)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("encode Metal categorical cross entropy gradient")
		return err
	}

	return nil
}

func (m *metalBackend) encodeMatMul(
	scope,
	left,
	right,
	result any,
	dimensions matMulDimensions,
	operation Operation,
) (err error) {
	var (
		scopeHandle     C.NNMetalScope
		leftHandle      C.NNMetalBuffer
		rightHandle     C.NNMetalBuffer
		resultHandle    C.NNMetalBuffer
		metalDimensions C.NNMetalMatMulDimensions
		tensorPipeline  C.int
		weightDType     C.uint32_t
		status          C.int
	)

	if scopeHandle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	if leftHandle, err = m.bufferHandle(left); err != nil {
		return err
	}
	if rightHandle, err = m.bufferHandle(right); err != nil {
		return err
	}
	if resultHandle, err = m.bufferHandle(result); err != nil {
		return err
	}

	metalDimensions.leftRows = C.uint32_t(dimensions.leftRows)
	metalDimensions.leftCols = C.uint32_t(dimensions.leftCols)
	metalDimensions.rightRows = C.uint32_t(dimensions.rightRows)
	metalDimensions.rightCols = C.uint32_t(dimensions.rightCols)
	metalDimensions.resultRows = C.uint32_t(dimensions.resultRows)
	metalDimensions.resultCols = C.uint32_t(dimensions.resultCols)
	switch operation {
	case OperationTensorMatMul:
		tensorPipeline = 1
		metalDimensions.variant = C.uint32_t(OperationMatMul)
	case OperationTensorMatMulRightTranspose:
		tensorPipeline = 1
		metalDimensions.variant = C.uint32_t(OperationMatMulRightTranspose)
	case OperationTensorMatMulFloat16:
		weightDType = C.NNMetalWeightFloat16
		metalDimensions.variant = C.uint32_t(OperationMatMul)
	case OperationTensorMatMulRightTransposeFloat16:
		weightDType = C.NNMetalWeightFloat16
		metalDimensions.variant = C.uint32_t(OperationMatMulRightTranspose)
	case OperationTensorMatMulBFloat16:
		weightDType = C.NNMetalWeightBFloat16
		metalDimensions.variant = C.uint32_t(OperationMatMul)
	case OperationTensorMatMulRightTransposeBFloat16:
		weightDType = C.NNMetalWeightBFloat16
		metalDimensions.variant = C.uint32_t(OperationMatMulRightTranspose)
	case OperationTensorMatMulRightTransposeQ8_0:
		weightDType = C.NNMetalWeightQ8_0
		metalDimensions.variant = C.uint32_t(OperationMatMulRightTranspose)
	case OperationTensorMatMulRightTransposeQ4_0:
		weightDType = C.NNMetalWeightQ4_0
		metalDimensions.variant = C.uint32_t(OperationMatMulRightTranspose)
	default:
		metalDimensions.variant = C.uint32_t(operation)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if weightDType == C.NNMetalWeightQ8_0 || weightDType == C.NNMetalWeightQ4_0 {
		status = C.nn_metal_scope_encode_matmul_quant(
			scopeHandle,
			leftHandle,
			rightHandle,
			resultHandle,
			metalDimensions,
			weightDType,
		)
	} else if weightDType != 0 {
		status = C.nn_metal_scope_encode_matmul_mixed(
			scopeHandle,
			leftHandle,
			rightHandle,
			resultHandle,
			metalDimensions,
			weightDType,
		)
	} else {
		status = C.nn_metal_scope_encode_matmul(
			scopeHandle,
			leftHandle,
			rightHandle,
			resultHandle,
			metalDimensions,
			tensorPipeline,
		)
	}
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("encode Metal matrix multiplication")
		return err
	}

	return nil
}

func (m *metalBackend) commit(scope any) (err error) {
	var (
		handle C.NNMetalScope
		status C.int
	)

	if handle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_commit(handle)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("commit Metal command scope")
		return err
	}

	return nil
}

func (m *metalBackend) completed(scope any) (complete bool, err error) {
	var (
		handle C.NNMetalScope
		status C.int
	)

	if handle, err = m.scopeHandle(scope); err != nil {
		return false, err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_completed(handle)
	switch status {
	case C.NNMetalStatusSuccess:
		return true, nil
	case C.NNMetalStatusUnavailable:
		return false, nil
	default:
		err = m.lastError("inspect Metal command completion")
		return false, err
	}
}

func (m *metalBackend) wait(scope any) (err error) {
	var (
		handle C.NNMetalScope
		status C.int
	)

	if handle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_scope_wait(handle)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("wait for Metal command scope")
		return err
	}

	return nil
}

func (m *metalBackend) releaseScope(scope any) (err error) {
	var (
		handle C.NNMetalScope
	)

	if handle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	C.nn_metal_scope_release(handle)
	return nil
}

func (m *metalBackend) resourceSnapshot() (snapshot ResourceSnapshot) {
	var value C.NNMetalResourceSnapshot

	C.nn_metal_resource_snapshot(&value)
	snapshot.LiveBuffers = uint64(value.liveBuffers)
	snapshot.LiveBufferBytes = uint64(value.liveBufferBytes)
	snapshot.PeakBuffers = uint64(value.peakBuffers)
	snapshot.PeakBufferBytes = uint64(value.peakBufferBytes)
	snapshot.LiveScopes = uint64(value.liveScopes)
	snapshot.PeakScopes = uint64(value.peakScopes)
	snapshot.CreatedBuffers = uint64(value.createdBuffers)
	snapshot.ReleasedBuffers = uint64(value.releasedBuffers)
	snapshot.CreatedScopes = uint64(value.createdScopes)
	snapshot.ReleasedScopes = uint64(value.releasedScopes)
	snapshot.SubmittedCommands = uint64(value.submittedCommands)
	snapshot.CompletedCommands = uint64(value.completedCommands)
	return snapshot
}

func (m *metalBackend) resetResourcePeaks() (err error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if C.nn_metal_resource_reset() != C.NNMetalStatusSuccess {
		err = m.lastError("reset Metal resource counters")
		return err
	}

	return nil
}

func (m *metalBackend) testMissingKernel(name string) (err error) {
	var (
		nameValue *C.char
		status    C.int
	)

	nameValue = C.CString(name)
	defer C.free(unsafe.Pointer(nameValue))
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_test_missing_kernel(nameValue)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("create Metal test pipeline")
		return err
	}

	return nil
}

func (m *metalBackend) testCompileSource(source string) (err error) {
	var (
		sourceValue *C.char
		status      C.int
	)

	sourceValue = C.CString(source)
	defer C.free(unsafe.Pointer(sourceValue))
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_test_compile_source(sourceValue)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("compile Metal test shader")
		return err
	}

	return nil
}

func (m *metalBackend) testFailScope(scope any) (err error) {
	var (
		handle C.NNMetalScope
		status C.int
	)

	if handle, err = m.scopeHandle(scope); err != nil {
		return err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status = C.nn_metal_test_scope_fail(handle)
	if status != C.NNMetalStatusSuccess {
		err = m.lastError("inject Metal command failure")
		return err
	}

	return nil
}

func (m *metalBackend) bufferHandle(handle any) (buffer C.NNMetalBuffer, err error) {
	var (
		pointer unsafe.Pointer
		ok      bool
	)

	if pointer, ok = handle.(unsafe.Pointer); !ok || pointer == nil {
		err = errors.New("metal: buffer handle is nil or invalid")
		return nil, err
	}

	buffer = C.NNMetalBuffer(pointer)
	return buffer, nil
}

func (m *metalBackend) scopeHandle(handle any) (scope C.NNMetalScope, err error) {
	var (
		pointer unsafe.Pointer
		ok      bool
	)

	if pointer, ok = handle.(unsafe.Pointer); !ok || pointer == nil {
		err = errors.New("metal: command scope handle is nil or invalid")
		return nil, err
	}

	scope = C.NNMetalScope(pointer)
	return scope, nil
}

// lastError must run on the same OS thread as the failed native call.
func (m *metalBackend) lastError(operation string) (err error) {
	var message string

	message = C.GoString(C.nn_metal_last_error())
	if message == "" {
		message = "unknown Metal error"
	}
	err = fmt.Errorf("%s: %s", operation, message)
	return err
}
