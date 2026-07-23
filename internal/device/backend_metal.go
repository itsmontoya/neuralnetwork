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
	"unsafe"
)

type metalBackend struct{}

func newPlatformBackend() (runtimeBackend backend) {
	runtimeBackend = &metalBackend{}
	return runtimeBackend
}

func (m *metalBackend) available() (available bool, err error) {
	var status C.int

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

func (m *metalBackend) newBuffer(bytes uint64) (handle any, err error) {
	var buffer C.NNMetalBuffer

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
	metalDimensions.variant = C.uint32_t(operation)
	status = C.nn_metal_scope_encode_matmul(
		scopeHandle,
		leftHandle,
		rightHandle,
		resultHandle,
		metalDimensions,
	)
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

func (m *metalBackend) lastError(operation string) (err error) {
	var message string

	message = C.GoString(C.nn_metal_last_error())
	if message == "" {
		message = "unknown Metal error"
	}
	err = fmt.Errorf("%s: %s", operation, message)
	return err
}
