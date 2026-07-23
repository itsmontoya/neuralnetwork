//go:build darwin && cgo && metal && !purego

package matrix

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/device"
)

func addRowVectorInPlaceDevice(
	values,
	rowVector *Matrix,
) (handled bool, err error) {
	var (
		execution       *device.Execution
		valuesBuffer    *device.Buffer
		rowVectorBuffer *device.Buffer
		pending         bool
		allocated       bool
		uploaded        bool
	)

	if execution, err = compatibleExecution(values, rowVector); err != nil {
		return false, err
	}
	if execution == nil || !execution.Activated() || !metalElementwiseSupported(values) {
		return false, nil
	}
	if values.residency != nil {
		valuesBuffer, pending = values.residency.PendingBuffer(execution)
	}
	if !pending {
		return false, nil
	}

	if err = execution.Bind(rowVector); err != nil {
		return false, fmt.Errorf("matrix: bind Metal row vector: %w", err)
	}
	if rowVectorBuffer, allocated, uploaded, err = rowVector.ensureExecutionDeviceBuffer(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal row vector: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(rowVector.data))*4)
	if err = execution.EncodeAddRowVector(
		valuesBuffer,
		rowVectorBuffer,
		uint32(values.rows),
		uint32(values.cols),
	); err != nil {
		return false, fmt.Errorf("matrix: encode Metal row-vector addition: %w", err)
	}
	if err = execution.MarkRead(rowVector); err != nil {
		return false, fmt.Errorf("matrix: record Metal row-vector use: %w", err)
	}

	return true, nil
}

func reluForwardDevice(input, result *Matrix) (handled bool, err error) {
	var (
		execution    *device.Execution
		inputBuffer  *device.Buffer
		resultBuffer *device.Buffer
		publication  device.Publication
		allocated    bool
		uploaded     bool
	)

	if execution, err = compatibleExecution(input, result); err != nil {
		return false, err
	}
	if execution == nil || !execution.Activated() || !metalElementwiseSupported(input) {
		return false, nil
	}
	if err = execution.Bind(input); err != nil {
		return false, fmt.Errorf("matrix: bind Metal ReLU input: %w", err)
	}
	if err = execution.Bind(result); err != nil {
		return false, fmt.Errorf("matrix: bind Metal ReLU destination: %w", err)
	}
	if inputBuffer, allocated, uploaded, err = input.ensureExecutionDeviceBuffer(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal ReLU input: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(input.data))*4)
	if resultBuffer, allocated, err = result.beginExecutionDeviceWrite(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal ReLU destination: %w", err)
	}
	execution.RecordDevicePreparation(allocated, false, 0)
	publication.Publish = func() (publishErr error) {
		publishErr = result.publishDeviceWrite(resultBuffer)
		return publishErr
	}
	publication.Discard = func(cause error) (discardErr error) {
		discardErr = result.failDeviceWrite(resultBuffer, cause)
		return discardErr
	}
	if err = execution.EncodeReLU(
		inputBuffer,
		resultBuffer,
		uint64(len(result.data))*4,
		publication,
	); err != nil {
		return false, fmt.Errorf("matrix: encode Metal ReLU: %w", err)
	}
	if err = execution.MarkRead(input); err != nil {
		return false, fmt.Errorf("matrix: record Metal ReLU input use: %w", err)
	}

	return true, nil
}

func softmaxRowsIntoDevice(input, result *Matrix) (handled bool, err error) {
	var (
		execution    *device.Execution
		inputBuffer  *device.Buffer
		resultBuffer *device.Buffer
		publication  device.Publication
		allocated    bool
		uploaded     bool
	)

	if execution, err = compatibleExecution(input, result); err != nil {
		return false, err
	}
	if execution == nil || !execution.Activated() || !metalElementwiseSupported(input) {
		return false, nil
	}
	if err = execution.Bind(input); err != nil {
		return false, fmt.Errorf("matrix: bind Metal Softmax input: %w", err)
	}
	if err = execution.Bind(result); err != nil {
		return false, fmt.Errorf("matrix: bind Metal Softmax destination: %w", err)
	}
	if inputBuffer, allocated, uploaded, err = input.ensureExecutionDeviceBuffer(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal Softmax input: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(input.data))*4)
	if resultBuffer, allocated, err = result.beginExecutionDeviceWrite(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal Softmax destination: %w", err)
	}
	execution.RecordDevicePreparation(allocated, false, 0)
	publication.Publish = func() (publishErr error) {
		publishErr = result.publishDeviceWrite(resultBuffer)
		return publishErr
	}
	publication.Discard = func(cause error) (discardErr error) {
		discardErr = result.failDeviceWrite(resultBuffer, cause)
		return discardErr
	}
	if err = execution.EncodeSoftmaxRows(
		inputBuffer,
		resultBuffer,
		uint32(input.rows),
		uint32(input.cols),
		uint64(len(result.data))*4,
		publication,
	); err != nil {
		return false, fmt.Errorf("matrix: encode Metal Softmax: %w", err)
	}
	if err = execution.MarkRead(input); err != nil {
		return false, fmt.Errorf("matrix: record Metal Softmax input use: %w", err)
	}

	return true, nil
}

func metalElementwiseSupported(value *Matrix) (ok bool) {
	if value == nil || len(value.data) == 0 {
		return false
	}
	if !metalDimensionSupported(value.rows) || !metalDimensionSupported(value.cols) {
		return false
	}
	if uint64(len(value.data)) > maxMetalUint32 {
		return false
	}

	return true
}
