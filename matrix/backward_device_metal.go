//go:build darwin && cgo && metal && !purego

package matrix

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/device"
)

func addIntoDevice(left, right, result *Matrix) (handled bool, err error) {
	var (
		execution    *device.Execution
		leftBuffer   *device.Buffer
		rightBuffer  *device.Buffer
		resultBuffer *device.Buffer
		allocated    bool
		uploaded     bool
	)

	if execution, err = compatibleExecution(left, right, result); err != nil {
		return false, err
	}
	if execution == nil || !execution.Activated() || !metalElementwiseSupported(left) {
		return false, nil
	}
	if err = execution.Bind(left); err != nil {
		return false, fmt.Errorf("matrix: bind Metal addition left input: %w", err)
	}
	if err = execution.Bind(right); err != nil {
		return false, fmt.Errorf("matrix: bind Metal addition right input: %w", err)
	}
	if err = execution.Bind(result); err != nil {
		return false, fmt.Errorf("matrix: bind Metal addition destination: %w", err)
	}
	if leftBuffer, allocated, uploaded, err = left.ensureExecutionDeviceBuffer(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal addition left input: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(left.data))*4)
	if rightBuffer, allocated, uploaded, err = right.ensureExecutionDeviceBuffer(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal addition right input: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(right.data))*4)
	if resultBuffer, allocated, err = result.beginExecutionDeviceWrite(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal addition destination: %w", err)
	}
	execution.RecordDevicePreparation(allocated, false, 0)
	if err = execution.EncodeAddScaled(
		leftBuffer,
		rightBuffer,
		resultBuffer,
		1,
		uint64(len(result.data))*4,
		deviceWritePublication(result, resultBuffer),
	); err != nil {
		return false, fmt.Errorf("matrix: encode Metal addition: %w", err)
	}
	if err = execution.MarkRead(left); err != nil {
		return false, fmt.Errorf("matrix: record Metal addition left input use: %w", err)
	}
	if err = execution.MarkRead(right); err != nil {
		return false, fmt.Errorf("matrix: record Metal addition right input use: %w", err)
	}

	return true, nil
}

func reluBackwardDevice(
	input,
	outputGradient,
	inputGradient *Matrix,
) (handled bool, err error) {
	var (
		execution            *device.Execution
		inputBuffer          *device.Buffer
		outputGradientBuffer *device.Buffer
		inputGradientBuffer  *device.Buffer
		allocated            bool
		uploaded             bool
	)

	if execution, err = compatibleExecution(input, outputGradient, inputGradient); err != nil {
		return false, err
	}
	if !metalBackwardElementwiseSupported(execution, input) {
		return false, nil
	}
	if err = execution.Bind(input); err != nil {
		return false, fmt.Errorf("matrix: bind Metal ReLU backward input: %w", err)
	}
	if err = execution.Bind(outputGradient); err != nil {
		return false, fmt.Errorf("matrix: bind Metal ReLU backward output gradient: %w", err)
	}
	if err = execution.Bind(inputGradient); err != nil {
		return false, fmt.Errorf("matrix: bind Metal ReLU backward destination: %w", err)
	}
	if inputBuffer, allocated, uploaded, err = input.ensureExecutionDeviceBuffer(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal ReLU backward input: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(input.data))*4)
	if outputGradientBuffer, allocated, uploaded, err =
		outputGradient.ensureExecutionDeviceBuffer(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal ReLU backward output gradient: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(outputGradient.data))*4)
	if inputGradientBuffer, allocated, err = inputGradient.beginExecutionDeviceWrite(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal ReLU backward destination: %w", err)
	}
	execution.RecordDevicePreparation(allocated, false, 0)
	if err = execution.EncodeReLUBackward(
		inputBuffer,
		outputGradientBuffer,
		inputGradientBuffer,
		uint64(len(inputGradient.data))*4,
		deviceWritePublication(inputGradient, inputGradientBuffer),
	); err != nil {
		return false, fmt.Errorf("matrix: encode Metal ReLU backward: %w", err)
	}
	if err = execution.MarkRead(input); err != nil {
		return false, fmt.Errorf("matrix: record Metal ReLU backward input use: %w", err)
	}
	if err = execution.MarkRead(outputGradient); err != nil {
		return false, fmt.Errorf("matrix: record Metal ReLU backward output gradient use: %w", err)
	}

	return true, nil
}

func softmaxRowsBackwardIntoDevice(
	input,
	outputGradient,
	inputGradient *Matrix,
) (handled bool, err error) {
	var (
		execution            *device.Execution
		inputBuffer          *device.Buffer
		outputGradientBuffer *device.Buffer
		inputGradientBuffer  *device.Buffer
		allocated            bool
		uploaded             bool
	)

	if execution, err = compatibleExecution(input, outputGradient, inputGradient); err != nil {
		return false, err
	}
	if !metalBackwardElementwiseSupported(execution, input) {
		return false, nil
	}
	if err = execution.Bind(input); err != nil {
		return false, fmt.Errorf("matrix: bind Metal Softmax backward input: %w", err)
	}
	if err = execution.Bind(outputGradient); err != nil {
		return false, fmt.Errorf("matrix: bind Metal Softmax backward output gradient: %w", err)
	}
	if err = execution.Bind(inputGradient); err != nil {
		return false, fmt.Errorf("matrix: bind Metal Softmax backward destination: %w", err)
	}
	if inputBuffer, allocated, uploaded, err = input.ensureExecutionDeviceBuffer(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal Softmax backward input: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(input.data))*4)
	if outputGradientBuffer, allocated, uploaded, err =
		outputGradient.ensureExecutionDeviceBuffer(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal Softmax backward output gradient: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(outputGradient.data))*4)
	if inputGradientBuffer, allocated, err = inputGradient.beginExecutionDeviceWrite(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal Softmax backward destination: %w", err)
	}
	execution.RecordDevicePreparation(allocated, false, 0)
	if err = execution.EncodeSoftmaxRowsBackward(
		inputBuffer,
		outputGradientBuffer,
		inputGradientBuffer,
		uint32(input.rows),
		uint32(input.cols),
		uint64(len(inputGradient.data))*4,
		deviceWritePublication(inputGradient, inputGradientBuffer),
	); err != nil {
		return false, fmt.Errorf("matrix: encode Metal Softmax backward: %w", err)
	}
	if err = execution.MarkRead(input); err != nil {
		return false, fmt.Errorf("matrix: record Metal Softmax backward input use: %w", err)
	}
	if err = execution.MarkRead(outputGradient); err != nil {
		return false, fmt.Errorf("matrix: record Metal Softmax backward output gradient use: %w", err)
	}

	return true, nil
}

func columnSumsIntoDevice(input, result *Matrix) (handled bool, err error) {
	var (
		execution    *device.Execution
		inputBuffer  *device.Buffer
		resultBuffer *device.Buffer
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
		return false, fmt.Errorf("matrix: bind Metal column sums input: %w", err)
	}
	if err = execution.Bind(result); err != nil {
		return false, fmt.Errorf("matrix: bind Metal column sums destination: %w", err)
	}
	if inputBuffer, allocated, uploaded, err = input.ensureExecutionDeviceBuffer(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal column sums input: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(input.data))*4)
	if resultBuffer, allocated, err = result.beginExecutionDeviceWrite(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal column sums destination: %w", err)
	}
	execution.RecordDevicePreparation(allocated, false, 0)
	if err = execution.EncodeColumnSums(
		inputBuffer,
		resultBuffer,
		uint32(input.rows),
		uint32(input.cols),
		uint64(len(result.data))*4,
		deviceWritePublication(result, resultBuffer),
	); err != nil {
		return false, fmt.Errorf("matrix: encode Metal column sums: %w", err)
	}
	if err = execution.MarkRead(input); err != nil {
		return false, fmt.Errorf("matrix: record Metal column sums input use: %w", err)
	}

	return true, nil
}

func accumulateColumnSumsIntoDevice(input, result *Matrix) (handled bool, err error) {
	var (
		execution           *device.Execution
		inputBuffer         *device.Buffer
		currentResultBuffer *device.Buffer
		resultBuffer        *device.Buffer
		allocated           bool
		uploaded            bool
	)

	if execution, err = compatibleExecution(input, result); err != nil {
		return false, err
	}
	if execution == nil || !execution.Activated() || !metalElementwiseSupported(input) {
		return false, nil
	}
	if err = execution.Bind(input); err != nil {
		return false, fmt.Errorf("matrix: bind Metal accumulated column sums input: %w", err)
	}
	if err = execution.Bind(result); err != nil {
		return false, fmt.Errorf("matrix: bind Metal accumulated column sums destination: %w", err)
	}
	if inputBuffer, allocated, uploaded, err = input.ensureExecutionDeviceBuffer(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal accumulated column sums input: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(input.data))*4)
	if currentResultBuffer, allocated, uploaded, err =
		result.ensureExecutionDeviceBuffer(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare current Metal accumulated column sums destination: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(result.data))*4)
	if resultBuffer, allocated, err = result.beginExecutionDeviceWrite(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal accumulated column sums destination: %w", err)
	}
	execution.RecordDevicePreparation(allocated, false, 0)
	if err = execution.EncodeCopy(
		currentResultBuffer,
		resultBuffer,
		uint64(len(result.data))*4,
		deviceWritePublication(result, resultBuffer),
	); err != nil {
		return false, fmt.Errorf("matrix: encode Metal accumulated column sums seed: %w", err)
	}
	if err = execution.EncodeAccumulateColumnSums(
		inputBuffer,
		resultBuffer,
		uint32(input.rows),
		uint32(input.cols),
	); err != nil {
		return false, fmt.Errorf("matrix: encode Metal accumulated column sums: %w", err)
	}
	if err = execution.MarkRead(input); err != nil {
		return false, fmt.Errorf("matrix: record Metal accumulated column sums input use: %w", err)
	}
	if err = execution.MarkRead(result); err != nil {
		return false, fmt.Errorf("matrix: record Metal accumulated column sums destination use: %w", err)
	}

	return true, nil
}

func resetDevice(value *Matrix) (handled bool, err error) {
	var (
		execution *device.Execution
		buffer    *device.Buffer
		allocated bool
		owned     bool
	)

	execution = value.execution()
	if execution != nil || !matrixDeviceCurrent(value) {
		return false, nil
	}
	execution = device.NewExecution(value.residency.Runtime())
	owned = true
	if execution == nil {
		err = errors.New("matrix: create Metal reset execution: runtime is nil")
		return false, err
	}
	if owned {
		defer func() {
			if err != nil && execution.Active() {
				err = errors.Join(err, execution.Abort(err))
			}
		}()
	}
	if err = execution.Bind(value); err != nil {
		return false, fmt.Errorf("matrix: bind Metal reset destination: %w", err)
	}
	if buffer, allocated, err = value.beginExecutionDeviceWrite(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal reset destination: %w", err)
	}
	execution.RecordDevicePreparation(allocated, false, 0)
	if err = execution.EncodeFill(
		buffer,
		0,
		uint64(len(value.data))*4,
		deviceWritePublication(value, buffer),
	); err != nil {
		return false, fmt.Errorf("matrix: encode Metal reset: %w", err)
	}
	if owned {
		if err = execution.Finish(); err != nil {
			return false, fmt.Errorf("matrix: finish Metal reset execution: %w", err)
		}
	}

	return true, nil
}

func metalBackwardElementwiseSupported(execution *device.Execution, input *Matrix) (ok bool) {
	if execution == nil || !metalElementwiseSupported(input) {
		return false
	}
	if execution.Activated() {
		return true
	}

	ok = matrixDeviceCurrent(input)
	return ok
}

func matrixDeviceCurrent(value *Matrix) (current bool) {
	var snapshot device.ResidencySnapshot

	if value == nil || value.residency == nil {
		return false
	}
	snapshot = value.residency.Snapshot()
	current = snapshot.HasBuffer &&
		snapshot.DeviceRevision == snapshot.LogicalRevision
	return current
}

func deviceWritePublication(
	result *Matrix,
	buffer *device.Buffer,
) (publication device.Publication) {
	publication.Publish = func() (err error) {
		err = result.publishDeviceWrite(buffer)
		return err
	}
	publication.Discard = func(cause error) (err error) {
		err = result.failDeviceWrite(buffer, cause)
		return err
	}
	return publication
}
