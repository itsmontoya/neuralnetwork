//go:build darwin && cgo && metal && !purego

package matrix

import (
	"errors"
	"fmt"
	"sync"

	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
)

const (
	metalMatMulMinOperations = 1 << 20
	maxMetalUint32           = 1<<32 - 1
)

const (
	metalMatMulStandard uint32 = iota
	metalMatMulLeftTranspose
	metalMatMulRightTranspose
)

var (
	metalErrorMutex sync.Mutex
	metalError      string
)

func metalRunMatMul(left, right, result *Matrix, variant uint32) (err error) {
	var (
		runtimeValue *device.Runtime
		execution    *device.Execution
		available    bool
		owned        bool
	)

	if !metalMatMulSupported(left, right, result, variant) {
		if err = inheritExecution(result, left, right); err != nil {
			return err
		}
		err = matMulHost(left, right, result, variant)
		return err
	}
	if execution, err = compatibleExecution(left, right, result); err != nil {
		return err
	}
	if execution == nil {
		if runtimeValue, available, err = device.SharedRuntime(); err != nil {
			err = fmt.Errorf("matrix: initialize Metal runtime: %w", err)
			metalRecordFailure(err)
			return err
		}
		if !available {
			err = matMulHost(left, right, result, variant)
			return err
		}
		execution = device.NewExecution(runtimeValue)
		owned = true
	} else {
		runtimeValue = execution.Runtime()
	}
	if execution == nil || runtimeValue == nil {
		err = errors.New("matrix: create Metal execution: runtime is nil")
		return err
	}

	if err = metalCallMatMul(execution, left, right, result, variant); err != nil {
		if owned {
			err = errors.Join(err, execution.Abort(err))
		}
		metalRecordFailure(err)
		return err
	}
	if owned {
		if err = execution.Finish(); err != nil {
			err = fmt.Errorf("matrix: finish Metal multiplication execution: %w", err)
			metalRecordFailure(err)
			return err
		}
	}
	return nil
}

func metalCallMatMul(
	execution *device.Execution,
	left,
	right,
	result *Matrix,
	variant uint32,
) (err error) {
	var (
		leftBuffer   *device.Buffer
		rightBuffer  *device.Buffer
		resultBuffer *device.Buffer
		operation    device.Operation
		publication  device.Publication
		allocated    bool
		uploaded     bool
	)

	if operation, err = metalOperation(variant); err != nil {
		return err
	}
	if err = execution.Bind(left); err != nil {
		return fmt.Errorf("matrix: bind Metal left input: %w", err)
	}
	if err = execution.Bind(right); err != nil {
		return fmt.Errorf("matrix: bind Metal right input: %w", err)
	}
	if err = execution.Bind(result); err != nil {
		return fmt.Errorf("matrix: bind Metal destination: %w", err)
	}
	if leftBuffer, allocated, uploaded, err = left.ensureExecutionDeviceBuffer(execution); err != nil {
		return fmt.Errorf("matrix: prepare Metal left input: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded)
	if rightBuffer, allocated, uploaded, err = right.ensureExecutionDeviceBuffer(execution); err != nil {
		return fmt.Errorf("matrix: prepare Metal right input: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded)
	if resultBuffer, allocated, err = result.beginExecutionDeviceWrite(execution); err != nil {
		return fmt.Errorf("matrix: prepare Metal destination: %w", err)
	}
	execution.RecordDevicePreparation(allocated, false)
	publication.Publish = func() (publishErr error) {
		publishErr = result.publishDeviceWrite(resultBuffer)
		return publishErr
	}
	publication.Discard = func(cause error) (discardErr error) {
		discardErr = result.failDeviceWrite(resultBuffer, cause)
		return discardErr
	}
	if err = execution.EncodeMatMul(
		leftBuffer,
		rightBuffer,
		resultBuffer,
		uint32(left.rows),
		uint32(left.cols),
		uint32(right.rows),
		uint32(right.cols),
		uint32(result.rows),
		uint32(result.cols),
		operation,
		uint64(len(result.data))*4,
		publication,
	); err != nil {
		return fmt.Errorf("matrix: encode Metal multiplication: %w", err)
	}
	if err = execution.MarkRead(left); err != nil {
		return fmt.Errorf("matrix: record Metal left input use: %w", err)
	}
	if err = execution.MarkRead(right); err != nil {
		return fmt.Errorf("matrix: record Metal right input use: %w", err)
	}
	return nil
}

func matMulHost(left, right, result *Matrix, variant uint32) (err error) {
	if err = inheritExecution(result, left, right); err != nil {
		return err
	}
	if err = left.ensureHostCurrent(); err != nil {
		return err
	}
	if err = right.ensureHostCurrent(); err != nil {
		return err
	}
	if err = result.markHostWrite(); err != nil {
		return err
	}

	switch variant {
	case metalMatMulStandard:
		matMulIntoPure(left, right, result)
	case metalMatMulLeftTranspose:
		matMulLeftTransposeIntoPure(left, right, result)
	case metalMatMulRightTranspose:
		matMulRightTransposeIntoPure(left, right, result)
	default:
		err = fmt.Errorf("matrix: unsupported Metal multiplication variant: %d", variant)
		return err
	}

	return nil
}

func metalOperation(variant uint32) (operation device.Operation, err error) {
	switch variant {
	case metalMatMulStandard:
		operation = device.OperationMatMul
	case metalMatMulLeftTranspose:
		operation = device.OperationMatMulLeftTranspose
	case metalMatMulRightTranspose:
		operation = device.OperationMatMulRightTranspose
	default:
		err = fmt.Errorf("matrix: unsupported Metal multiplication variant: %d", variant)
	}
	return operation, err
}

func metalAvailable() (ok bool) {
	var (
		available bool
		err       error
	)

	_, available, err = device.SharedRuntime()
	if err != nil {
		metalSetError(err)
		return false
	}
	if !available {
		metalSetError(errors.New("metal: no default device"))
		return false
	}

	ok = true
	return ok
}

func metalLastError() (message string) {
	metalErrorMutex.Lock()
	message = metalError
	metalErrorMutex.Unlock()
	return message
}

func metalRecordFailure(err error) {
	metalSetError(err)
	metaltest.RecordFailure(err.Error())
}

func metalSetError(err error) {
	metalErrorMutex.Lock()
	metalError = err.Error()
	metalErrorMutex.Unlock()
}

func metalMatMulSupported(left, right, result *Matrix, variant uint32) (ok bool) {
	var (
		inner     int
		operation uint64
	)

	if len(left.data) == 0 || len(right.data) == 0 || len(result.data) == 0 {
		return false
	}

	if !metalDimensionSupported(left.rows) ||
		!metalDimensionSupported(left.cols) ||
		!metalDimensionSupported(right.rows) ||
		!metalDimensionSupported(right.cols) ||
		!metalDimensionSupported(result.rows) ||
		!metalDimensionSupported(result.cols) {
		return false
	}

	switch variant {
	case metalMatMulStandard:
		inner = left.cols
	case metalMatMulLeftTranspose:
		inner = left.rows
	case metalMatMulRightTranspose:
		inner = left.cols
	default:
		return false
	}

	operation = uint64(result.rows) * uint64(result.cols) * uint64(inner)
	if operation < metalMatMulMinOperations {
		return false
	}

	ok = true
	return ok
}

func metalDimensionSupported(dimension int) (ok bool) {
	if uint64(dimension) > maxMetalUint32 {
		return false
	}

	ok = true
	return ok
}
