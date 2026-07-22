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

type metalBridgeActivity struct {
	bufferCreations    uint64
	inputUploads       uint64
	resultDownloads    uint64
	commandSubmissions uint64
	waits              uint64
}

func metalRunMatMul(left, right, result *Matrix, variant uint32) (err error) {
	var (
		runtimeValue *device.Runtime
		available    bool
	)

	if !metalMatMulSupported(left, right, result, variant) {
		err = matMulHost(left, right, result, variant)
		return err
	}
	if runtimeValue, available, err = device.SharedRuntime(); err != nil {
		err = fmt.Errorf("matrix: initialize Metal runtime: %w", err)
		metalRecordFailure(err)
		return err
	}
	if !available {
		err = matMulHost(left, right, result, variant)
		return err
	}

	if err = metalCallMatMul(runtimeValue, left, right, result, variant); err != nil {
		metalRecordFailure(err)
		return err
	}
	return nil
}

func metalCallMatMul(
	runtimeValue *device.Runtime,
	left,
	right,
	result *Matrix,
	variant uint32,
) (err error) {
	var (
		leftBuffer   *device.Buffer
		rightBuffer  *device.Buffer
		resultBuffer *device.Buffer
		scope        *device.Scope
		operation    device.Operation
		activity     metalBridgeActivity
		allocated    bool
		uploaded     bool
		published    bool
	)

	if metaltest.Enabled() {
		defer func() {
			metaltest.RecordBridgeActivity(
				activity.bufferCreations,
				activity.inputUploads,
				activity.resultDownloads,
				activity.commandSubmissions,
				activity.waits,
			)
		}()
	}

	if operation, err = metalOperation(variant); err != nil {
		return err
	}
	if leftBuffer, allocated, uploaded, err = left.ensureDeviceBuffer(runtimeValue); err != nil {
		return fmt.Errorf("matrix: prepare Metal left input: %w", err)
	}
	activity.recordDevicePreparation(allocated, uploaded)
	if rightBuffer, allocated, uploaded, err = right.ensureDeviceBuffer(runtimeValue); err != nil {
		return fmt.Errorf("matrix: prepare Metal right input: %w", err)
	}
	activity.recordDevicePreparation(allocated, uploaded)
	if resultBuffer, allocated, err = result.beginDeviceWrite(runtimeValue); err != nil {
		return fmt.Errorf("matrix: prepare Metal destination: %w", err)
	}
	if allocated {
		activity.bufferCreations++
	}
	defer func() {
		var cleanupErr error
		if !published {
			cleanupErr = result.failDeviceWrite(resultBuffer, err)
			if cleanupErr != nil {
				err = errors.Join(err, cleanupErr)
			}
		}
	}()

	if scope, err = runtimeValue.NewScope(); err != nil {
		return fmt.Errorf("matrix: create Metal multiplication scope: %w", err)
	}
	defer func() {
		var releaseErr error
		if releaseErr = scope.Release(); err == nil && releaseErr != nil {
			err = fmt.Errorf("matrix: release Metal multiplication scope: %w", releaseErr)
		}
	}()

	if err = scope.EncodeMatMul(
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
	); err != nil {
		return fmt.Errorf("matrix: encode Metal multiplication: %w", err)
	}
	if err = scope.Commit(); err != nil {
		return fmt.Errorf("matrix: commit Metal multiplication: %w", err)
	}
	activity.commandSubmissions++
	if err = scope.Wait(); err != nil {
		activity.waits++
		return fmt.Errorf("matrix: wait for Metal multiplication: %w", err)
	}
	activity.waits++
	if err = result.publishDeviceWrite(resultBuffer); err != nil {
		return err
	}
	published = true
	return nil
}

func matMulHost(left, right, result *Matrix, variant uint32) (err error) {
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

func (a *metalBridgeActivity) recordDevicePreparation(allocated, uploaded bool) {
	if allocated {
		a.bufferCreations++
	}
	if uploaded {
		a.inputUploads++
	}
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
