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

func metalRunMatMul(left, right, result *Matrix, variant uint32) (ok bool) {
	var err error

	if !metalMatMulSupported(left, right, result, variant) {
		return false
	}

	if err = metalCallMatMul(left, right, result, variant); err != nil {
		metalSetError(err)
		metaltest.RecordFailure(err.Error())
		return false
	}

	ok = true
	return ok
}

func metalCallMatMul(left, right, result *Matrix, variant uint32) (err error) {
	var (
		runtime      *device.Runtime
		leftBuffer   *device.Buffer
		rightBuffer  *device.Buffer
		resultBuffer *device.Buffer
		scope        *device.Scope
		operation    device.Operation
		activity     metalBridgeActivity
		available    bool
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

	if runtime, available, err = device.SharedRuntime(); err != nil {
		return fmt.Errorf("matrix: initialize Metal runtime: %w", err)
	}
	if !available {
		return errors.New("matrix: Metal runtime unavailable: no default device")
	}

	switch variant {
	case metalMatMulStandard:
		operation = device.OperationMatMul
	case metalMatMulLeftTranspose:
		operation = device.OperationMatMulLeftTranspose
	case metalMatMulRightTranspose:
		operation = device.OperationMatMulRightTranspose
	default:
		return fmt.Errorf("matrix: unsupported Metal multiplication variant: %d", variant)
	}

	if leftBuffer, err = runtime.NewBuffer(uint64(len(left.data))); err != nil {
		return fmt.Errorf("matrix: allocate Metal left input: %w", err)
	}
	activity.bufferCreations++
	defer leftBuffer.Release()
	if rightBuffer, err = runtime.NewBuffer(uint64(len(right.data))); err != nil {
		return fmt.Errorf("matrix: allocate Metal right input: %w", err)
	}
	activity.bufferCreations++
	defer rightBuffer.Release()
	if resultBuffer, err = runtime.NewBuffer(uint64(len(result.data))); err != nil {
		return fmt.Errorf("matrix: allocate Metal destination: %w", err)
	}
	activity.bufferCreations++
	defer resultBuffer.Release()

	if err = leftBuffer.Upload(left.data); err != nil {
		return fmt.Errorf("matrix: upload Metal left input: %w", err)
	}
	activity.inputUploads++
	if err = rightBuffer.Upload(right.data); err != nil {
		return fmt.Errorf("matrix: upload Metal right input: %w", err)
	}
	activity.inputUploads++

	if scope, err = runtime.NewScope(); err != nil {
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
	if err = resultBuffer.Download(result.data); err != nil {
		return fmt.Errorf("matrix: download Metal multiplication result: %w", err)
	}
	activity.resultDownloads++

	return nil
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
