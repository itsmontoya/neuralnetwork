//go:build darwin && cgo && metal && !purego

package matrix

/*
#cgo LDFLAGS: -framework Foundation -framework Metal
#include "metal_backend.h"
*/
import "C"

import (
	"unsafe"

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

func metalRunMatMul(left, right, result *Matrix, variant uint32) (ok bool) {
	if !metalMatMulSupported(left, right, result, variant) {
		return false
	}

	if !metalAvailable() {
		metaltest.RecordFailure(metalLastError())
		return false
	}

	var status C.int
	if metaltest.Enabled() {
		var activity C.NNMetalCounters
		status = metalCallMatMul(left, right, result, variant, &activity)
		metaltest.RecordBridgeActivity(
			uint64(activity.bufferCreations),
			uint64(activity.inputUploads),
			uint64(activity.resultDownloads),
			uint64(activity.commandSubmissions),
			uint64(activity.waits),
		)
	} else {
		status = metalCallMatMul(left, right, result, variant, nil)
	}

	if status == 0 {
		metaltest.RecordFailure(metalLastError())
		return false
	}

	ok = true
	return ok
}

func metalCallMatMul(left, right, result *Matrix, variant uint32, activity *C.NNMetalCounters) (status C.int) {
	status = C.nn_metal_matmul(
		(*C.float)(unsafe.Pointer(&left.data[0])),
		(*C.float)(unsafe.Pointer(&right.data[0])),
		(*C.float)(unsafe.Pointer(&result.data[0])),
		C.uint32_t(left.rows),
		C.uint32_t(left.cols),
		C.uint32_t(right.rows),
		C.uint32_t(right.cols),
		C.uint32_t(result.rows),
		C.uint32_t(result.cols),
		C.uint32_t(variant),
		activity,
	)
	return status
}

func metalAvailable() (ok bool) {
	if C.nn_metal_available() == 0 {
		return false
	}

	ok = true
	return ok
}

func metalLastError() (message string) {
	message = C.GoString(C.nn_metal_last_error())
	return message
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
