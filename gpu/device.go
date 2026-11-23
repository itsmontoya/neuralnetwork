package gpu

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/itsmontoya/neuralnetwork/internal/device"
)

// OpenDefault opens a logical handle to the system's default Metal device.
// It returns ErrUnavailable if this build or machine cannot use Metal.
func OpenDefault() (out *Device, err error) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		return nil, ErrUnavailable
	}

	var (
		native    *device.Runtime
		available bool
	)

	if native, available, err = device.ExperimentalRuntime(); err != nil {
		return nil, fmt.Errorf("gpu: open default device: %w", err)
	}
	if !available {
		return nil, ErrUnavailable
	}

	return newDevice(metalBackend{runtime: native}, native.MaxBufferLength()), nil
}

func newDevice(runtimeBackend backend, maxBufferLength uint64) (out *Device) {
	var state deviceState
	state.backend = runtimeBackend
	state.maxBufferLength = maxBufferLength
	return &Device{state: &state}
}

// Device owns a logical namespace for tensors and scopes on the default Metal device.
// Copies of a Device handle refer to the same namespace and lifetime.
type Device struct {
	state *deviceState
}

type deviceState struct {
	mutex           sync.Mutex // Owns tensor leases and child lifetimes.
	backend         backend
	maxBufferLength uint64
	children        int
	closed          bool
}

// NewFloat32 allocates an uninitialized contiguous row-major FP32 tensor.
// Its shape is copied and must contain at least one positive dimension.
func (d *Device) NewFloat32(shape []int64) (out *Tensor, err error) {
	if d == nil || d.state == nil {
		return nil, ErrClosed
	}

	var state = d.state
	state.mutex.Lock()
	defer state.mutex.Unlock()
	if state.closed {
		return nil, ErrClosed
	}

	var (
		count  uint64
		buffer allocation
	)
	if count, err = float32Count(shape, state.maxBufferLength); err != nil {
		return nil, err
	}
	if buffer, err = state.backend.newBuffer(count); err != nil {
		return nil, fmt.Errorf("gpu: allocate FP32 tensor: %w", err)
	}
	state.children++
	return newTensor(state, shape, count, buffer), nil
}

// NewScope creates an independent command scope owned by the device.
func (d *Device) NewScope() (out *Scope, err error) {
	if d == nil || d.state == nil {
		return nil, ErrClosed
	}

	var state = d.state
	state.mutex.Lock()
	defer state.mutex.Unlock()
	if state.closed {
		return nil, ErrClosed
	}

	var commands commandBuffer
	if commands, err = state.backend.newScope(); err != nil {
		return nil, fmt.Errorf("gpu: create scope: %w", err)
	}
	state.children++
	return newScope(state, commands), nil
}

// Close closes the logical handle. It returns ErrBusy while any children remain.
func (d *Device) Close() (err error) {
	if d == nil || d.state == nil {
		return nil
	}

	var state = d.state
	state.mutex.Lock()
	defer state.mutex.Unlock()
	if state.children != 0 {
		return ErrBusy
	}
	state.closed = true
	return nil
}

func float32Count(shape []int64, maxBufferLength uint64) (count uint64, err error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("%w: tensor rank must be positive", ErrInvalid)
	}

	var limit = min(uint64(math.MaxInt), maxBufferLength) / 4
	count = 1
	for axis, dimension := range shape {
		if dimension <= 0 {
			return 0, fmt.Errorf("%w: tensor dimension %d must be positive", ErrInvalid, axis)
		}
		if count > limit/uint64(dimension) {
			return 0, fmt.Errorf("%w: FP32 tensor exceeds the host or device buffer length limit", ErrInvalid)
		}
		count *= uint64(dimension)
	}

	return count, nil
}
