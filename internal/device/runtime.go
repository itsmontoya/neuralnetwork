package device

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
)

var (
	sharedRuntimeOnce  sync.Once
	sharedRuntime      *Runtime
	sharedRuntimeReady atomic.Bool
)

// SharedRuntime returns the process-wide private device runtime when available.
func SharedRuntime() (runtime *Runtime, available bool, err error) {
	sharedRuntimeOnce.Do(func() {
		sharedRuntime = newRuntime(newPlatformBackend())
	})

	if sharedRuntime == nil {
		err = errors.New("device: shared runtime initialization returned nil")
		return nil, false, err
	}

	available, err = sharedRuntime.backend.available()
	if err != nil {
		return nil, false, fmt.Errorf("device: initialize runtime: %w", err)
	}
	if !available {
		return nil, false, nil
	}

	sharedRuntimeReady.Store(true)
	runtime = sharedRuntime
	return runtime, true, nil
}

func newRuntime(runtimeBackend backend) (runtime *Runtime) {
	var value Runtime
	value.backend = runtimeBackend
	return &value
}

// Runtime owns shared device resources and creates buffers and command scopes.
type Runtime struct {
	backend backend
}

// Available reports whether the runtime backend can execute device work.
func (r *Runtime) Available() (available bool, err error) {
	if r == nil || r.backend == nil {
		err = errors.New("device: runtime is nil")
		return false, err
	}

	available, err = r.backend.available()
	if err != nil {
		return false, fmt.Errorf("device: inspect runtime availability: %w", err)
	}

	return available, nil
}

// NewBuffer allocates a private float32 device buffer.
func (r *Runtime) NewBuffer(count uint64) (buffer *Buffer, err error) {
	var (
		bytes  uint64
		handle any
		value  Buffer
	)

	if r == nil || r.backend == nil {
		err = errors.New("device: runtime is nil")
		return nil, err
	}

	if bytes, err = float32Bytes(count); err != nil {
		return nil, err
	}

	if handle, err = r.backend.newBuffer(bytes); err != nil {
		return nil, fmt.Errorf("device: allocate %d-byte buffer: %w", bytes, err)
	}
	if handle == nil {
		err = errors.New("device: allocate buffer: backend returned nil handle")
		return nil, err
	}

	value.runtime = r
	value.handle = handle
	value.count = count
	value.bytes = bytes
	return &value, nil
}

// NewScope creates an independent device command scope.
func (r *Runtime) NewScope() (scope *Scope, err error) {
	var (
		handle any
		value  Scope
	)

	if r == nil || r.backend == nil {
		err = errors.New("device: runtime is nil")
		return nil, err
	}

	if handle, err = r.backend.newScope(); err != nil {
		return nil, fmt.Errorf("device: create command scope: %w", err)
	}
	if handle == nil {
		err = errors.New("device: create command scope: backend returned nil handle")
		return nil, err
	}

	value.runtime = r
	value.handle = handle
	value.state = scopeStateEncoding
	return &value, nil
}

// ResourceSnapshot returns aggregate private runtime resource counters.
func (r *Runtime) ResourceSnapshot() (snapshot ResourceSnapshot) {
	if r == nil || r.backend == nil {
		return snapshot
	}

	snapshot = r.backend.resourceSnapshot()
	return snapshot
}

// ResetResourcePeaks resets aggregate counters when the runtime is idle.
func (r *Runtime) ResetResourcePeaks() (err error) {
	if r == nil || r.backend == nil {
		err = errors.New("device: runtime is nil")
		return err
	}

	if err = r.backend.resetResourcePeaks(); err != nil {
		return fmt.Errorf("device: reset resource counters: %w", err)
	}

	return nil
}

func float32Bytes(count uint64) (bytes uint64, err error) {
	if count == 0 {
		err = errors.New("device: buffer element count must be positive")
		return 0, err
	}

	if count > math.MaxUint64/4 {
		err = fmt.Errorf("device: float32 buffer length overflow: count=%d", count)
		return 0, err
	}

	bytes = count * 4
	return bytes, nil
}
