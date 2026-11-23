package gpu

import (
	"fmt"
	"slices"
)

func newTensor(device *deviceState, shape []int64, count uint64, buffer allocation) (out *Tensor) {
	var state tensorState
	state.device = device
	state.shape = slices.Clone(shape)
	state.count = count
	state.buffer = buffer
	return &Tensor{state: &state}
}

// Tensor owns one contiguous row-major FP32 allocation on its creating device.
// Copies of a Tensor handle refer to the same allocation and lifetime.
type Tensor struct {
	state *tensorState
}

type tensorState struct {
	device *deviceState
	shape  []int64 // Immutable. The creating device protects the mutable fields below.
	count  uint64
	buffer allocation

	readers     int
	writer      *scopeState
	hostAccess  bool
	initialized bool
	closed      bool
}

// Shape returns a copy of the immutable dimensions, including after Close.
func (t *Tensor) Shape() (shape []int64) {
	if t == nil || t.state == nil {
		return nil
	}
	return slices.Clone(t.state.shape)
}

// Upload copies a full set of FP32 values before returning. It returns ErrBusy
// if another host transfer or a recording or submitted scope holds the tensor.
func (t *Tensor) Upload(values []float32) (err error) {
	var state *tensorState
	if state, err = t.beginTransfer(len(values), true); err != nil {
		return err
	}

	err = state.buffer.Upload(values)
	state.endTransfer(true, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: upload tensor: %w", err)
	}
	return nil
}

// DownloadInto copies all completed values into destination. The caller owns
// destination, which must have the exact element count. This call returns
// ErrBusy during an outstanding lease and ErrUninitialized before a full write.
func (t *Tensor) DownloadInto(destination []float32) (err error) {
	var state *tensorState
	if state, err = t.beginTransfer(len(destination), false); err != nil {
		return err
	}

	err = state.buffer.Download(destination)
	state.endTransfer(false, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: download tensor: %w", err)
	}
	return nil
}

// Close prevents new work on the tensor and defers release until existing
// transfers and command scopes finish using it. Close is idempotent.
func (t *Tensor) Close() (err error) {
	if t == nil || t.state == nil {
		return nil
	}

	var state = t.state
	state.device.mutex.Lock()
	defer state.device.mutex.Unlock()
	state.closed = true
	state.releaseIfIdle()
	return nil
}

func (t *Tensor) beginTransfer(length int, write bool) (state *tensorState, err error) {
	if t == nil || t.state == nil {
		return nil, ErrClosed
	}

	state = t.state
	state.device.mutex.Lock()
	defer state.device.mutex.Unlock()
	if state.closed {
		return nil, ErrClosed
	}
	if state.hostAccess || state.readers > 0 || state.writer != nil {
		return nil, ErrBusy
	}
	if uint64(length) != state.count {
		return nil, fmt.Errorf("%w: transfer length is %d, want %d", ErrInvalid, length, state.count)
	}
	if !write && !state.initialized {
		return nil, ErrUninitialized
	}

	state.hostAccess = true
	return state, nil
}

func (t *tensorState) endTransfer(write, success bool) {
	t.device.mutex.Lock()
	defer t.device.mutex.Unlock()
	if write {
		t.initialized = success
	}
	t.hostAccess = false
	t.releaseIfIdle()
}

func (t *tensorState) releaseIfIdle() {
	if t.closed && t.buffer != nil && !t.hostAccess && t.readers == 0 && t.writer == nil {
		t.buffer.Release()
		t.buffer = nil
		t.device.children--
	}
}
