package gpu

import (
	"fmt"
	"math"
)

// NewKVCache allocates two pairs of contiguous FP32 key and value planes.
// Each plane has logical shape [capacity, heads, headWidth] and starts empty.
func NewKVCache(device *Device, capacity, heads, headWidth int64) (out *KVCache, err error) {
	return NewKVCacheWithDType(device, capacity, heads, headWidth, Float32)
}

// NewKVCacheWithDType allocates two pairs of contiguous key and value planes
// using the requested storage DType (Float32 or Float16). Each plane has
// logical shape [capacity, heads, headWidth] and starts empty.
func NewKVCacheWithDType(device *Device, capacity, heads, headWidth int64, dtype DType) (out *KVCache, err error) {
	if dtype != Float32 && dtype != Float16 {
		return nil, fmt.Errorf("gpu: KV cache: %w: unsupported KV cache dtype %s", ErrInvalid, dtype)
	}
	var (
		shape = []int64{2, capacity, heads, headWidth}
		state kvCacheState
	)
	if _, _, err = tensorLayout(shape, dtype, uint64(math.MaxUint32)*dtype.Bytes()); err != nil {
		return nil, fmt.Errorf("gpu: KV cache: %w", err)
	}
	if state.slots[0], err = device.NewTensorWithDType(shape, dtype); err != nil {
		return nil, err
	}
	if state.slots[1], err = device.NewTensorWithDType(shape, dtype); err != nil {
		state.slots[0].Close()
		return nil, err
	}
	state.device = state.slots[0].state.storage.device
	state.dtype = dtype
	state.capacity = capacity
	state.heads = heads
	state.width = headWidth
	return &KVCache{state: &state}, nil
}

// KVCache owns the key and value history for one caller-managed sequence.
// Copies of the handle share the same storage, length and lifetime. Callers
// serialize sequence-level steps and can share model weights across caches.
type KVCache struct {
	state *kvCacheState
}

type kvCacheState struct {
	device   *deviceState
	dtype    DType
	capacity int64
	heads    int64
	width    int64
	slots    [2]*Tensor

	// The device mutex protects the publication state and slot leases.
	active  int
	length  int64
	pending *kvAppendState
	closed  bool
}

// DType returns the immutable element storage type of the cache, including
// after Close. A nil cache handle returns the invalid zero DType.
func (c *KVCache) DType() (dtype DType) {
	if c == nil || c.state == nil {
		return 0
	}
	return c.state.dtype
}

// Len returns the number of committed rows, or zero for a closed cache.
func (c *KVCache) Len() (length int64) {
	if c == nil || c.state == nil {
		return 0
	}
	var state = c.state
	state.device.mutex.Lock()
	defer state.device.mutex.Unlock()
	return state.length
}

// Reset clears the committed length without reallocating. Previously encoded
// readers keep their snapshot. Reset returns ErrBusy while an append is pending.
func (c *KVCache) Reset() (err error) {
	if c == nil || c.state == nil {
		return ErrClosed
	}
	var state = c.state
	state.device.mutex.Lock()
	defer state.device.mutex.Unlock()
	if state.closed {
		return ErrClosed
	}
	if state.pending != nil {
		return ErrBusy
	}
	state.length = 0
	return nil
}

// Close prevents new uses and publication and releases both slots after their
// outstanding GPU leases end. Close does not wait and is idempotent.
func (c *KVCache) Close() (err error) {
	if c == nil || c.state == nil {
		return nil
	}
	var state = c.state
	state.device.mutex.Lock()
	state.closed = true
	state.length = 0
	state.pending = nil
	state.device.mutex.Unlock()
	for _, slot := range state.slots {
		slot.Close()
	}
	return nil
}
