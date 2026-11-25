package gpu

import (
	"fmt"
	"slices"
)

func newTensor(device *deviceState, dtype DType, shape []int64, count, bytes uint64, buffer allocation) (out *Tensor) {
	var state tensorState
	state.device = device
	state.dtype = dtype
	state.shape = slices.Clone(shape)
	state.count = count
	state.bytes = bytes
	state.buffer = buffer
	return &Tensor{state: &state}
}

// Tensor owns one contiguous row-major allocation on its creating device.
// Copies of a Tensor handle refer to the same allocation and lifetime.
type Tensor struct {
	state *tensorState
}

type tensorState struct {
	device *deviceState
	dtype  DType
	shape  []int64 // Immutable. The creating device protects the mutable fields below.
	count  uint64
	bytes  uint64
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

// DType returns the immutable element storage type, including after Close.
// It returns an invalid zero DType for a nil or uninitialized handle.
func (t *Tensor) DType() (dtype DType) {
	if t == nil || t.state == nil {
		return 0
	}
	return t.state.dtype
}

// Bytes returns the immutable allocated byte size, including after Close.
func (t *Tensor) Bytes() (bytes uint64) {
	if t == nil || t.state == nil {
		return 0
	}
	return t.state.bytes
}

// Upload copies a full set of FP32 values before returning. It returns
// ErrInvalid for non-Float32 tensors and ErrBusy if another host transfer or a
// recording or submitted scope holds the tensor.
func (t *Tensor) Upload(values []float32) (err error) {
	var state *tensorState
	if state, err = t.beginTransfer(len(values), true, func(dtype DType) bool { return dtype == Float32 }); err != nil {
		return err
	}

	err = state.buffer.Upload(values)
	state.endTransfer(true, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: upload tensor: %w", err)
	}
	return nil
}

// DownloadInto copies all completed FP32 values into destination. The caller
// owns destination, which must have the exact element count. This call returns
// ErrInvalid for non-Float32 tensors, ErrBusy during an outstanding lease, and
// ErrUninitialized before a full write.
func (t *Tensor) DownloadInto(destination []float32) (err error) {
	var state *tensorState
	if state, err = t.beginTransfer(len(destination), false, func(dtype DType) bool { return dtype == Float32 }); err != nil {
		return err
	}

	err = state.buffer.Download(destination)
	state.endTransfer(false, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: download tensor: %w", err)
	}
	return nil
}

// UploadBits copies a full set of raw 16-bit element bits into a Float16 or
// BFloat16 tensor without conversion. It returns ErrInvalid for Float32
// tensors or length mismatches.
func (t *Tensor) UploadBits(values []uint16) (err error) {
	var state *tensorState
	if state, err = t.beginTransfer(len(values), true, DType.is16Bit); err != nil {
		return err
	}

	err = state.buffer.UploadUint16(values)
	state.endTransfer(true, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: upload tensor bits: %w", err)
	}
	return nil
}

// DownloadBitsInto copies all raw 16-bit element bits from a Float16 or
// BFloat16 tensor into destination without conversion. It returns ErrInvalid
// for Float32 tensors or length mismatches.
func (t *Tensor) DownloadBitsInto(destination []uint16) (err error) {
	var state *tensorState
	if state, err = t.beginTransfer(len(destination), false, DType.is16Bit); err != nil {
		return err
	}

	err = state.buffer.DownloadUint16(destination)
	state.endTransfer(false, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: download tensor bits: %w", err)
	}
	return nil
}

// UploadBytes copies a full raw byte buffer into the tensor without conversion.
// The slice length must equal Tensor.Bytes(). This supports all valid DTypes,
// including pre-packed Q8_0 and Q4_0 weight blocks.
func (t *Tensor) UploadBytes(values []byte) (err error) {
	var state *tensorState
	if state, err = t.beginByteTransfer(len(values), true); err != nil {
		return err
	}

	err = state.buffer.UploadBytes(values)
	state.endTransfer(true, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: upload tensor bytes: %w", err)
	}
	return nil
}

// DownloadBytesInto copies the full raw byte contents of the tensor into
// destination without conversion. The slice length must equal Tensor.Bytes().
func (t *Tensor) DownloadBytesInto(destination []byte) (err error) {
	var state *tensorState
	if state, err = t.beginByteTransfer(len(destination), false); err != nil {
		return err
	}

	err = state.buffer.DownloadBytes(destination)
	state.endTransfer(false, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: download tensor bytes: %w", err)
	}
	return nil
}

// UploadFloat32As converts FP32 values on the host to the tensor's DType
// (using IEEE-754 round-to-nearest-even for Float16/BFloat16 and block
// quantization for Q8_0/Q4_0) and copies the full set into the tensor. For
// Float32 tensors, values are uploaded directly without conversion.
func (t *Tensor) UploadFloat32As(values []float32) (err error) {
	var state *tensorState
	if state, err = t.beginTransfer(len(values), true, DType.Valid); err != nil {
		return err
	}

	switch state.dtype {
	case Float32:
		err = state.buffer.Upload(values)
	case Float16:
		var encoded = make([]uint16, len(values))
		for index, value := range values {
			encoded[index] = Float32ToFloat16(value)
		}
		err = state.buffer.UploadUint16(encoded)
	case BFloat16:
		var encoded = make([]uint16, len(values))
		for index, value := range values {
			encoded[index] = Float32ToBFloat16(value)
		}
		err = state.buffer.UploadUint16(encoded)
	case Q8_0:
		var encoded []byte
		if encoded, err = QuantizeQ8_0(values); err == nil {
			err = state.buffer.UploadBytes(encoded)
		}
	case Q4_0:
		var encoded []byte
		if encoded, err = QuantizeQ4_0(values); err == nil {
			err = state.buffer.UploadBytes(encoded)
		}
	}
	state.endTransfer(true, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: upload float32 as %s: %w", state.dtype, err)
	}
	return nil
}

// DownloadFloat32Into downloads all tensor elements and converts or dequantizes
// them to FP32 into destination. For Float32 tensors, values are downloaded directly.
func (t *Tensor) DownloadFloat32Into(destination []float32) (err error) {
	var state *tensorState
	if state, err = t.beginTransfer(len(destination), false, DType.Valid); err != nil {
		return err
	}

	switch state.dtype {
	case Float32:
		err = state.buffer.Download(destination)
	case Float16:
		var raw = make([]uint16, len(destination))
		if err = state.buffer.DownloadUint16(raw); err == nil {
			for index, bits := range raw {
				destination[index] = Float16ToFloat32(bits)
			}
		}
	case BFloat16:
		var raw = make([]uint16, len(destination))
		if err = state.buffer.DownloadUint16(raw); err == nil {
			for index, bits := range raw {
				destination[index] = BFloat16ToFloat32(bits)
			}
		}
	case Q8_0:
		var raw = make([]byte, state.bytes)
		if err = state.buffer.DownloadBytes(raw); err == nil {
			dequantizeQ8_0Into(raw, destination)
		}
	case Q4_0:
		var raw = make([]byte, state.bytes)
		if err = state.buffer.DownloadBytes(raw); err == nil {
			dequantizeQ4_0Into(raw, destination)
		}
	}
	state.endTransfer(false, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: download %s as float32: %w", state.dtype, err)
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

func (t *Tensor) beginByteTransfer(length int, write bool) (state *tensorState, err error) {
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
	if !state.dtype.Valid() {
		return nil, fmt.Errorf("%w: transfer incompatible with %s tensor", ErrInvalid, state.dtype)
	}
	if uint64(length) != state.bytes {
		return nil, fmt.Errorf("%w: byte transfer length is %d, want %d", ErrInvalid, length, state.bytes)
	}
	if !write && !state.initialized {
		return nil, ErrUninitialized
	}

	state.hostAccess = true
	return state, nil
}

func (t *Tensor) beginTransfer(length int, write bool, acceptDType func(DType) bool) (state *tensorState, err error) {
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
	if !acceptDType(state.dtype) {
		return nil, fmt.Errorf("%w: transfer incompatible with %s tensor", ErrInvalid, state.dtype)
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
