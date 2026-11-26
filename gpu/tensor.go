package gpu

import (
	"fmt"
	"slices"
)

func newTensor(device *deviceState, dtype DType, shape []int64, count, bytes uint64, buffer allocation) (out *Tensor) {
	var storage tensorStorage
	storage.device = device
	storage.dtype = dtype
	storage.count = count
	storage.bytes = bytes
	storage.buffer = buffer
	storage.liveHandles = 1

	var state tensorState
	state.storage = &storage
	state.shape = slices.Clone(shape)
	return &Tensor{state: &state}
}

// Tensor references one contiguous row-major allocation on its creating device.
// Copies of a Tensor handle refer to the same view handle and lifetime. Views
// created via View share the underlying device allocation, lease tracking, and
// initialization state while maintaining independent shapes and Close states.
type Tensor struct {
	state *tensorState
}

type tensorStorage struct {
	device *deviceState
	dtype  DType
	count  uint64
	bytes  uint64
	buffer allocation

	readers     int
	writer      *scopeState
	hostAccess  bool
	initialized bool
	liveHandles int
}

type tensorState struct {
	storage *tensorStorage
	shape   []int64 // Immutable. The creating device protects closed and storage state.
	closed  bool
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
	return t.state.storage.dtype
}

// Bytes returns the immutable allocated byte size, including after Close.
func (t *Tensor) Bytes() (bytes uint64) {
	if t == nil || t.state == nil {
		return 0
	}
	return t.state.storage.bytes
}

// View returns a zero-copy Tensor sharing the same underlying device allocation
// with a new contiguous row-major shape of identical element count. Views are
// supported for non-quantized dtypes (Float32, Float16, BFloat16). The returned
// handle has an independent Close lifecycle and shares lease, initialization,
// and alias tracking with its source tensor and sibling views.
func (t *Tensor) View(shape []int64) (out *Tensor, err error) {
	if t == nil || t.state == nil {
		return nil, ErrClosed
	}

	var (
		state   = t.state
		storage = state.storage
	)
	storage.device.mutex.Lock()
	defer storage.device.mutex.Unlock()
	if state.closed {
		return nil, ErrClosed
	}
	if !storage.dtype.Valid() {
		return nil, fmt.Errorf("gpu: view tensor: %w: unsupported tensor dtype %s", ErrInvalid, storage.dtype)
	}
	if storage.dtype.IsQuantized() {
		return nil, fmt.Errorf("gpu: view tensor: %w: quantized %s tensors do not support views", ErrInvalid, storage.dtype)
	}

	var count uint64
	if count, _, err = tensorLayout(shape, storage.dtype, storage.device.maxBufferLength); err != nil {
		return nil, fmt.Errorf("gpu: view tensor: %w", err)
	}
	if count != storage.count {
		return nil, fmt.Errorf("gpu: view tensor: %w: view element count %d does not match tensor element count %d", ErrInvalid, count, storage.count)
	}

	storage.liveHandles++
	var view tensorState
	view.storage = storage
	view.shape = slices.Clone(shape)
	return &Tensor{state: &view}, nil
}

// Upload copies a full set of FP32 values before returning. It returns
// ErrInvalid for non-Float32 tensors and ErrBusy if another host transfer or a
// recording or submitted scope holds the tensor.
func (t *Tensor) Upload(values []float32) (err error) {
	var storage *tensorStorage
	if storage, err = t.beginTransfer(len(values), true, func(dtype DType) bool { return dtype == Float32 }); err != nil {
		return err
	}

	err = storage.buffer.Upload(values)
	storage.endTransfer(true, err == nil)
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
	var storage *tensorStorage
	if storage, err = t.beginTransfer(len(destination), false, func(dtype DType) bool { return dtype == Float32 }); err != nil {
		return err
	}

	err = storage.buffer.Download(destination)
	storage.endTransfer(false, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: download tensor: %w", err)
	}
	return nil
}

// UploadBits copies a full set of raw 16-bit element bits into a Float16 or
// BFloat16 tensor without conversion. It returns ErrInvalid for Float32
// tensors or length mismatches.
func (t *Tensor) UploadBits(values []uint16) (err error) {
	var storage *tensorStorage
	if storage, err = t.beginTransfer(len(values), true, DType.is16Bit); err != nil {
		return err
	}

	err = storage.buffer.UploadUint16(values)
	storage.endTransfer(true, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: upload tensor bits: %w", err)
	}
	return nil
}

// DownloadBitsInto copies all raw 16-bit element bits from a Float16 or
// BFloat16 tensor into destination without conversion. It returns ErrInvalid
// for Float32 tensors or length mismatches.
func (t *Tensor) DownloadBitsInto(destination []uint16) (err error) {
	var storage *tensorStorage
	if storage, err = t.beginTransfer(len(destination), false, DType.is16Bit); err != nil {
		return err
	}

	err = storage.buffer.DownloadUint16(destination)
	storage.endTransfer(false, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: download tensor bits: %w", err)
	}
	return nil
}

// UploadBytes copies a full raw byte buffer into the tensor without conversion.
// The slice length must equal Tensor.Bytes(). This supports all valid DTypes,
// including pre-packed Q8_0, Q4_0, and Q4_1 weight blocks.
func (t *Tensor) UploadBytes(values []byte) (err error) {
	var storage *tensorStorage
	if storage, err = t.beginByteTransfer(len(values), true); err != nil {
		return err
	}

	err = storage.buffer.UploadBytes(values)
	storage.endTransfer(true, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: upload tensor bytes: %w", err)
	}
	return nil
}

// DownloadBytesInto copies the full raw byte contents of the tensor into
// destination without conversion. The slice length must equal Tensor.Bytes().
func (t *Tensor) DownloadBytesInto(destination []byte) (err error) {
	var storage *tensorStorage
	if storage, err = t.beginByteTransfer(len(destination), false); err != nil {
		return err
	}

	err = storage.buffer.DownloadBytes(destination)
	storage.endTransfer(false, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: download tensor bytes: %w", err)
	}
	return nil
}

// UploadFloat32As converts FP32 values on the host to the tensor's DType
// (using IEEE-754 round-to-nearest-even for Float16/BFloat16 and block
// quantization for Q8_0/Q4_0/Q4_1) and copies the full set into the tensor. For
// Float32 tensors, values are uploaded directly without conversion.
func (t *Tensor) UploadFloat32As(values []float32) (err error) {
	var storage *tensorStorage
	if storage, err = t.beginTransfer(len(values), true, DType.Valid); err != nil {
		return err
	}

	switch storage.dtype {
	case Float32:
		err = storage.buffer.Upload(values)
	case Float16:
		var encoded = make([]uint16, len(values))
		for index, value := range values {
			encoded[index] = Float32ToFloat16(value)
		}
		err = storage.buffer.UploadUint16(encoded)
	case BFloat16:
		var encoded = make([]uint16, len(values))
		for index, value := range values {
			encoded[index] = Float32ToBFloat16(value)
		}
		err = storage.buffer.UploadUint16(encoded)
	case Q8_0:
		var encoded []byte
		if encoded, err = QuantizeQ8_0(values); err == nil {
			err = storage.buffer.UploadBytes(encoded)
		}
	case Q4_0:
		var encoded []byte
		if encoded, err = QuantizeQ4_0(values); err == nil {
			err = storage.buffer.UploadBytes(encoded)
		}
	case Q4_1:
		var encoded []byte
		if encoded, err = QuantizeQ4_1(values); err == nil {
			err = storage.buffer.UploadBytes(encoded)
		}
	}
	storage.endTransfer(true, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: upload float32 as %s: %w", storage.dtype, err)
	}
	return nil
}

// DownloadFloat32Into downloads all tensor elements and converts or dequantizes
// them to FP32 into destination. For Float32 tensors, values are downloaded directly.
func (t *Tensor) DownloadFloat32Into(destination []float32) (err error) {
	var storage *tensorStorage
	if storage, err = t.beginTransfer(len(destination), false, DType.Valid); err != nil {
		return err
	}

	switch storage.dtype {
	case Float32:
		err = storage.buffer.Download(destination)
	case Float16:
		var raw = make([]uint16, len(destination))
		if err = storage.buffer.DownloadUint16(raw); err == nil {
			for index, bits := range raw {
				destination[index] = Float16ToFloat32(bits)
			}
		}
	case BFloat16:
		var raw = make([]uint16, len(destination))
		if err = storage.buffer.DownloadUint16(raw); err == nil {
			for index, bits := range raw {
				destination[index] = BFloat16ToFloat32(bits)
			}
		}
	case Q8_0:
		var raw = make([]byte, storage.bytes)
		if err = storage.buffer.DownloadBytes(raw); err == nil {
			dequantizeQ8_0Into(raw, destination)
		}
	case Q4_0:
		var raw = make([]byte, storage.bytes)
		if err = storage.buffer.DownloadBytes(raw); err == nil {
			dequantizeQ4_0Into(raw, destination)
		}
	case Q4_1:
		var raw = make([]byte, storage.bytes)
		if err = storage.buffer.DownloadBytes(raw); err == nil {
			dequantizeQ4_1Into(raw, destination)
		}
	}
	storage.endTransfer(false, err == nil)
	if err != nil {
		return fmt.Errorf("gpu: download %s as float32: %w", storage.dtype, err)
	}
	return nil
}

// Close prevents new work on this tensor handle and defers underlying buffer
// release until all sibling views close and existing transfers and command
// scopes finish using it. Close is idempotent.
func (t *Tensor) Close() (err error) {
	if t == nil || t.state == nil {
		return nil
	}

	var (
		state   = t.state
		storage = state.storage
	)
	storage.device.mutex.Lock()
	defer storage.device.mutex.Unlock()
	if !state.closed {
		state.closed = true
		storage.liveHandles--
		storage.releaseIfIdle()
	}
	return nil
}

func (t *Tensor) beginByteTransfer(length int, write bool) (storage *tensorStorage, err error) {
	if t == nil || t.state == nil {
		return nil, ErrClosed
	}

	var state = t.state
	storage = state.storage
	storage.device.mutex.Lock()
	defer storage.device.mutex.Unlock()
	if state.closed {
		return nil, ErrClosed
	}
	if storage.hostAccess || storage.readers > 0 || storage.writer != nil {
		return nil, ErrBusy
	}
	if !storage.dtype.Valid() {
		return nil, fmt.Errorf("%w: transfer incompatible with %s tensor", ErrInvalid, storage.dtype)
	}
	if uint64(length) != storage.bytes {
		return nil, fmt.Errorf("%w: byte transfer length is %d, want %d", ErrInvalid, length, storage.bytes)
	}
	if !write && !storage.initialized {
		return nil, ErrUninitialized
	}

	storage.hostAccess = true
	return storage, nil
}

func (t *Tensor) beginTransfer(length int, write bool, acceptDType func(DType) bool) (storage *tensorStorage, err error) {
	if t == nil || t.state == nil {
		return nil, ErrClosed
	}

	var state = t.state
	storage = state.storage
	storage.device.mutex.Lock()
	defer storage.device.mutex.Unlock()
	if state.closed {
		return nil, ErrClosed
	}
	if storage.hostAccess || storage.readers > 0 || storage.writer != nil {
		return nil, ErrBusy
	}
	if !acceptDType(storage.dtype) {
		return nil, fmt.Errorf("%w: transfer incompatible with %s tensor", ErrInvalid, storage.dtype)
	}
	if uint64(length) != storage.count {
		return nil, fmt.Errorf("%w: transfer length is %d, want %d", ErrInvalid, length, storage.count)
	}
	if !write && !storage.initialized {
		return nil, ErrUninitialized
	}

	storage.hostAccess = true
	return storage, nil
}

func (s *tensorStorage) endTransfer(write, success bool) {
	s.device.mutex.Lock()
	defer s.device.mutex.Unlock()
	if write {
		s.initialized = success
	}
	s.hostAccess = false
	s.releaseIfIdle()
}

func (s *tensorStorage) releaseIfIdle() {
	if s.liveHandles == 0 && s.buffer != nil && !s.hostAccess && s.readers == 0 && s.writer == nil {
		s.buffer.Release()
		s.buffer = nil
		s.device.children--
	}
}
