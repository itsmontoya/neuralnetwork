package device

import (
	"errors"
	"fmt"
	"sync"
)

// Buffer owns one opaque device allocation.
type Buffer struct {
	mutex    sync.Mutex
	runtime  *Runtime
	handle   any
	count    uint64
	bytes    uint64
	released bool
}

// Count returns the buffer's float32 element capacity.
func (b *Buffer) Count() (count uint64) {
	if b == nil {
		return 0
	}

	b.mutex.Lock()
	count = b.count
	b.mutex.Unlock()
	return count
}

// Bytes returns the buffer's allocated byte length.
func (b *Buffer) Bytes() (bytes uint64) {
	if b == nil {
		return 0
	}

	b.mutex.Lock()
	bytes = b.bytes
	b.mutex.Unlock()
	return bytes
}

// Upload copies a complete float32 value set into the buffer.
func (b *Buffer) Upload(values []float32) (err error) {
	var handle any

	if handle, err = b.lockedHandle("upload"); err != nil {
		return err
	}
	defer b.mutex.Unlock()

	if uint64(len(values)) != b.count {
		err = fmt.Errorf("device: upload length mismatch: got %d, want %d", len(values), b.count)
		return err
	}

	if err = b.runtime.backend.upload(handle, values); err != nil {
		return fmt.Errorf("device: upload buffer: %w", err)
	}

	return nil
}

// Download copies the complete buffer value set into destination.
func (b *Buffer) Download(destination []float32) (err error) {
	var handle any

	if handle, err = b.lockedHandle("download"); err != nil {
		return err
	}
	defer b.mutex.Unlock()

	if uint64(len(destination)) != b.count {
		err = fmt.Errorf("device: download length mismatch: got %d, want %d", len(destination), b.count)
		return err
	}

	if err = b.runtime.backend.download(handle, destination); err != nil {
		return fmt.Errorf("device: download buffer: %w", err)
	}

	return nil
}

// UploadBytes copies a complete byte slice into the buffer.
func (b *Buffer) UploadBytes(values []byte) (err error) {
	var handle any

	if handle, err = b.lockedHandle("upload bytes"); err != nil {
		return err
	}
	defer b.mutex.Unlock()

	if uint64(len(values)) != b.bytes {
		err = fmt.Errorf("device: upload bytes length mismatch: got %d, want %d", len(values), b.bytes)
		return err
	}

	if err = b.runtime.backend.uploadBytes(handle, values); err != nil {
		return fmt.Errorf("device: upload buffer bytes: %w", err)
	}

	return nil
}

// DownloadBytes copies the complete buffer bytes into destination.
func (b *Buffer) DownloadBytes(destination []byte) (err error) {
	var handle any

	if handle, err = b.lockedHandle("download bytes"); err != nil {
		return err
	}
	defer b.mutex.Unlock()

	if uint64(len(destination)) != b.bytes {
		err = fmt.Errorf("device: download bytes length mismatch: got %d, want %d", len(destination), b.bytes)
		return err
	}

	if err = b.runtime.backend.downloadBytes(handle, destination); err != nil {
		return fmt.Errorf("device: download buffer bytes: %w", err)
	}

	return nil
}

// UploadUint16 copies a complete uint16 slice into the buffer.
func (b *Buffer) UploadUint16(values []uint16) (err error) {
	var handle any

	if handle, err = b.lockedHandle("upload uint16"); err != nil {
		return err
	}
	defer b.mutex.Unlock()

	if uint64(len(values))*2 != b.bytes {
		err = fmt.Errorf("device: upload uint16 length mismatch: got %d bytes, want %d", uint64(len(values))*2, b.bytes)
		return err
	}

	if err = b.runtime.backend.uploadUint16(handle, values); err != nil {
		return fmt.Errorf("device: upload uint16 buffer: %w", err)
	}

	return nil
}

// DownloadUint16 copies the complete buffer contents into a uint16 slice.
func (b *Buffer) DownloadUint16(destination []uint16) (err error) {
	var handle any

	if handle, err = b.lockedHandle("download uint16"); err != nil {
		return err
	}
	defer b.mutex.Unlock()

	if uint64(len(destination))*2 != b.bytes {
		err = fmt.Errorf("device: download uint16 length mismatch: got %d bytes, want %d", uint64(len(destination))*2, b.bytes)
		return err
	}

	if err = b.runtime.backend.downloadUint16(handle, destination); err != nil {
		return fmt.Errorf("device: download uint16 buffer: %w", err)
	}

	return nil
}

// Release relinquishes the buffer owner's device reference.
func (b *Buffer) Release() {
	if b == nil {
		return
	}

	b.mutex.Lock()
	if b.released {
		b.mutex.Unlock()
		return
	}

	if b.runtime != nil && b.runtime.backend != nil && b.handle != nil {
		b.runtime.backend.releaseBuffer(b.handle)
	}
	b.handle = nil
	b.released = true
	b.mutex.Unlock()
}

func (b *Buffer) lockedHandle(operation string) (handle any, err error) {
	if b == nil {
		err = fmt.Errorf("device: %s buffer: buffer is nil", operation)
		return nil, err
	}

	b.mutex.Lock()
	if b.released {
		b.mutex.Unlock()
		err = fmt.Errorf("device: %s buffer: %w", operation, ErrReleased)
		return nil, err
	}
	if b.runtime == nil || b.runtime.backend == nil || b.handle == nil {
		b.mutex.Unlock()
		err = errors.New("device: buffer has nil runtime or handle")
		return nil, err
	}

	handle = b.handle
	return handle, nil
}
