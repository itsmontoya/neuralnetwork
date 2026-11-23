package gpu

import (
	"slices"
	"testing"
)

func Test_TensorTransferOwnership(t *testing.T) {
	var (
		b      fakeBackend
		d      = newDevice(&b, 128)
		shape  = []int64{2, 2}
		tensor = requireTensor(t, d, shape...)
		values = []float32{-3, 0, 4, 1.5}
		result [4]float32
	)

	shape[0] = 1
	tensor.Shape()[1] = 3
	if !slices.Equal(tensor.Shape(), []int64{2, 2}) {
		t.Fatalf("shape was changed: %v", tensor.Shape())
	}
	requireError(t, tensor.DownloadInto(result[:]), ErrUninitialized)
	requireError(t, tensor.DownloadInto(result[:3]), ErrInvalid)
	requireError(t, tensor.Upload(values[:3]), ErrInvalid)
	requireError(t, tensor.Upload(values), nil)
	values[0] = 99
	requireError(t, tensor.DownloadInto(result[:]), nil)
	if result != [4]float32{-3, 0, 4, 1.5} {
		t.Fatalf("host upload alias: %v", result)
	}
	result[1] = 22
	requireError(t, tensor.DownloadInto(result[:]), nil)
	if result != [4]float32{-3, 0, 4, 1.5} {
		t.Fatalf("host download alias: %v", result)
	}

	var alias = *tensor
	requireError(t, alias.Close(), nil)
	requireError(t, tensor.Close(), nil)
	requireError(t, tensor.DownloadInto(result[:]), ErrClosed)
	requireError(t, alias.Upload(values), ErrClosed)
	if !slices.Equal(tensor.Shape(), []int64{2, 2}) {
		t.Fatalf("closed tensor lost shape: %v", tensor.Shape())
	}
	requireError(t, d.Close(), nil)
}

func Test_TensorCloseDuringHostTransfer(t *testing.T) {
	var (
		b      fakeBackend
		d      = newDevice(&b, 128)
		tensor = requireTensor(t, d, 1)
		target = requireTensor(t, d, 1)
		scope  = requireScope(t, d)
		buffer = tensor.state.buffer.(*fakeAllocation)
		gate   = make(chan struct{})
		result = make(chan error, 1)
	)
	buffer.uploadStarted = make(chan struct{})
	buffer.uploadGate = gate
	var unblock = unblockAtCleanup(t, gate)
	go func() { result <- tensor.Upload([]float32{4}) }()
	waitSignal(t, buffer.uploadStarted)

	requireError(t, tensor.DownloadInto(make([]float32, 1)), ErrBusy)
	requireError(t, scope.Copy(tensor, target), ErrBusy)
	requireError(t, tensor.Close(), nil)
	requireError(t, tensor.Upload([]float32{1}), ErrClosed)
	if got := b.liveBuffers.Load(); got != 2 {
		t.Fatalf("live buffers during upload = %d, want 2", got)
	}

	unblock()
	requireError(t, <-result, nil)
	if got := b.liveBuffers.Load(); got != 1 {
		t.Fatalf("live buffers after upload = %d, want 1", got)
	}
	requireError(t, scope.Close(), nil)
	requireError(t, target.Close(), nil)
	requireError(t, d.Close(), nil)
}
