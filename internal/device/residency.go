package device

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
)

type residencyState uint8

const (
	residencyStateNew residencyState = iota
	residencyStateHostNewer
	residencyStateSynchronized
	residencyStateDeviceNewer
	residencyStatePending
	residencyStateFailed
	residencyStatePooled
	residencyStateReleased
)

// NewResidency constructs a host-current private residency record.
func NewResidency(runtimeValue *Runtime, count uint64) (out *Residency, err error) {
	if runtimeValue == nil || runtimeValue.backend == nil {
		err = errors.New("device: residency runtime is nil")
		return nil, err
	}
	if count == 0 {
		err = errors.New("device: residency element count must be positive")
		return nil, err
	}

	var value Residency
	value.runtime = runtimeValue
	value.count = count
	value.logicalRevision = 1
	value.hostRevision = 1
	value.state = residencyStateNew
	out = &value
	runtime.SetFinalizer(out, finalizeResidency)
	return out, nil
}

// Residency owns one matrix's committed device buffer and coherence revisions.
type Residency struct {
	mutex                  sync.Mutex
	runtime                *Runtime
	count                  uint64
	buffer                 *Buffer
	pendingBuffer          *Buffer
	logicalRevision        uint64
	hostRevision           uint64
	deviceRevision         uint64
	pendingRevision        uint64
	state                  residencyState
	stateBeforePending     residencyState
	stateBeforePool        residencyState
	lastError              error
	hostRevisionAdvances   uint64
	deviceRevisionAdvances uint64
	proposedRevisions      uint64
	publications           uint64
	discardedPublications  uint64
	uploads                uint64
	downloads              uint64
	avoidedUploads         uint64
	deviceCopies           uint64
	execution              *Execution
}

// Runtime reports the runtime that owns the residency buffers.
func (r *Residency) Runtime() (runtimeValue *Runtime) {
	if r == nil {
		return nil
	}

	r.mutex.Lock()
	runtimeValue = r.runtime
	r.mutex.Unlock()
	return runtimeValue
}

// BindExecution attaches one active execution to the residency record.
func (r *Residency) BindExecution(execution *Execution) (err error) {
	if r == nil {
		err = errors.New("device: bind execution: residency is nil")
		return err
	}
	if execution == nil {
		err = errors.New("device: bind execution: execution is nil")
		return err
	}

	r.mutex.Lock()
	defer r.mutex.Unlock()
	if r.execution != nil && r.execution != execution {
		err = errors.New("device: residency is bound to another execution")
		return err
	}
	if execution.Runtime() != r.runtime {
		err = errors.New("device: residency execution belongs to another runtime")
		return err
	}
	r.execution = execution
	return nil
}

// Execution returns the active execution bound to the residency record.
func (r *Residency) Execution() (execution *Execution) {
	if r == nil {
		return nil
	}

	r.mutex.Lock()
	execution = r.execution
	r.mutex.Unlock()
	return execution
}

// UnbindExecution removes execution when it still owns the binding.
func (r *Residency) UnbindExecution(execution *Execution) (err error) {
	if r == nil {
		return nil
	}

	r.mutex.Lock()
	defer r.mutex.Unlock()
	if r.execution == nil {
		return nil
	}
	if r.execution != execution {
		err = errors.New("device: unbind execution: binding owner mismatch")
		return err
	}
	r.execution = nil
	return nil
}

// PendingBuffer returns the proposed value owned by execution.
func (r *Residency) PendingBuffer(execution *Execution) (buffer *Buffer, ok bool) {
	if r == nil || execution == nil {
		return nil, false
	}

	r.mutex.Lock()
	if r.execution == execution && r.state == residencyStatePending && r.pendingBuffer != nil {
		buffer = r.pendingBuffer
		ok = true
	}
	r.mutex.Unlock()
	return buffer, ok
}

// EnsureDevice returns a device buffer containing the latest logical values.
func (r *Residency) EnsureDevice(host []float32) (
	buffer *Buffer,
	allocated bool,
	uploaded bool,
	err error,
) {
	if r == nil {
		err = errors.New("device: ensure residency device value: residency is nil")
		return nil, false, false, err
	}

	r.mutex.Lock()
	defer r.mutex.Unlock()
	if err = r.validateHostLength(host); err != nil {
		return nil, false, false, err
	}
	if err = r.requireUsable("ensure device value"); err != nil {
		return nil, false, false, err
	}
	if r.hostRevision != r.logicalRevision && r.deviceRevision != r.logicalRevision {
		err = errors.New("device: ensure device value: no committed current value")
		return nil, false, false, err
	}

	if r.buffer == nil {
		if r.buffer, err = r.runtime.NewBuffer(r.count); err != nil {
			return nil, false, false, fmt.Errorf("device: allocate residency buffer: %w", err)
		}
		allocated = true
	}
	if r.deviceRevision == r.logicalRevision {
		r.avoidedUploads++
		r.state = r.currentState()
		return r.buffer, allocated, false, nil
	}
	if r.hostRevision != r.logicalRevision {
		err = errors.New("device: upload residency buffer: host value is stale")
		return nil, allocated, false, err
	}
	if err = r.buffer.Upload(host); err != nil {
		return nil, allocated, false, fmt.Errorf("device: upload residency buffer: %w", err)
	}

	r.deviceRevision = r.logicalRevision
	r.deviceRevisionAdvances++
	r.uploads++
	r.state = residencyStateSynchronized
	return r.buffer, allocated, true, nil
}

// EnsureHost downloads a device-newer value into complete host storage.
func (r *Residency) EnsureHost(host []float32) (downloaded bool, err error) {
	if r == nil {
		return false, nil
	}

	r.mutex.Lock()
	defer r.mutex.Unlock()
	if err = r.validateHostLength(host); err != nil {
		return false, err
	}
	if err = r.requireUsable("ensure host value"); err != nil {
		return false, err
	}
	if r.hostRevision == r.logicalRevision {
		return false, nil
	}
	if r.deviceRevision != r.logicalRevision || r.buffer == nil {
		err = errors.New("device: ensure host value: device value is not current")
		return false, err
	}
	if err = r.buffer.Download(host); err != nil {
		r.lastError = err
		return false, fmt.Errorf("device: download residency buffer: %w", err)
	}

	r.hostRevision = r.logicalRevision
	r.hostRevisionAdvances++
	r.downloads++
	r.state = residencyStateSynchronized
	return true, nil
}

// MarkHostWrite publishes a completed host mutation and invalidates stale device content.
func (r *Residency) MarkHostWrite() (err error) {
	if r == nil {
		return nil
	}

	r.mutex.Lock()
	defer r.mutex.Unlock()
	if err = r.requireUsable("publish host write"); err != nil {
		return err
	}

	r.rebaseForAdvance()
	r.logicalRevision++
	r.hostRevision = r.logicalRevision
	r.hostRevisionAdvances++
	r.lastError = nil
	r.state = residencyStateHostNewer
	return nil
}

// BeginDeviceWrite allocates staging for a proposed full device write.
func (r *Residency) BeginDeviceWrite() (buffer *Buffer, allocated bool, err error) {
	if r == nil {
		err = errors.New("device: begin residency device write: residency is nil")
		return nil, false, err
	}

	r.mutex.Lock()
	defer r.mutex.Unlock()
	if err = r.requireUsable("begin device write"); err != nil {
		return nil, false, err
	}

	r.rebaseForAdvance()
	r.stateBeforePending = r.state
	if r.pendingBuffer, err = r.runtime.NewBuffer(r.count); err != nil {
		return nil, false, fmt.Errorf("device: allocate residency staging buffer: %w", err)
	}
	r.pendingRevision = r.logicalRevision + 1
	r.proposedRevisions++
	r.state = residencyStatePending
	return r.pendingBuffer, true, nil
}

// PublishDeviceWrite atomically replaces the committed device value with staging.
func (r *Residency) PublishDeviceWrite(buffer *Buffer) (err error) {
	var previous *Buffer

	if r == nil {
		err = errors.New("device: publish residency device write: residency is nil")
		return err
	}

	r.mutex.Lock()
	if r.state != residencyStatePending || r.pendingBuffer == nil || r.pendingBuffer != buffer {
		r.mutex.Unlock()
		err = errors.New("device: publish residency device write: pending buffer mismatch")
		return err
	}

	previous = r.buffer
	r.buffer = r.pendingBuffer
	r.pendingBuffer = nil
	r.logicalRevision = r.pendingRevision
	r.deviceRevision = r.pendingRevision
	r.pendingRevision = 0
	r.deviceRevisionAdvances++
	r.publications++
	r.lastError = nil
	r.state = residencyStateDeviceNewer
	r.stateBeforePending = residencyStateNew
	r.mutex.Unlock()

	if previous != nil {
		previous.Release()
	}
	return nil
}

// FailDeviceWrite discards staging and preserves the last committed revisions.
func (r *Residency) FailDeviceWrite(buffer *Buffer, cause error) (err error) {
	if r == nil {
		err = errors.New("device: fail residency device write: residency is nil")
		return err
	}
	if cause == nil {
		cause = errors.New("device: device write failed")
	}

	r.mutex.Lock()
	if r.state != residencyStatePending || r.pendingBuffer == nil || r.pendingBuffer != buffer {
		r.mutex.Unlock()
		err = errors.New("device: fail residency device write: pending buffer mismatch")
		return err
	}
	r.pendingBuffer = nil
	r.pendingRevision = 0
	r.discardedPublications++
	r.lastError = cause
	r.state = residencyStateFailed
	r.mutex.Unlock()
	buffer.Release()
	return nil
}

// RestoreCommitted clears a reported failure and restores the last committed state.
func (r *Residency) RestoreCommitted() (err error) {
	if r == nil {
		err = errors.New("device: restore residency: residency is nil")
		return err
	}

	r.mutex.Lock()
	defer r.mutex.Unlock()
	if r.state != residencyStateFailed {
		err = errors.New("device: restore residency: residency is not failed")
		return err
	}
	r.state = r.stateBeforePending
	r.stateBeforePending = residencyStateNew
	return nil
}

// RecordDeviceCopy records a completed device-to-device copy for diagnostics.
func (r *Residency) RecordDeviceCopy() {
	if r == nil {
		return
	}

	r.mutex.Lock()
	r.deviceCopies++
	r.mutex.Unlock()
}

// MarkPooled overlays the committed state while scratch storage is idle.
func (r *Residency) MarkPooled() (err error) {
	if r == nil {
		return nil
	}

	r.mutex.Lock()
	defer r.mutex.Unlock()
	if err = r.requireUsable("pool residency"); err != nil {
		return err
	}
	r.stateBeforePool = r.state
	r.state = residencyStatePooled
	return nil
}

// ReusePooled restores a pooled residency's committed coherence state.
func (r *Residency) ReusePooled() (err error) {
	if r == nil {
		return nil
	}

	r.mutex.Lock()
	defer r.mutex.Unlock()
	if r.state != residencyStatePooled {
		err = errors.New("device: reuse residency: residency is not pooled")
		return err
	}
	r.state = r.stateBeforePool
	return nil
}

// Release detaches device storage after host storage has become current.
func (r *Residency) Release() (err error) {
	if r == nil {
		return nil
	}

	r.mutex.Lock()
	if err = r.requireUsable("release residency"); err != nil {
		r.mutex.Unlock()
		return err
	}
	if r.hostRevision != r.logicalRevision {
		r.mutex.Unlock()
		err = errors.New("device: release residency: host value is stale")
		return err
	}
	r.releaseBuffersLocked()
	r.deviceRevision = 0
	r.state = residencyStateReleased
	r.mutex.Unlock()
	return nil
}

// Snapshot returns private coherence and transfer diagnostics.
func (r *Residency) Snapshot() (snapshot ResidencySnapshot) {
	if r == nil {
		return snapshot
	}

	r.mutex.Lock()
	snapshot.State = r.state.String()
	snapshot.LogicalRevision = r.logicalRevision
	snapshot.HostRevision = r.hostRevision
	snapshot.DeviceRevision = r.deviceRevision
	snapshot.PendingRevision = r.pendingRevision
	snapshot.HostRevisionAdvances = r.hostRevisionAdvances
	snapshot.DeviceRevisionAdvances = r.deviceRevisionAdvances
	snapshot.ProposedRevisions = r.proposedRevisions
	snapshot.Publications = r.publications
	snapshot.DiscardedPublications = r.discardedPublications
	snapshot.Uploads = r.uploads
	snapshot.Downloads = r.downloads
	snapshot.AvoidedUploads = r.avoidedUploads
	snapshot.DeviceCopies = r.deviceCopies
	snapshot.HasBuffer = r.buffer != nil
	snapshot.HasPendingBuffer = r.pendingBuffer != nil
	if r.lastError != nil {
		snapshot.LastError = r.lastError.Error()
	}
	r.mutex.Unlock()
	return snapshot
}

func (r *Residency) validateHostLength(host []float32) (err error) {
	if uint64(len(host)) != r.count {
		err = fmt.Errorf("device: residency host length mismatch: got %d, want %d", len(host), r.count)
		return err
	}
	return nil
}

func (r *Residency) requireUsable(operation string) (err error) {
	switch r.state {
	case residencyStatePending:
		err = fmt.Errorf("device: %s: device write is pending", operation)
	case residencyStateFailed:
		err = fmt.Errorf("device: %s: %w", operation, r.lastError)
	case residencyStatePooled:
		err = fmt.Errorf("device: %s: residency is pooled", operation)
	}
	return err
}

func (r *Residency) rebaseForAdvance() {
	if r.logicalRevision != ^uint64(0) {
		return
	}

	if r.hostRevision == r.logicalRevision {
		r.hostRevision = 1
	} else {
		r.hostRevision = 0
	}
	if r.deviceRevision == r.logicalRevision {
		r.deviceRevision = 1
	} else {
		r.deviceRevision = 0
	}
	r.logicalRevision = 1
}

func (r *Residency) currentState() (state residencyState) {
	if r.buffer == nil {
		if r.hostRevision == r.logicalRevision {
			return residencyStateHostNewer
		}
		return residencyStateReleased
	}
	if r.hostRevision == r.logicalRevision && r.deviceRevision == r.logicalRevision {
		return residencyStateSynchronized
	}
	if r.hostRevision == r.logicalRevision {
		return residencyStateHostNewer
	}
	if r.deviceRevision == r.logicalRevision {
		return residencyStateDeviceNewer
	}
	return residencyStateReleased
}

func (r *Residency) releaseBuffersLocked() {
	if r.buffer != nil {
		r.buffer.Release()
		r.buffer = nil
	}
	if r.pendingBuffer != nil {
		r.pendingBuffer.Release()
		r.pendingBuffer = nil
		r.pendingRevision = 0
	}
}

func (r *Residency) discard() {
	if r == nil {
		return
	}

	r.mutex.Lock()
	r.releaseBuffersLocked()
	r.deviceRevision = 0
	r.state = residencyStateReleased
	r.mutex.Unlock()
}

func finalizeResidency(residency *Residency) {
	residency.discard()
}

func (s residencyState) String() (name string) {
	switch s {
	case residencyStateNew:
		name = "new"
	case residencyStateHostNewer:
		name = "host-newer"
	case residencyStateSynchronized:
		name = "synchronized"
	case residencyStateDeviceNewer:
		name = "device-newer"
	case residencyStatePending:
		name = "pending"
	case residencyStateFailed:
		name = "failed"
	case residencyStatePooled:
		name = "pooled"
	case residencyStateReleased:
		name = "released"
	default:
		name = "unknown"
	}
	return name
}
