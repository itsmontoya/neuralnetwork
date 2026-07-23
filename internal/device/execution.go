package device

import (
	"errors"
	"fmt"
	"reflect"
	"sync"
)

const (
	executionKernelLimit         = 64
	executionTransientBytesLimit = 64 << 20
)

// NewSharedExecution constructs a lazy execution using the shared backend.
func NewSharedExecution() (execution *Execution, available bool, err error) {
	var runtimeValue *Runtime

	if runtimeValue, available, err = SharedRuntime(); err != nil {
		return nil, false, err
	}
	if !available {
		return nil, false, nil
	}

	execution = NewExecution(runtimeValue)
	return execution, true, nil
}

// NewExecution constructs a lazy execution for runtimeValue.
func NewExecution(runtimeValue *Runtime) (execution *Execution) {
	if runtimeValue == nil {
		return nil
	}

	var value Execution
	value.runtime = runtimeValue
	return &value
}

// Execution owns bounded command scopes and their pending publications.
type Execution struct {
	mutex          sync.Mutex
	runtime        *Runtime
	scope          *Scope
	publications   []Publication
	bindings       []any
	reads          []any
	validations    []executionValidation
	kernelCount    uint64
	transientBytes uint64
	activated      bool
	closed         bool
	err            error
	snapshot       ExecutionSnapshot
}

// Reset prepares a completed execution for reuse with runtimeValue.
func (e *Execution) Reset(runtimeValue *Runtime) (err error) {
	if e == nil {
		err = errors.New("device: reset execution: execution is nil")
		return err
	}
	if runtimeValue == nil {
		err = errors.New("device: reset execution: runtime is nil")
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if !e.closed || e.scope != nil || len(e.publications) != 0 || len(e.bindings) != 0 {
		err = errors.New("device: reset execution before cleanup completed")
		return err
	}

	e.runtime = runtimeValue
	e.kernelCount = 0
	e.transientBytes = 0
	e.activated = false
	e.closed = false
	e.err = nil
	e.snapshot = ExecutionSnapshot{}
	clear(e.validations)
	e.validations = e.validations[:0]
	return nil
}

// Runtime returns the backend runtime used by the execution.
func (e *Execution) Runtime() (runtimeValue *Runtime) {
	if e == nil {
		return nil
	}

	e.mutex.Lock()
	runtimeValue = e.runtime
	e.mutex.Unlock()
	return runtimeValue
}

// Bind attaches the execution to an opaque matrix value.
func (e *Execution) Bind(value any) (err error) {
	_, err = e.bind(value)
	return err
}

// MarkRead records that the current command scope consumes value.
func (e *Execution) MarkRead(value any) (err error) {
	var (
		key     any
		current any
	)

	if e == nil {
		return nil
	}
	if key, err = e.bind(value); err != nil {
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	for _, current = range e.reads {
		if current == key {
			return nil
		}
	}
	e.reads = append(e.reads, key)
	return nil
}

// Uses reports whether the current command scope reads value.
func (e *Execution) Uses(value any) (used bool, err error) {
	var (
		key     any
		current any
	)

	if e == nil {
		return false, nil
	}
	if key, err = e.bind(value); err != nil {
		return false, err
	}

	e.mutex.Lock()
	for _, current = range e.reads {
		if current == key {
			used = true
			break
		}
	}
	e.mutex.Unlock()
	return used, nil
}

func (e *Execution) bind(value any) (key any, err error) {
	var (
		adapter ExecutionAdapter
		current any
	)

	if e == nil {
		return nil, nil
	}
	if value == nil {
		err = errors.New("device: bind execution value is nil")
		return nil, err
	}
	if adapter, err = currentExecutionAdapter(); err != nil {
		return nil, err
	}
	if key, err = adapter.Bind(value, e); err != nil {
		return nil, fmt.Errorf("device: bind execution value: %w", err)
	}
	if key == nil || !reflect.TypeOf(key).Comparable() {
		adapter.Unbind(key, e)
		err = errors.New("device: execution adapter returned an invalid binding key")
		return nil, err
	}

	e.mutex.Lock()
	if e.closed {
		e.mutex.Unlock()
		adapter.Unbind(key, e)
		err = errors.New("device: bind value to closed execution")
		return nil, err
	}
	for _, current = range e.bindings {
		if current == key {
			e.mutex.Unlock()
			return key, nil
		}
	}
	e.bindings = append(e.bindings, key)
	e.snapshot.BoundValues++
	if uint64(len(e.bindings)) > e.snapshot.PeakBoundValues {
		e.snapshot.PeakBoundValues = uint64(len(e.bindings))
	}
	e.mutex.Unlock()
	return key, nil
}

// Active reports whether the execution can accept more work.
func (e *Execution) Active() (active bool) {
	if e == nil {
		return false
	}

	e.mutex.Lock()
	active = !e.closed
	e.mutex.Unlock()
	return active
}

// Activated reports whether this execution has encoded device work.
func (e *Execution) Activated() (activated bool) {
	if e == nil {
		return false
	}

	e.mutex.Lock()
	activated = e.activated
	e.mutex.Unlock()
	return activated
}

// Error returns the first operational error retained by the execution.
func (e *Execution) Error() (err error) {
	if e == nil {
		return nil
	}

	e.mutex.Lock()
	err = e.err
	e.mutex.Unlock()
	return err
}

// RecordDevicePreparation adds buffer allocation and upload diagnostics.
func (e *Execution) RecordDevicePreparation(allocated, uploaded bool, bytes uint64) {
	if e == nil {
		return
	}

	e.mutex.Lock()
	if allocated {
		e.snapshot.BufferCreations++
	}
	if uploaded {
		e.snapshot.InputUploads++
		e.snapshot.InputUploadBytes += bytes
	}
	e.mutex.Unlock()
}

// RecordDownload adds one host download diagnostic.
func (e *Execution) RecordDownload(bytes uint64) {
	if e == nil {
		return
	}

	e.mutex.Lock()
	e.snapshot.ResultDownloads++
	e.snapshot.ResultDownloadBytes += bytes
	e.mutex.Unlock()
}

// RecordValidation retains one successful value revision validation.
func (e *Execution) RecordValidation(value any, revision uint64) (err error) {
	var (
		key        any
		validation executionValidation
	)

	if e == nil {
		err = errors.New("device: record validation execution is nil")
		return err
	}
	if key, err = e.bind(value); err != nil {
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	for _, validation = range e.validations {
		if validation.key == key && validation.revision == revision {
			return nil
		}
	}
	validation.key = key
	validation.revision = revision
	e.validations = append(e.validations, validation)
	return nil
}

// Validated reports whether value's logical revision passed validation.
func (e *Execution) Validated(value any, revision uint64) (validated bool, err error) {
	var (
		key        any
		validation executionValidation
	)

	if e == nil {
		return false, nil
	}
	if key, err = e.bind(value); err != nil {
		return false, err
	}

	e.mutex.Lock()
	for _, validation = range e.validations {
		if validation.key == key && validation.revision == revision {
			validated = true
			break
		}
	}
	e.mutex.Unlock()
	return validated, nil
}

// CanEncodeAtomic reports whether one unbroken publication batch fits the execution limits.
func (e *Execution) CanEncodeAtomic(kernelCount, transientBytes uint64) (ok bool) {
	if e == nil || kernelCount == 0 {
		return false
	}

	e.mutex.Lock()
	ok = !e.closed &&
		e.err == nil &&
		e.kernelCount <= executionKernelLimit &&
		kernelCount <= executionKernelLimit-e.kernelCount &&
		e.transientBytes <= executionTransientBytesLimit &&
		transientBytes <= executionTransientBytesLimit-e.transientBytes
	e.mutex.Unlock()
	return ok
}

// EncodeCopy appends a device copy and its destination publication.
func (e *Execution) EncodeCopy(
	source,
	destination *Buffer,
	transientBytes uint64,
	publication Publication,
) (err error) {
	if e == nil {
		err = errors.New("device: encode copy execution is nil")
		return err
	}
	if err = publication.validate(); err != nil {
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if err = e.prepareLocked(transientBytes, publication); err != nil {
		return err
	}
	if err = e.scope.EncodeCopy(source, destination); err != nil {
		err = fmt.Errorf("device: execution encode copy: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.recordEncodeLocked(transientBytes)
	return nil
}

// EncodeFill appends a device fill and its destination publication.
func (e *Execution) EncodeFill(
	destination *Buffer,
	value float32,
	transientBytes uint64,
	publication Publication,
) (err error) {
	if e == nil {
		err = errors.New("device: encode fill execution is nil")
		return err
	}
	if err = publication.validate(); err != nil {
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if err = e.prepareLocked(transientBytes, publication); err != nil {
		return err
	}
	if err = e.scope.EncodeFill(destination, value); err != nil {
		err = fmt.Errorf("device: execution encode fill: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.recordEncodeLocked(transientBytes)
	return nil
}

// EncodeDependentFill appends an in-place fill to a pending publication.
func (e *Execution) EncodeDependentFill(destination *Buffer, value float32) (err error) {
	if e == nil {
		err = errors.New("device: encode dependent fill execution is nil")
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if err = e.prepareDependentLocked(); err != nil {
		return err
	}
	if err = e.scope.EncodeFill(destination, value); err != nil {
		err = fmt.Errorf("device: execution encode dependent fill: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.recordEncodeLocked(0)
	return nil
}

// EncodeAddRowVector appends a dependent in-place row-vector addition.
//
// The values buffer must belong to a pending publication already encoded in
// the current command scope.
func (e *Execution) EncodeAddRowVector(
	values,
	rowVector *Buffer,
	rows,
	cols uint32,
) (err error) {
	if e == nil {
		err = errors.New("device: encode row-vector addition execution is nil")
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if err = e.prepareDependentLocked(); err != nil {
		return err
	}
	if err = e.scope.EncodeAddRowVector(values, rowVector, rows, cols); err != nil {
		err = fmt.Errorf("device: execution encode row-vector addition: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.recordEncodeLocked(0)
	return nil
}

// EncodeAddScaled appends scaled elementwise addition and its destination publication.
func (e *Execution) EncodeAddScaled(
	left,
	right,
	result *Buffer,
	scale float32,
	transientBytes uint64,
	publication Publication,
) (err error) {
	if e == nil {
		err = errors.New("device: encode scaled addition execution is nil")
		return err
	}
	if err = publication.validate(); err != nil {
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if err = e.prepareLocked(transientBytes, publication); err != nil {
		return err
	}
	if err = e.scope.EncodeAddScaled(left, right, result, scale); err != nil {
		err = fmt.Errorf("device: execution encode scaled addition: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.recordEncodeLocked(transientBytes)
	return nil
}

// EncodeReLU appends a ReLU forward operation and destination publication.
func (e *Execution) EncodeReLU(
	input,
	result *Buffer,
	transientBytes uint64,
	publication Publication,
) (err error) {
	if e == nil {
		err = errors.New("device: encode ReLU execution is nil")
		return err
	}
	if err = publication.validate(); err != nil {
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if err = e.prepareLocked(transientBytes, publication); err != nil {
		return err
	}
	if err = e.scope.EncodeReLU(input, result); err != nil {
		err = fmt.Errorf("device: execution encode ReLU: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.recordEncodeLocked(transientBytes)
	return nil
}

// EncodeReLUBackward appends a ReLU derivative and its destination publication.
func (e *Execution) EncodeReLUBackward(
	input,
	outputGradient,
	result *Buffer,
	transientBytes uint64,
	publication Publication,
) (err error) {
	if e == nil {
		err = errors.New("device: encode ReLU backward execution is nil")
		return err
	}
	if err = publication.validate(); err != nil {
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if err = e.prepareLocked(transientBytes, publication); err != nil {
		return err
	}
	if err = e.scope.EncodeReLUBackward(input, outputGradient, result); err != nil {
		err = fmt.Errorf("device: execution encode ReLU backward: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.recordEncodeLocked(transientBytes)
	return nil
}

// EncodeSoftmaxRows appends a stable row-wise Softmax and destination publication.
func (e *Execution) EncodeSoftmaxRows(
	input,
	result *Buffer,
	rows,
	cols uint32,
	transientBytes uint64,
	publication Publication,
) (err error) {
	if e == nil {
		err = errors.New("device: encode Softmax execution is nil")
		return err
	}
	if err = publication.validate(); err != nil {
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if err = e.prepareLocked(transientBytes, publication); err != nil {
		return err
	}
	if err = e.scope.EncodeSoftmaxRows(input, result, rows, cols); err != nil {
		err = fmt.Errorf("device: execution encode Softmax: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.recordEncodeLocked(transientBytes)
	return nil
}

// EncodeSoftmaxRowsBackward appends a Softmax Jacobian product and destination publication.
func (e *Execution) EncodeSoftmaxRowsBackward(
	input,
	outputGradient,
	result *Buffer,
	rows,
	cols uint32,
	transientBytes uint64,
	publication Publication,
) (err error) {
	if e == nil {
		err = errors.New("device: encode Softmax backward execution is nil")
		return err
	}
	if err = publication.validate(); err != nil {
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if err = e.prepareLocked(transientBytes, publication); err != nil {
		return err
	}
	if err = e.scope.EncodeSoftmaxRowsBackward(
		input,
		outputGradient,
		result,
		rows,
		cols,
	); err != nil {
		err = fmt.Errorf("device: execution encode Softmax backward: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.recordEncodeLocked(transientBytes)
	return nil
}

// EncodeColumnSums appends a column reduction and destination publication.
func (e *Execution) EncodeColumnSums(
	input,
	result *Buffer,
	rows,
	cols uint32,
	transientBytes uint64,
	publication Publication,
) (err error) {
	if e == nil {
		err = errors.New("device: encode column sums execution is nil")
		return err
	}
	if err = publication.validate(); err != nil {
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if err = e.prepareLocked(transientBytes, publication); err != nil {
		return err
	}
	if err = e.scope.EncodeColumnSums(input, result, rows, cols, false); err != nil {
		err = fmt.Errorf("device: execution encode column sums: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.recordEncodeLocked(transientBytes)
	return nil
}

// EncodeAccumulateColumnSums appends an in-place reduction to pending staging.
func (e *Execution) EncodeAccumulateColumnSums(
	input,
	result *Buffer,
	rows,
	cols uint32,
) (err error) {
	if e == nil {
		err = errors.New("device: encode accumulated column sums execution is nil")
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if err = e.prepareDependentLocked(); err != nil {
		return err
	}
	if err = e.scope.EncodeColumnSums(input, result, rows, cols, true); err != nil {
		err = fmt.Errorf("device: execution encode accumulated column sums: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.recordEncodeLocked(0)
	return nil
}

// EncodeCategoricalCrossEntropy appends a validated scalar loss reduction.
func (e *Execution) EncodeCategoricalCrossEntropy(
	predictions,
	targets,
	result *Buffer,
	rows,
	cols uint32,
	epsilon float32,
	transientBytes uint64,
	publication Publication,
) (err error) {
	if e == nil {
		err = errors.New("device: encode categorical cross entropy execution is nil")
		return err
	}
	if err = publication.validate(); err != nil {
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if err = e.prepareLocked(transientBytes, publication); err != nil {
		return err
	}
	if err = e.scope.EncodeCategoricalCrossEntropy(
		predictions,
		targets,
		result,
		rows,
		cols,
		epsilon,
	); err != nil {
		err = fmt.Errorf("device: execution encode categorical cross entropy: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.recordEncodeLocked(transientBytes)
	return nil
}

// EncodeCategoricalCrossEntropyGradient appends a mean gradient and publication.
func (e *Execution) EncodeCategoricalCrossEntropyGradient(
	predictions,
	targets,
	result *Buffer,
	rows,
	cols uint32,
	epsilon float32,
	transientBytes uint64,
	publication Publication,
) (err error) {
	if e == nil {
		err = errors.New("device: encode categorical cross entropy gradient execution is nil")
		return err
	}
	if err = publication.validate(); err != nil {
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if err = e.prepareLocked(transientBytes, publication); err != nil {
		return err
	}
	if err = e.scope.EncodeCategoricalCrossEntropyGradient(
		predictions,
		targets,
		result,
		rows,
		cols,
		epsilon,
	); err != nil {
		err = fmt.Errorf("device: execution encode categorical cross entropy gradient: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.recordEncodeLocked(transientBytes)
	return nil
}

// EncodeMatMul appends a multiplication and its destination publication.
func (e *Execution) EncodeMatMul(
	left,
	right,
	result *Buffer,
	leftRows,
	leftCols,
	rightRows,
	rightCols,
	resultRows,
	resultCols uint32,
	operation Operation,
	transientBytes uint64,
	publication Publication,
) (err error) {
	if e == nil {
		err = errors.New("device: encode matrix multiplication execution is nil")
		return err
	}
	if err = publication.validate(); err != nil {
		return err
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if err = e.prepareLocked(transientBytes, publication); err != nil {
		return err
	}
	if err = e.scope.EncodeMatMul(
		left,
		right,
		result,
		leftRows,
		leftCols,
		rightRows,
		rightCols,
		resultRows,
		resultCols,
		operation,
	); err != nil {
		err = fmt.Errorf("device: execution encode matrix multiplication: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.recordEncodeLocked(transientBytes)
	return nil
}

// Barrier completes the current command scope without closing the execution.
func (e *Execution) Barrier(reason Boundary) (err error) {
	if e == nil {
		return nil
	}

	e.mutex.Lock()
	defer e.mutex.Unlock()
	if e.closed {
		err = errors.New("device: barrier on closed execution")
		return err
	}
	if e.err != nil {
		return e.err
	}
	if e.scope == nil {
		return nil
	}

	e.snapshot.Barriers++
	if reason == BoundaryCPUFallback || reason == BoundaryHostObservation || reason == BoundaryHostMutation {
		e.snapshot.FallbackBarriers++
	}
	err = e.flushLocked()
	return err
}

// Finish completes pending work, records diagnostics, and detaches all values.
func (e *Execution) Finish() (err error) {
	if e == nil {
		return nil
	}

	e.mutex.Lock()
	if e.closed {
		err = errors.New("device: finish closed execution")
		e.mutex.Unlock()
		return err
	}
	if e.err != nil {
		err = e.err
	} else if e.scope != nil {
		e.snapshot.Barriers++
		err = e.flushLocked()
	}
	e.closed = true
	e.mutex.Unlock()

	if detachErr := e.detachBindings(); detachErr != nil {
		err = errors.Join(err, detachErr)
	}
	e.recordSnapshot()
	return err
}

// Abort discards uncommitted publications and detaches all bound values.
func (e *Execution) Abort(cause error) (err error) {
	if e == nil {
		return nil
	}
	if cause == nil {
		cause = errors.New("device: execution aborted")
	}

	e.mutex.Lock()
	if e.closed {
		e.mutex.Unlock()
		return nil
	}
	err = e.discardBatchLocked(cause)
	e.closed = true
	e.mutex.Unlock()

	if detachErr := e.detachBindings(); detachErr != nil {
		err = errors.Join(err, detachErr)
	}
	e.recordSnapshot()
	return err
}

// Snapshot returns the execution's private batching counters.
func (e *Execution) Snapshot() (snapshot ExecutionSnapshot) {
	if e == nil {
		return snapshot
	}

	e.mutex.Lock()
	snapshot = e.snapshot
	e.mutex.Unlock()
	return snapshot
}

func (e *Execution) prepareLocked(transientBytes uint64, publication Publication) (err error) {
	if e.closed {
		err = errors.New("device: encode command on closed execution")
		err = errors.Join(err, publication.Discard(err))
		return err
	}
	if e.err != nil {
		err = errors.Join(e.err, publication.Discard(e.err))
		return err
	}
	if e.runtime == nil {
		err = errors.New("device: execution runtime is nil")
		err = errors.Join(err, publication.Discard(err))
		return err
	}

	if e.scope != nil && e.kernelCount > 0 &&
		(e.kernelCount >= executionKernelLimit ||
			transientBytes > executionTransientBytesLimit ||
			e.transientBytes > executionTransientBytesLimit-transientBytes) {
		e.snapshot.Barriers++
		if err = e.flushLocked(); err != nil {
			err = errors.Join(err, publication.Discard(err))
			return err
		}
	}
	if e.scope == nil {
		if e.scope, err = e.runtime.NewScope(); err != nil {
			err = fmt.Errorf("device: execution create command scope: %w", err)
			err = errors.Join(err, publication.Discard(err))
			e.err = err
			return err
		}
	}

	e.publications = append(e.publications, publication)
	return nil
}

func (e *Execution) prepareDependentLocked() (err error) {
	if e.closed {
		err = errors.New("device: encode dependent command on closed execution")
		return err
	}
	if e.err != nil {
		return e.err
	}
	if e.runtime == nil {
		err = errors.New("device: execution runtime is nil")
		return err
	}
	if e.scope == nil || len(e.publications) == 0 {
		err = errors.New("device: dependent command requires a pending publication")
		return err
	}

	return nil
}

func (e *Execution) recordEncodeLocked(transientBytes uint64) {
	e.kernelCount++
	e.transientBytes += transientBytes
	e.activated = true
	e.snapshot.KernelEncodes++
	e.snapshot.TransientBytes += transientBytes
	if e.transientBytes > e.snapshot.PeakTransientBytes {
		e.snapshot.PeakTransientBytes = e.transientBytes
	}
}

func (e *Execution) flushLocked() (err error) {
	var (
		index       int
		publication Publication
		releaseErr  error
	)

	if e.scope == nil {
		return nil
	}
	if err = e.scope.Commit(); err != nil {
		err = fmt.Errorf("device: execution commit commands: %w", err)
		e.failBatchLocked(err)
		return err
	}
	e.snapshot.CommandSubmissions++
	if err = e.scope.Wait(); err != nil {
		err = fmt.Errorf("device: execution wait for commands: %w", err)
		e.snapshot.Waits++
		e.failBatchLocked(err)
		return err
	}
	e.snapshot.Waits++

	for index, publication = range e.publications {
		if err = publication.Publish(); err != nil {
			err = fmt.Errorf("device: publish completed write: %w", err)
			err = errors.Join(err, publication.Discard(err))
			e.snapshot.DiscardedWrites += uint64(len(e.publications) - index)
			for _, publication = range e.publications[index+1:] {
				err = errors.Join(err, publication.Discard(err))
			}
			e.err = err
			break
		}
		e.snapshot.Publications++
	}
	clear(e.publications)
	e.publications = e.publications[:0]
	if releaseErr = e.scope.Release(); releaseErr != nil {
		releaseErr = fmt.Errorf("device: release completed command scope: %w", releaseErr)
		err = errors.Join(err, releaseErr)
		e.err = err
	}
	e.scope = nil
	e.kernelCount = 0
	e.transientBytes = 0
	clear(e.reads)
	e.reads = e.reads[:0]
	return err
}

func (e *Execution) failBatchLocked(cause error) {
	var discardErr error

	discardErr = e.discardBatchLocked(cause)
	e.err = errors.Join(cause, discardErr)
}

func (e *Execution) discardBatchLocked(cause error) (err error) {
	var publication Publication

	for _, publication = range e.publications {
		if discardErr := publication.Discard(cause); discardErr != nil {
			err = errors.Join(err, discardErr)
		}
		e.snapshot.DiscardedWrites++
	}
	clear(e.publications)
	e.publications = e.publications[:0]
	if e.scope != nil {
		if releaseErr := e.scope.Release(); releaseErr != nil {
			err = errors.Join(err, releaseErr)
		}
	}
	e.scope = nil
	e.kernelCount = 0
	e.transientBytes = 0
	clear(e.reads)
	e.reads = e.reads[:0]
	return err
}

func (e *Execution) detachBindings() (err error) {
	var (
		adapter  ExecutionAdapter
		bindings []any
		key      any
	)

	e.mutex.Lock()
	bindings = e.bindings
	e.bindings = nil
	e.mutex.Unlock()
	if len(bindings) == 0 {
		return nil
	}
	if adapter, err = currentExecutionAdapter(); err != nil {
		return err
	}
	for _, key = range bindings {
		if unbindErr := adapter.Unbind(key, e); unbindErr != nil {
			err = errors.Join(err, unbindErr)
		}
	}
	clear(bindings)
	e.mutex.Lock()
	e.bindings = bindings[:0]
	e.mutex.Unlock()
	return err
}

func (e *Execution) recordSnapshot() {
	var adapter ExecutionAdapter
	var err error

	if adapter, err = currentExecutionAdapter(); err != nil || adapter.Record == nil {
		return
	}
	adapter.Record(e.Snapshot())
}
