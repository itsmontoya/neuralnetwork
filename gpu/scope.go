package gpu

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
)

func newScope(device *deviceState, commands commandBuffer) (out *Scope) {
	var state scopeState
	state.device = device
	state.commands = commands
	state.leases = make(map[*tensorState]tensorAccess)
	return &Scope{state: &state}
}

// Scope records ordered operations for one explicit submission. Close releases
// the scope. Copies of a Scope handle refer to the same recording and lifetime.
type Scope struct {
	state *scopeState
}

type scopeStatus uint8

const (
	scopeRecording scopeStatus = iota
	scopePoisoned
	scopeDiscarded
	scopeSubmitted
	scopeFinished
	scopeClosed
)

type tensorAccess uint8

const (
	readAccess tensorAccess = iota + 1
	writeAccess
)

type scopeCompletion struct {
	done chan struct{}
	err  error
}

type scopeState struct {
	mutex      sync.Mutex // Acquire before the device mutex when both are needed.
	device     *deviceState
	commands   commandBuffer
	leases     map[*tensorState]tensorAccess
	status     scopeStatus
	completion *scopeCompletion
	result     error
}

// Fill records a full FP32 write using the given scalar.
func (s *Scope) Fill(destination *Tensor, value float32) (err error) {
	var state *scopeState
	if state, err = s.beginEncoding(); err != nil {
		return err
	}
	defer state.mutex.Unlock()

	if destination == nil || destination.state == nil {
		return state.reject("fill", fmt.Errorf("%w: missing destination", ErrInvalid))
	}
	if destination.state.count > math.MaxUint32 {
		return state.reject("fill", fmt.Errorf("%w: element count exceeds the Metal fill index range", ErrInvalid))
	}
	if err = state.acquire(destination); err != nil {
		return state.reject("fill", err)
	}
	if err = state.commands.encodeFill(destination.state.buffer, value); err != nil {
		state.status = scopePoisoned
		return fmt.Errorf("gpu: encode fill: %w", err)
	}
	return nil
}

// Copy records a full transfer between distinct tensors of identical shape
// on this device. An earlier write in the same scope can initialize source.
func (s *Scope) Copy(source, destination *Tensor) (err error) {
	var state *scopeState
	if state, err = s.beginEncoding(); err != nil {
		return err
	}
	defer state.mutex.Unlock()

	if source == nil || source.state == nil || destination == nil || destination.state == nil {
		return state.reject("copy", fmt.Errorf("%w: missing source or destination", ErrInvalid))
	}
	if !slices.Equal(source.state.shape, destination.state.shape) {
		return state.reject("copy", fmt.Errorf("%w: tensor shapes differ", ErrInvalid))
	}
	if err = state.acquire(destination, source); err != nil {
		return state.reject("copy", err)
	}
	if err = state.commands.encodeCopy(source.state.buffer, destination.state.buffer); err != nil {
		state.status = scopePoisoned
		return fmt.Errorf("gpu: encode copy: %w", err)
	}
	return nil
}

// MatMul records destination[M,N] = left[M,K] × right[K,N]. All tensors must
// be contiguous row-major FP32 matrices with positive dimensions and at most
// 2³²−1 elements each. Use M=1 for a single-row projection. Operands must be
// on this device. The inputs may share an allocation. They must already be
// initialized or written by an earlier operation in the same scope. Destination
// is fully overwritten and must be distinct from both inputs.
//
// Each dot product accumulates in FP32 in contraction order and may use fused
// multiply-add. NaNs propagate, and infinite operands follow IEEE arithmetic,
// including NaN for zero times infinity or addition of opposite infinities.
// NaN payloads, signed zero and subnormal preservation are unspecified.
func (s *Scope) MatMul(left, right, destination *Tensor) (err error) {
	return s.matMul(left, right, destination, false)
}

// MatMulRightTranspose records destination[M,N] = left[M,K] × right[N,K]ᵀ
// without moving or transposing the stored right operand. It uses the same
// row-major output, ownership, shape limits and FP32 arithmetic as MatMul.
func (s *Scope) MatMulRightTranspose(left, right, destination *Tensor) (err error) {
	return s.matMul(left, right, destination, true)
}

func (s *Scope) matMul(left, right, destination *Tensor, rightTransposed bool) (err error) {
	var state *scopeState
	if state, err = s.beginEncoding(); err != nil {
		return err
	}
	defer state.mutex.Unlock()

	for _, operand := range [3]*Tensor{left, right, destination} {
		if operand == nil || operand.state == nil || len(operand.state.shape) != 2 {
			return state.reject("matmul", fmt.Errorf("%w: operands must be rank-two tensors", ErrInvalid))
		}
		if operand.state.count > math.MaxUint32 {
			return state.reject("matmul", fmt.Errorf("%w: tensor element count exceeds the Metal matmul index range", ErrInvalid))
		}
	}

	var (
		m      = left.state.shape[0]
		k      = left.state.shape[1]
		rightK = right.state.shape[0]
		n      = right.state.shape[1]
	)
	if rightTransposed {
		rightK, n = n, rightK
	}
	if k != rightK || destination.state.shape[0] != m || destination.state.shape[1] != n {
		return state.reject("matmul", fmt.Errorf("%w: incompatible left %v, right %v and destination %v shapes (right transposed: %t)",
			ErrInvalid, left.state.shape, right.state.shape, destination.state.shape, rightTransposed))
	}
	if err = state.acquire(destination, left, right); err != nil {
		return state.reject("matmul", err)
	}
	if err = state.commands.encodeMatMul(left.state.buffer, right.state.buffer, destination.state.buffer,
		uint32(m), uint32(n), uint32(k), rightTransposed); err != nil {
		state.status = scopePoisoned
		return fmt.Errorf("gpu: encode matmul: %w", err)
	}
	return nil
}

// Submit commits a nonempty recording once and returns without waiting.
// If ctx is already canceled at the commit boundary, Submit discards the recording.
func (s *Scope) Submit(ctx context.Context) (err error) {
	var state *scopeState
	if state, err = s.beginEncoding(); err != nil {
		return err
	}
	defer state.mutex.Unlock()

	if ctx == nil {
		return ErrInvalid
	}
	if err = ctx.Err(); err != nil {
		return state.finish(scopeDiscarded, err)
	}
	if len(state.leases) == 0 {
		return fmt.Errorf("%w: scope has no operations", ErrInvalid)
	}
	if err = state.commands.Commit(); err != nil {
		return state.finish(scopeDiscarded, fmt.Errorf("gpu: submit scope: %w", err))
	}

	state.status = scopeSubmitted
	return nil
}

// Wait observes a submitted scope's terminal result. Cancellation while work
// is outstanding leaves tensor leases held for a later Wait or Close.
func (s *Scope) Wait(ctx context.Context) (err error) {
	if s == nil || s.state == nil {
		return ErrClosed
	}
	if ctx == nil {
		return ErrInvalid
	}

	var state = s.state
	state.mutex.Lock()
	switch state.status {
	case scopeClosed:
		state.mutex.Unlock()
		return ErrClosed
	case scopeFinished:
		err = state.result
		state.mutex.Unlock()
		return err
	case scopeSubmitted:
		if err = ctx.Err(); err != nil {
			state.mutex.Unlock()
			return err
		}
	default:
		state.mutex.Unlock()
		return ErrInvalid
	}

	var completion = state.startWait()
	state.mutex.Unlock()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-completion.done:
		if err = ctx.Err(); err != nil {
			return err
		}
	}

	state.mutex.Lock()
	defer state.mutex.Unlock()
	if state.status == scopeSubmitted {
		state.finish(scopeFinished, completion.err)
	}
	return state.result
}

// Close discards unsubmitted work or drains a submission. If Close observes
// the command result first, it returns any execution error. Close is idempotent.
func (s *Scope) Close() (err error) {
	if s == nil || s.state == nil {
		return nil
	}

	var state = s.state
	state.mutex.Lock()
	if state.status == scopeSubmitted {
		var completion = state.startWait()
		state.mutex.Unlock()
		<-completion.done
		state.mutex.Lock()
		if state.status == scopeSubmitted {
			err = state.finish(scopeFinished, completion.err)
		}
	}

	defer state.mutex.Unlock()
	switch state.status {
	case scopeClosed:
		return nil
	case scopeRecording, scopePoisoned:
		err = state.finish(scopeDiscarded, nil)
	}

	state.device.mutex.Lock()
	state.device.children--
	state.device.mutex.Unlock()
	state.status = scopeClosed
	return err
}

func (s *Scope) beginEncoding() (state *scopeState, err error) {
	if s == nil || s.state == nil {
		return nil, ErrClosed
	}

	state = s.state
	state.mutex.Lock()
	if state.status != scopeRecording {
		err = ErrInvalid
		if state.status == scopeClosed {
			err = ErrClosed
		}
		state.mutex.Unlock()
		return nil, err
	}
	return state, nil
}

func (s *scopeState) reject(operation string, cause error) (err error) {
	if len(s.leases) > 0 {
		s.status = scopePoisoned
	}
	return fmt.Errorf("gpu: %s: %w", operation, cause)
}

// acquire validates every operand before installing any new leases.
func (s *scopeState) acquire(destination *Tensor, sources ...*Tensor) (err error) {
	s.device.mutex.Lock()
	defer s.device.mutex.Unlock()

	var target = destination.state
	if err = s.validateAccess(target, writeAccess); err != nil {
		return err
	}
	for _, source := range sources {
		if source.state == target {
			return fmt.Errorf("%w: destination aliases a source", ErrInvalid)
		}
		if err = s.validateAccess(source.state, readAccess); err != nil {
			return err
		}
	}

	if s.leases[target] == readAccess {
		target.readers--
	}
	target.writer = s
	s.leases[target] = writeAccess
	for _, source := range sources {
		if s.leases[source.state] == 0 {
			source.state.readers++
			s.leases[source.state] = readAccess
		}
	}
	return nil
}

func (s *scopeState) validateAccess(t *tensorState, access tensorAccess) (err error) {
	if t.device != s.device {
		return fmt.Errorf("%w: tensor belongs to another device", ErrInvalid)
	}
	if t.closed {
		return ErrClosed
	}
	if t.hostAccess || (t.writer != nil && t.writer != s) {
		return ErrBusy
	}
	if access == writeAccess && (t.readers > 1 || (t.readers == 1 && s.leases[t] != readAccess)) {
		return ErrBusy
	}
	if access == readAccess && !t.initialized && t.writer != s {
		return ErrUninitialized
	}
	return nil
}

func (s *scopeState) startWait() (completion *scopeCompletion) {
	if s.completion != nil {
		return s.completion
	}

	completion = &scopeCompletion{done: make(chan struct{})}
	s.completion = completion
	go completion.wait(s.commands)
	return completion
}

func (c *scopeCompletion) wait(commands commandBuffer) {
	c.err = commands.Wait()
	close(c.done)
}

// finish releases native scope references before dropping tensor leases. A
// discarded recording leaves previous contents intact.
func (s *scopeState) finish(status scopeStatus, commandErr error) (err error) {
	err = commandErr
	var releaseErr = s.commands.Release()
	if releaseErr != nil {
		err = errors.Join(err, fmt.Errorf("gpu: release scope: %w", releaseErr))
	}
	s.commands = nil
	s.status = status
	s.result = err

	s.device.mutex.Lock()
	defer s.device.mutex.Unlock()
	for tensor, access := range s.leases {
		if access == writeAccess {
			if status == scopeFinished {
				tensor.initialized = commandErr == nil
			}
			tensor.writer = nil
		} else {
			tensor.readers--
		}
		tensor.releaseIfIdle()
	}
	s.leases = nil
	return err
}
