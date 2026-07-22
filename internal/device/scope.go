package device

import (
	"errors"
	"fmt"
	"sync"
)

type scopeState uint8

const (
	scopeStateEncoding scopeState = iota
	scopeStateCommitted
	scopeStateCompleted
	scopeStateFailed
	scopeStateReleased
)

// Scope owns one command buffer and its encoded resource references.
type Scope struct {
	mutex   sync.Mutex
	runtime *Runtime
	handle  any
	state   scopeState
	err     error
}

// EncodeCopy appends a complete buffer copy to the scope.
func (s *Scope) EncodeCopy(source, destination *Buffer) (err error) {
	var (
		sourceHandle      any
		destinationHandle any
	)

	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(source, "copy source"); err != nil {
		return err
	}
	if err = s.validateBuffer(destination, "copy destination"); err != nil {
		return err
	}
	if source == destination {
		err = errors.New("device: copy destination must not alias source")
		return err
	}
	if source.count != destination.count {
		err = fmt.Errorf("device: copy length mismatch: source=%d destination=%d", source.count, destination.count)
		return err
	}

	sourceHandle = source.handle
	destinationHandle = destination.handle
	if err = s.runtime.backend.encodeCopy(s.handle, sourceHandle, destinationHandle, source.bytes); err != nil {
		err = fmt.Errorf("device: encode copy: %w", err)
		s.fail(err)
		return err
	}

	return nil
}

// EncodeFill appends a complete float32 buffer fill to the scope.
func (s *Scope) EncodeFill(buffer *Buffer, value float32) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(buffer, "fill destination"); err != nil {
		return err
	}

	if err = s.runtime.backend.encodeFill(s.handle, buffer.handle, value, buffer.count); err != nil {
		err = fmt.Errorf("device: encode fill: %w", err)
		s.fail(err)
		return err
	}

	return nil
}

// EncodeMatMul appends a matrix multiplication to the scope.
func (s *Scope) EncodeMatMul(
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
) (err error) {
	var dimensions matMulDimensions

	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(left, "matrix multiplication left input"); err != nil {
		return err
	}
	if err = s.validateBuffer(right, "matrix multiplication right input"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "matrix multiplication destination"); err != nil {
		return err
	}
	if left == result || right == result {
		err = errors.New("device: matrix multiplication destination must not alias an input")
		return err
	}
	if operation > OperationMatMulRightTranspose {
		err = fmt.Errorf("device: unsupported matrix multiplication operation: %d", operation)
		return err
	}

	dimensions.leftRows = leftRows
	dimensions.leftCols = leftCols
	dimensions.rightRows = rightRows
	dimensions.rightCols = rightCols
	dimensions.resultRows = resultRows
	dimensions.resultCols = resultCols
	if err = s.validateMatMulCounts(left, right, result, dimensions); err != nil {
		return err
	}

	if err = s.runtime.backend.encodeMatMul(
		s.handle,
		left.handle,
		right.handle,
		result.handle,
		dimensions,
		operation,
	); err != nil {
		err = fmt.Errorf("device: encode matrix multiplication: %w", err)
		s.fail(err)
		return err
	}

	return nil
}

// Commit submits the encoded commands without waiting for completion.
func (s *Scope) Commit() (err error) {
	if s == nil {
		err = errors.New("device: commit scope: scope is nil")
		return err
	}

	s.mutex.Lock()
	defer s.mutex.Unlock()
	if s.state != scopeStateEncoding {
		err = fmt.Errorf("device: commit scope in state %s: %w", s.state, ErrInvalidState)
		return err
	}

	if err = s.runtime.backend.commit(s.handle); err != nil {
		err = fmt.Errorf("device: commit command scope: %w", err)
		s.fail(err)
		return err
	}

	s.state = scopeStateCommitted
	return nil
}

// Completed reports whether a committed scope has finished.
func (s *Scope) Completed() (complete bool, err error) {
	if s == nil {
		err = errors.New("device: inspect scope completion: scope is nil")
		return false, err
	}

	s.mutex.Lock()
	defer s.mutex.Unlock()
	switch s.state {
	case scopeStateCompleted:
		return true, nil
	case scopeStateFailed:
		return false, s.err
	case scopeStateCommitted:
		complete, err = s.runtime.backend.completed(s.handle)
		if err != nil {
			err = fmt.Errorf("device: inspect command completion: %w", err)
			s.fail(err)
			return false, err
		}
		if complete {
			s.state = scopeStateCompleted
		}
		return complete, nil
	default:
		err = fmt.Errorf("device: inspect completion in state %s: %w", s.state, ErrInvalidState)
		return false, err
	}
}

// Wait blocks until a committed scope completes and reports command failures.
func (s *Scope) Wait() (err error) {
	if s == nil {
		err = errors.New("device: wait for scope: scope is nil")
		return err
	}

	s.mutex.Lock()
	defer s.mutex.Unlock()
	switch s.state {
	case scopeStateCompleted:
		return nil
	case scopeStateFailed:
		return s.err
	case scopeStateCommitted:
		if err = s.runtime.backend.wait(s.handle); err != nil {
			err = fmt.Errorf("device: wait for command scope: %w", err)
			s.fail(err)
			return err
		}
		s.state = scopeStateCompleted
		return nil
	default:
		err = fmt.Errorf("device: wait for scope in state %s: %w", s.state, ErrInvalidState)
		return err
	}
}

// Release waits when necessary and releases all command-scope resources.
func (s *Scope) Release() (err error) {
	if s == nil {
		return nil
	}

	s.mutex.Lock()
	if s.state == scopeStateReleased {
		s.mutex.Unlock()
		return nil
	}
	if s.state == scopeStateCommitted {
		if err = s.runtime.backend.wait(s.handle); err != nil {
			err = fmt.Errorf("device: release command scope after wait: %w", err)
			s.fail(err)
		} else {
			s.state = scopeStateCompleted
		}
	}
	if s.runtime != nil && s.runtime.backend != nil && s.handle != nil {
		s.runtime.backend.releaseScope(s.handle)
	}
	s.handle = nil
	s.state = scopeStateReleased
	s.mutex.Unlock()
	return err
}

func (s *Scope) lockForEncoding() (err error) {
	if s == nil {
		err = errors.New("device: encode command: scope is nil")
		return err
	}

	s.mutex.Lock()
	if s.state != scopeStateEncoding {
		s.mutex.Unlock()
		err = fmt.Errorf("device: encode command in state %s: %w", s.state, ErrInvalidState)
		return err
	}
	if s.runtime == nil || s.runtime.backend == nil || s.handle == nil {
		s.mutex.Unlock()
		err = errors.New("device: command scope has nil runtime or handle")
		return err
	}

	return nil
}

func (s *Scope) validateBuffer(buffer *Buffer, name string) (err error) {
	if buffer == nil {
		err = fmt.Errorf("device: %s is nil", name)
		return err
	}
	if buffer.runtime != s.runtime {
		err = fmt.Errorf("device: %s belongs to another runtime", name)
		return err
	}
	if buffer.released || buffer.handle == nil {
		err = fmt.Errorf("device: %s: %w", name, ErrReleased)
		return err
	}

	return nil
}

func (s *Scope) validateMatMulCounts(
	left,
	right,
	result *Buffer,
	dimensions matMulDimensions,
) (err error) {
	var (
		leftCount   uint64
		rightCount  uint64
		resultCount uint64
	)

	leftCount = uint64(dimensions.leftRows) * uint64(dimensions.leftCols)
	rightCount = uint64(dimensions.rightRows) * uint64(dimensions.rightCols)
	resultCount = uint64(dimensions.resultRows) * uint64(dimensions.resultCols)
	if left.count != leftCount || right.count != rightCount || result.count != resultCount {
		err = fmt.Errorf(
			"device: matrix multiplication buffer length mismatch: left=%d/%d right=%d/%d result=%d/%d",
			left.count,
			leftCount,
			right.count,
			rightCount,
			result.count,
			resultCount,
		)
		return err
	}

	return nil
}

func (s *Scope) fail(err error) {
	s.err = err
	s.state = scopeStateFailed
}

func (s scopeState) String() (name string) {
	switch s {
	case scopeStateEncoding:
		name = "encoding"
	case scopeStateCommitted:
		name = "committed"
	case scopeStateCompleted:
		name = "completed"
	case scopeStateFailed:
		name = "failed"
	case scopeStateReleased:
		name = "released"
	default:
		name = "unknown"
	}
	return name
}
