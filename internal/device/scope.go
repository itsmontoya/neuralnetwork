package device

import (
	"errors"
	"fmt"
	"math"
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

// EncodeAddRowVector appends an in-place row-vector addition to the scope.
func (s *Scope) EncodeAddRowVector(
	values,
	rowVector *Buffer,
	rows,
	cols uint32,
) (err error) {
	var expectedValues uint64

	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(values, "row-vector addition values"); err != nil {
		return err
	}
	if err = s.validateBuffer(rowVector, "row-vector addition row vector"); err != nil {
		return err
	}
	if rows == 0 || cols == 0 {
		err = errors.New("device: row-vector addition dimensions must be positive")
		return err
	}

	expectedValues = uint64(rows) * uint64(cols)
	if values.count != expectedValues || rowVector.count != uint64(cols) {
		err = fmt.Errorf(
			"device: row-vector addition buffer length mismatch: values=%d/%d rowVector=%d/%d",
			values.count,
			expectedValues,
			rowVector.count,
			cols,
		)
		return err
	}
	if err = s.runtime.backend.encodeAddRowVector(
		s.handle,
		values.handle,
		rowVector.handle,
		rows,
		cols,
	); err != nil {
		err = fmt.Errorf("device: encode row-vector addition: %w", err)
		s.fail(err)
		return err
	}

	return nil
}

// EncodeAddScaled appends result=left+scale*right to the scope.
func (s *Scope) EncodeAddScaled(
	left,
	right,
	result *Buffer,
	scale float32,
) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(left, "scaled addition left input"); err != nil {
		return err
	}
	if err = s.validateBuffer(right, "scaled addition right input"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "scaled addition destination"); err != nil {
		return err
	}
	if left.count != right.count || left.count != result.count {
		err = fmt.Errorf(
			"device: scaled addition buffer length mismatch: left=%d right=%d destination=%d",
			left.count,
			right.count,
			result.count,
		)
		return err
	}
	if left.count > uint64(^uint32(0)) {
		err = fmt.Errorf("device: scaled addition element count exceeds uint32: %d", left.count)
		return err
	}
	if err = s.runtime.backend.encodeAddScaled(
		s.handle,
		left.handle,
		right.handle,
		result.handle,
		scale,
		uint32(left.count),
	); err != nil {
		err = fmt.Errorf("device: encode scaled addition: %w", err)
		s.fail(err)
		return err
	}

	return nil
}

// EncodeReLU appends a rectified-linear-unit forward operation to the scope.
func (s *Scope) EncodeReLU(input, result *Buffer) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(input, "ReLU input"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "ReLU destination"); err != nil {
		return err
	}
	if input.count != result.count {
		err = fmt.Errorf(
			"device: ReLU buffer length mismatch: input=%d destination=%d",
			input.count,
			result.count,
		)
		return err
	}
	if input.count > uint64(^uint32(0)) {
		err = fmt.Errorf("device: ReLU element count exceeds uint32: %d", input.count)
		return err
	}
	if err = s.runtime.backend.encodeReLU(
		s.handle,
		input.handle,
		result.handle,
		uint32(input.count),
	); err != nil {
		err = fmt.Errorf("device: encode ReLU: %w", err)
		s.fail(err)
		return err
	}

	return nil
}

// EncodeReLUBackward appends a rectified-linear-unit derivative operation.
func (s *Scope) EncodeReLUBackward(
	input,
	outputGradient,
	result *Buffer,
) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(input, "ReLU backward input"); err != nil {
		return err
	}
	if err = s.validateBuffer(outputGradient, "ReLU backward output gradient"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "ReLU backward destination"); err != nil {
		return err
	}
	if input.count != outputGradient.count || input.count != result.count {
		err = fmt.Errorf(
			"device: ReLU backward buffer length mismatch: input=%d gradient=%d destination=%d",
			input.count,
			outputGradient.count,
			result.count,
		)
		return err
	}
	if input.count > uint64(^uint32(0)) {
		err = fmt.Errorf("device: ReLU backward element count exceeds uint32: %d", input.count)
		return err
	}
	if err = s.runtime.backend.encodeReLUBackward(
		s.handle,
		input.handle,
		outputGradient.handle,
		result.handle,
		uint32(input.count),
	); err != nil {
		err = fmt.Errorf("device: encode ReLU backward: %w", err)
		s.fail(err)
		return err
	}

	return nil
}

// EncodeSoftmaxRows appends a stable row-wise Softmax operation to the scope.
func (s *Scope) EncodeSoftmaxRows(
	input,
	result *Buffer,
	rows,
	cols uint32,
) (err error) {
	var expectedCount uint64

	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(input, "Softmax input"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "Softmax destination"); err != nil {
		return err
	}
	if rows == 0 || cols == 0 {
		err = errors.New("device: Softmax dimensions must be positive")
		return err
	}

	expectedCount = uint64(rows) * uint64(cols)
	if expectedCount > uint64(^uint32(0)) {
		err = fmt.Errorf("device: Softmax element count exceeds uint32: %d", expectedCount)
		return err
	}
	if input.count != expectedCount || result.count != expectedCount {
		err = fmt.Errorf(
			"device: Softmax buffer length mismatch: input=%d/%d destination=%d/%d",
			input.count,
			expectedCount,
			result.count,
			expectedCount,
		)
		return err
	}
	if err = s.runtime.backend.encodeSoftmaxRows(
		s.handle,
		input.handle,
		result.handle,
		rows,
		cols,
	); err != nil {
		err = fmt.Errorf("device: encode Softmax: %w", err)
		s.fail(err)
		return err
	}

	return nil
}

// EncodeSoftmaxRowsBackward appends a stable row-wise Softmax Jacobian product.
func (s *Scope) EncodeSoftmaxRowsBackward(
	input,
	outputGradient,
	result *Buffer,
	rows,
	cols uint32,
) (err error) {
	var expectedCount uint64

	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(input, "Softmax backward input"); err != nil {
		return err
	}
	if err = s.validateBuffer(outputGradient, "Softmax backward output gradient"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "Softmax backward destination"); err != nil {
		return err
	}
	if rows == 0 || cols == 0 {
		err = errors.New("device: Softmax backward dimensions must be positive")
		return err
	}

	expectedCount = uint64(rows) * uint64(cols)
	if expectedCount > uint64(^uint32(0)) {
		err = fmt.Errorf("device: Softmax backward element count exceeds uint32: %d", expectedCount)
		return err
	}
	if input.count != expectedCount ||
		outputGradient.count != expectedCount ||
		result.count != expectedCount {
		err = fmt.Errorf(
			"device: Softmax backward buffer length mismatch: input=%d/%d gradient=%d/%d destination=%d/%d",
			input.count,
			expectedCount,
			outputGradient.count,
			expectedCount,
			result.count,
			expectedCount,
		)
		return err
	}
	if err = s.runtime.backend.encodeSoftmaxRowsBackward(
		s.handle,
		input.handle,
		outputGradient.handle,
		result.handle,
		rows,
		cols,
	); err != nil {
		err = fmt.Errorf("device: encode Softmax backward: %w", err)
		s.fail(err)
		return err
	}

	return nil
}

// EncodeColumnSums appends a deterministic column reduction.
func (s *Scope) EncodeColumnSums(
	input,
	result *Buffer,
	rows,
	cols uint32,
	accumulate bool,
) (err error) {
	var expectedCount uint64

	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(input, "column sums input"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "column sums destination"); err != nil {
		return err
	}
	if rows == 0 || cols == 0 {
		err = errors.New("device: column sums dimensions must be positive")
		return err
	}

	expectedCount = uint64(rows) * uint64(cols)
	if input.count != expectedCount || result.count != uint64(cols) {
		err = fmt.Errorf(
			"device: column sums buffer length mismatch: input=%d/%d destination=%d/%d",
			input.count,
			expectedCount,
			result.count,
			cols,
		)
		return err
	}
	if err = s.runtime.backend.encodeColumnSums(
		s.handle,
		input.handle,
		result.handle,
		rows,
		cols,
		accumulate,
	); err != nil {
		err = fmt.Errorf("device: encode column sums: %w", err)
		s.fail(err)
		return err
	}

	return nil
}

// EncodeCategoricalCrossEntropy appends one validated mean loss reduction.
func (s *Scope) EncodeCategoricalCrossEntropy(
	predictions,
	targets,
	result *Buffer,
	rows,
	cols uint32,
	epsilon float32,
) (err error) {
	var expectedCount uint64

	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(predictions, "categorical predictions"); err != nil {
		return err
	}
	if err = s.validateBuffer(targets, "categorical targets"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "categorical result"); err != nil {
		return err
	}
	if rows == 0 || cols == 0 {
		err = errors.New("device: categorical dimensions must be positive")
		return err
	}
	if epsilon <= 0 || epsilon >= 0.5 ||
		math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
		err = fmt.Errorf("device: categorical epsilon must be between 0 and 0.5: %g", epsilon)
		return err
	}

	expectedCount = uint64(rows) * uint64(cols)
	if predictions.count != expectedCount || targets.count != expectedCount ||
		result.count != CategoricalCrossEntropyResultCount {
		err = fmt.Errorf(
			"device: categorical buffer length mismatch: predictions=%d/%d targets=%d/%d result=%d/%d",
			predictions.count,
			expectedCount,
			targets.count,
			expectedCount,
			result.count,
			CategoricalCrossEntropyResultCount,
		)
		return err
	}
	if err = s.runtime.backend.encodeCategoricalCrossEntropy(
		s.handle,
		predictions.handle,
		targets.handle,
		result.handle,
		rows,
		cols,
		epsilon,
	); err != nil {
		err = fmt.Errorf("device: encode categorical cross entropy: %w", err)
		s.fail(err)
		return err
	}

	return nil
}

// EncodeCategoricalCrossEntropyGradient appends a mean prediction gradient.
func (s *Scope) EncodeCategoricalCrossEntropyGradient(
	predictions,
	targets,
	result *Buffer,
	rows,
	cols uint32,
	epsilon float32,
) (err error) {
	var expectedCount uint64

	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(predictions, "categorical gradient predictions"); err != nil {
		return err
	}
	if err = s.validateBuffer(targets, "categorical gradient targets"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "categorical gradient destination"); err != nil {
		return err
	}
	if rows == 0 || cols == 0 {
		err = errors.New("device: categorical gradient dimensions must be positive")
		return err
	}
	if epsilon <= 0 || epsilon >= 0.5 ||
		math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
		err = fmt.Errorf("device: categorical gradient epsilon must be between 0 and 0.5: %g", epsilon)
		return err
	}

	expectedCount = uint64(rows) * uint64(cols)
	if expectedCount > uint64(^uint32(0)) {
		err = fmt.Errorf("device: categorical gradient element count exceeds uint32: %d", expectedCount)
		return err
	}
	if predictions.count != expectedCount || targets.count != expectedCount ||
		result.count != expectedCount {
		err = fmt.Errorf(
			"device: categorical gradient buffer length mismatch: predictions=%d/%d targets=%d/%d destination=%d/%d",
			predictions.count,
			expectedCount,
			targets.count,
			expectedCount,
			result.count,
			expectedCount,
		)
		return err
	}
	if err = s.runtime.backend.encodeCategoricalCrossEntropyGradient(
		s.handle,
		predictions.handle,
		targets.handle,
		result.handle,
		rows,
		cols,
		epsilon,
	); err != nil {
		err = fmt.Errorf("device: encode categorical cross entropy gradient: %w", err)
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
