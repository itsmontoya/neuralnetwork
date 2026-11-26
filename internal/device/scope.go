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
	if source.bytes != destination.bytes {
		err = fmt.Errorf("device: copy length mismatch: source=%d destination=%d", source.bytes, destination.bytes)
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

// EncodeGatherRows copies each indexed row of elemBytes-wide elements or blocks
// (2, 4, 18, 20, or 34 bytes) into a disjoint destination.
func (s *Scope) EncodeGatherRows(source, destination *Buffer, columns, elemBytes uint64, rows []int64) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(source, "gather source"); err != nil {
		return err
	}
	if err = s.validateBuffer(destination, "gather destination"); err != nil {
		return err
	}
	if elemBytes != 2 && elemBytes != 4 && elemBytes != 18 && elemBytes != 20 && elemBytes != 34 {
		return errors.New("device: invalid gather element width")
	}
	if source == destination || columns == 0 || len(rows) == 0 || columns > math.MaxUint64/elemBytes {
		return errors.New("device: invalid gather buffers or row size")
	}
	var rowBytes = columns * elemBytes
	if rowBytes > source.bytes || source.bytes%rowBytes != 0 ||
		destination.bytes%rowBytes != 0 || destination.bytes/rowBytes != uint64(len(rows)) {
		return errors.New("device: invalid gather buffers or row size")
	}
	var rowCount = source.bytes / rowBytes
	for _, row := range rows {
		if row < 0 || uint64(row) >= rowCount {
			return errors.New("device: gather row index outside source")
		}
	}
	if err = s.runtime.backend.encodeGatherRows(s.handle, source.handle, destination.handle, rowBytes, rows); err != nil {
		s.fail(err)
	}
	return err
}

// EncodeGatherRowsMixed gathers indexed reduced-precision or block-quantized
// rows (sourceDType: 1=Float16, 2=BFloat16, 3=Q8_0, 4=Q4_0, 5=Q4_1) and
// converts them into a disjoint FP32 destination.
func (s *Scope) EncodeGatherRowsMixed(source, destination *Buffer, columns uint64, rows []int64, sourceDType uint32) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	var backend, available = s.runtime.backend.(attentionBackend)
	if !available {
		return ErrUnavailable
	}
	if err = s.validateBuffer(source, "mixed gather source"); err != nil {
		return err
	}
	if err = s.validateBuffer(destination, "mixed gather destination"); err != nil {
		return err
	}
	if source == destination || columns == 0 || columns > math.MaxUint32 || len(rows) == 0 ||
		sourceDType < 1 || sourceDType > 5 || source.count != 0 || destination.count == 0 {
		return errors.New("device: invalid mixed gather buffers, dtype, or row size")
	}
	var srcRowBytes uint64
	switch sourceDType {
	case 1, 2:
		srcRowBytes = columns * 2
	case 3:
		if columns%32 != 0 {
			return errors.New("device: Q8_0 mixed gather columns must be a multiple of 32")
		}
		srcRowBytes = (columns / 32) * 34
	case 4:
		if columns%32 != 0 {
			return errors.New("device: Q4_0 mixed gather columns must be a multiple of 32")
		}
		srcRowBytes = (columns / 32) * 18
	case 5:
		if columns%32 != 0 {
			return errors.New("device: Q4_1 mixed gather columns must be a multiple of 32")
		}
		srcRowBytes = (columns / 32) * 20
	}
	var dstRowCount = destination.count / columns
	if srcRowBytes > source.bytes || source.bytes%srcRowBytes != 0 ||
		destination.count%columns != 0 || dstRowCount != uint64(len(rows)) {
		return errors.New("device: invalid mixed gather buffers or row size")
	}
	var rowCount = source.bytes / srcRowBytes
	for _, row := range rows {
		if row < 0 || uint64(row) >= rowCount {
			return errors.New("device: gather row index outside source")
		}
	}
	if err = backend.encodeGatherRowsMixed(s.handle, source.handle, destination.handle, columns, rows, sourceDType); err != nil {
		s.fail(err)
	}
	return err
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

// EncodeElementwise appends a full tensor operation. Unary activations take one
// input, and binary operations take two. The destination must be disjoint from inputs.
func (s *Scope) EncodeElementwise(left, right, result *Buffer, operation ElementwiseOperation) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	var unary = operation.isUnary()
	if operation > ElementwiseDiv || (unary && right != nil) {
		return errors.New("device: unsupported elementwise operation or input count")
	}
	if err = s.validateBuffer(left, "elementwise input"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "elementwise destination"); err != nil {
		return err
	}
	if left == result || right == result || left.count != result.count ||
		result.count == 0 || result.count > math.MaxUint32 {
		return errors.New("device: invalid elementwise destination, buffer length or index range")
	}

	var rightHandle any
	if !unary {
		if err = s.validateBuffer(right, "elementwise second input"); err != nil {
			return err
		}
		if right.count != result.count {
			return errors.New("device: elementwise second input length differs")
		}
		rightHandle = right.handle
	}

	err = s.runtime.backend.encodeElementwise(s.handle, left.handle, rightHandle, result.handle, uint32(result.count), operation)
	if err != nil {
		err = fmt.Errorf("device: encode elementwise operation: %w", err)
		s.fail(err)
	}
	return err
}

// EncodeTensorScale multiplies each FP32 element of input by scale into a
// disjoint destination buffer of matching length.
func (s *Scope) EncodeTensorScale(input, result *Buffer, scale float32) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(input, "scale input"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "scale destination"); err != nil {
		return err
	}
	if input == result || input.count != result.count || result.count == 0 || result.count > math.MaxUint32 {
		return errors.New("device: invalid scale destination, buffer length or index range")
	}
	if err = s.runtime.backend.encodeTensorScale(s.handle, input.handle, result.handle, scale, uint32(result.count)); err != nil {
		err = fmt.Errorf("device: encode tensor scale: %w", err)
		s.fail(err)
	}
	return err
}

// EncodeTensorBroadcast appends a multidirectional FP32 broadcast binary
// operation into a disjoint destination buffer.
func (s *Scope) EncodeTensorBroadcast(left, right, result *Buffer, operation BroadcastOperation, spec BroadcastSpec) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if operation > BroadcastDiv {
		return errors.New("device: unsupported broadcast operation")
	}
	if err = s.validateBuffer(left, "broadcast left input"); err != nil {
		return err
	}
	if err = s.validateBuffer(right, "broadcast right input"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "broadcast destination"); err != nil {
		return err
	}
	if left == result || right == result || left.count == 0 || right.count == 0 ||
		spec.Rank == 0 || spec.Rank > 6 || spec.Count == 0 || uint64(spec.Count) != result.count {
		return errors.New("device: invalid broadcast destination, rank or buffer length")
	}
	var (
		product  uint64 = 1
		maxLeft  uint64
		maxRight uint64
	)
	for i := range int(spec.Rank) {
		var dim = uint64(spec.Dims[i])
		if dim == 0 {
			return errors.New("device: broadcast dimension must be positive")
		}
		product *= dim
		if product > math.MaxUint32 {
			return errors.New("device: broadcast element count exceeds uint32")
		}
		maxLeft += (dim - 1) * uint64(spec.LeftStrides[i])
		maxRight += (dim - 1) * uint64(spec.RightStrides[i])
	}
	if product != uint64(spec.Count) || maxLeft >= left.count || maxRight >= right.count {
		return errors.New("device: broadcast strides exceed input buffer length")
	}
	if err = s.runtime.backend.encodeTensorBroadcast(s.handle, left.handle, right.handle, result.handle, operation, spec); err != nil {
		err = fmt.Errorf("device: encode tensor broadcast: %w", err)
		s.fail(err)
	}
	return err
}

// EncodeTensorSoftmax appends a numerically stable FP32 softmax over the middle
// dimension of [outer, axisSize, inner] into a disjoint destination buffer.
func (s *Scope) EncodeTensorSoftmax(input, result *Buffer, outer, axisSize, inner uint32) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(input, "softmax input"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "softmax destination"); err != nil {
		return err
	}
	if input == result || outer == 0 || axisSize == 0 || inner == 0 {
		return errors.New("device: invalid softmax destination or dimensions")
	}
	var total = uint64(outer) * uint64(axisSize)
	if total > math.MaxUint32 {
		return errors.New("device: softmax element count exceeds uint32")
	}
	total *= uint64(inner)
	if total > math.MaxUint32 || input.count != total || result.count != total {
		return errors.New("device: softmax buffer length mismatch or index overflow")
	}
	if err = s.runtime.backend.encodeTensorSoftmax(s.handle, input.handle, result.handle, outer, axisSize, inner); err != nil {
		err = fmt.Errorf("device: encode tensor softmax: %w", err)
		s.fail(err)
	}
	return err
}

// EncodeTensorLayerNorm appends a numerically stable two-pass FP32 LayerNorm
// over rows of width elements with Float32 (0), Float16 (1), or BFloat16 (2)
// weight and optional bias buffers into a disjoint destination buffer.
func (s *Scope) EncodeTensorLayerNorm(input, weight, bias, result *Buffer, rows, width uint32, epsilon float32, weightDType, biasDType uint32) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(input, "LayerNorm input"); err != nil {
		return err
	}
	if err = s.validateBuffer(weight, "LayerNorm weight"); err != nil {
		return err
	}
	var biasHandle any
	if bias != nil {
		if err = s.validateBuffer(bias, "LayerNorm bias"); err != nil {
			return err
		}
		biasHandle = bias.handle
	}
	if err = s.validateBuffer(result, "LayerNorm destination"); err != nil {
		return err
	}
	if input == result || weight == result || (bias != nil && bias == result) ||
		rows == 0 || width == 0 || !(epsilon > 0 && epsilon <= math.MaxFloat32) ||
		weightDType > 2 || (bias != nil && biasDType > 2) {
		return errors.New("device: invalid LayerNorm destination, dimensions, epsilon or dtype")
	}
	var total = uint64(rows) * uint64(width)
	if total > math.MaxUint32 || input.count != total || result.count != total {
		return errors.New("device: LayerNorm input or destination buffer length mismatch")
	}
	var weightElemBytes uint64 = 4
	if weightDType != 0 {
		weightElemBytes = 2
	}
	if weight.bytes != uint64(width)*weightElemBytes {
		return errors.New("device: LayerNorm weight buffer length mismatch")
	}
	if bias != nil {
		var biasElemBytes uint64 = 4
		if biasDType != 0 {
			biasElemBytes = 2
		}
		if bias.bytes != uint64(width)*biasElemBytes {
			return errors.New("device: LayerNorm bias buffer length mismatch")
		}
	}
	if err = s.runtime.backend.encodeTensorLayerNorm(s.handle, input.handle, weight.handle, biasHandle, result.handle, rows, width, epsilon, weightDType, biasDType); err != nil {
		err = fmt.Errorf("device: encode tensor LayerNorm: %w", err)
		s.fail(err)
	}
	return err
}

// EncodeTensorTranspose appends a contiguous 2-byte or 4-byte axis permutation
// described by spec into a disjoint destination buffer.
func (s *Scope) EncodeTensorTranspose(input, result *Buffer, spec TransposeSpec) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(input, "transpose input"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "transpose destination"); err != nil {
		return err
	}
	if input == result || spec.Rank == 0 || spec.Rank > 6 || spec.Count == 0 ||
		(spec.ElemBytes != 2 && spec.ElemBytes != 4) {
		return errors.New("device: invalid transpose destination, rank, count or element width")
	}
	var expectedBytes = uint64(spec.Count) * uint64(spec.ElemBytes)
	if input.bytes != expectedBytes || result.bytes != expectedBytes {
		return errors.New("device: transpose buffer length mismatch")
	}
	var (
		product uint64 = 1
		maxIn   uint64
	)
	for i := range int(spec.Rank) {
		var dim = uint64(spec.Dims[i])
		if dim == 0 {
			return errors.New("device: transpose dimension must be positive")
		}
		product *= dim
		if product > math.MaxUint32 {
			return errors.New("device: transpose element count exceeds uint32")
		}
		maxIn += (dim - 1) * uint64(spec.InStrides[i])
	}
	if product != uint64(spec.Count) || maxIn >= uint64(spec.Count) {
		return errors.New("device: transpose strides exceed buffer length")
	}
	if err = s.runtime.backend.encodeTensorTranspose(s.handle, input.handle, result.handle, spec); err != nil {
		err = fmt.Errorf("device: encode tensor transpose: %w", err)
		s.fail(err)
	}
	return err
}

// EncodeTensorMatMulLeftTranspose appends left[K,M]^T * right[K,N] into
// result[M,N] with FP32 accumulation for Float32 (0), Float16 (1), or
// BFloat16 (2) right operands.
func (s *Scope) EncodeTensorMatMulLeftTranspose(left, right, result *Buffer, m, n, k, rightDType uint32) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(left, "left-transposed matmul left input"); err != nil {
		return err
	}
	if err = s.validateBuffer(right, "left-transposed matmul right input"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "left-transposed matmul destination"); err != nil {
		return err
	}
	if left == result || right == result || m == 0 || n == 0 || k == 0 || rightDType > 2 {
		return errors.New("device: invalid left-transposed matmul destination, dimensions or dtype")
	}
	var (
		leftCount   = uint64(k) * uint64(m)
		rightCount  = uint64(k) * uint64(n)
		resultCount = uint64(m) * uint64(n)
	)
	if leftCount > math.MaxUint32 || rightCount > math.MaxUint32 || resultCount > math.MaxUint32 {
		return errors.New("device: left-transposed matmul element count exceeds uint32")
	}
	var rightElemBytes uint64 = 4
	if rightDType != 0 {
		rightElemBytes = 2
	}
	if left.count != leftCount || result.count != resultCount || right.bytes != rightCount*rightElemBytes {
		return errors.New("device: left-transposed matmul buffer length mismatch")
	}
	if err = s.runtime.backend.encodeTensorMatMulLeftTranspose(s.handle, left.handle, right.handle, result.handle, m, n, k, rightDType); err != nil {
		err = fmt.Errorf("device: encode left-transposed matmul: %w", err)
		s.fail(err)
	}
	return err
}

// EncodeTensorBatchedMatMul appends a rank 2..6 broadcast-batched matrix
// multiplication with FP32 accumulation for Float32 (0), Float16 (1), or
// BFloat16 (2) right operands.
func (s *Scope) EncodeTensorBatchedMatMul(left, right, result *Buffer, spec BatchedMatMulSpec) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()

	if err = s.validateBuffer(left, "batched matmul left input"); err != nil {
		return err
	}
	if err = s.validateBuffer(right, "batched matmul right input"); err != nil {
		return err
	}
	if err = s.validateBuffer(result, "batched matmul destination"); err != nil {
		return err
	}
	if left == result || right == result || spec.M == 0 || spec.N == 0 || spec.K == 0 ||
		spec.TotalBatches == 0 || spec.BatchRank > 4 || spec.RightDType > 2 {
		return errors.New("device: invalid batched matmul destination, dimensions or dtype")
	}
	var (
		leftMatrixElems  = uint64(spec.M) * uint64(spec.K)
		rightMatrixElems = uint64(spec.K) * uint64(spec.N)
		outMatrixElems   = uint64(spec.M) * uint64(spec.N)
	)
	if leftMatrixElems > math.MaxUint32 || rightMatrixElems > math.MaxUint32 || outMatrixElems > math.MaxUint32 {
		return errors.New("device: batched matmul matrix size exceeds uint32")
	}
	var totalOut = uint64(spec.TotalBatches) * outMatrixElems
	if totalOut > math.MaxUint32 || result.count != totalOut {
		return errors.New("device: batched matmul destination buffer length mismatch")
	}
	var rightElemBytes uint64 = 4
	if spec.RightDType != 0 {
		rightElemBytes = 2
	}
	if left.count == 0 || left.count%leftMatrixElems != 0 ||
		right.bytes == 0 || right.bytes%(rightMatrixElems*rightElemBytes) != 0 {
		return errors.New("device: batched matmul input buffer length mismatch")
	}
	if err = s.runtime.backend.encodeTensorBatchedMatMul(s.handle, left.handle, right.handle, result.handle, spec); err != nil {
		err = fmt.Errorf("device: encode batched matmul: %w", err)
		s.fail(err)
	}
	return err
}

// EncodeKVAppend writes keys and values directly into the uncommitted tail of
// target at offsets prefix and plane+prefix without copying the prior prefix.
func (s *Scope) EncodeKVAppend(keys, values, target *Buffer, plane, prefix uint32) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()
	var backend, available = s.runtime.backend.(attentionBackend)
	if !available {
		return ErrUnavailable
	}
	for _, buffer := range [3]*Buffer{keys, values, target} {
		if err = s.validateBuffer(buffer, "KV append"); err != nil {
			return err
		}
	}
	if target == keys || target == values || keys.count == 0 || keys.count != values.count ||
		target.count != 2*uint64(plane) || uint64(prefix)+keys.count > uint64(plane) {
		return errors.New("device: invalid KV append buffers or range")
	}
	if err = backend.encodeKVAppend(s.handle, keys.handle, values.handle, target.handle, plane, prefix); err != nil {
		s.fail(err)
	}
	return err
}

// EncodeKVAppendFloat16 converts FP32 keys and values directly into the
// uncommitted tail of a Float16 KV storage buffer at offsets prefix and
// plane+prefix without copying the prior prefix.
func (s *Scope) EncodeKVAppendFloat16(keys, values, target *Buffer, plane, prefix uint32) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()
	var backend, available = s.runtime.backend.(attentionBackend)
	if !available {
		return ErrUnavailable
	}
	for _, buffer := range [3]*Buffer{keys, values, target} {
		if err = s.validateBuffer(buffer, "FP16 KV append"); err != nil {
			return err
		}
	}
	if target == keys || target == values || keys.count == 0 || keys.count != values.count ||
		target.count != 0 || target.bytes != 2*uint64(plane)*2 || uint64(prefix)+keys.count > uint64(plane) {
		return errors.New("device: invalid FP16 KV append buffers or range")
	}
	if err = backend.encodeKVAppendFloat16(s.handle, keys.handle, values.handle, target.handle, plane, prefix); err != nil {
		s.fail(err)
	}
	return err
}

// EncodeAttentionMath binds disjoint input and output buffers to a
// tensor-only attention or normalization kernel. The caller validates the operation's shape.
func (s *Scope) EncodeAttentionMath(operation AttentionMathOperation, first, second, target, extra *Buffer, params [6]uint32, threads uint32) (err error) {
	if err = s.lockForEncoding(); err != nil {
		return err
	}
	defer s.mutex.Unlock()
	var (
		backend, available = s.runtime.backend.(attentionBackend)
		buffers            [4]any
	)
	if !available {
		return ErrUnavailable
	}
	if operation > TensorCausalAttentionFloat16 || threads == 0 || (extra != nil) != (operation == TensorRotatePairs) ||
		first == target || second == target || extra == target {
		return errors.New("device: invalid attention math operation or buffer alias")
	}
	for index, buffer := range [4]*Buffer{first, second, target, extra} {
		if index == 3 && extra == nil {
			break
		}
		if err = s.validateBuffer(buffer, "attention math"); err != nil {
			return err
		}
		buffers[index] = buffer.handle
	}
	if err = backend.encodeAttentionMath(s.handle, operation, buffers, params, threads); err != nil {
		s.fail(err)
	}
	return err
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
	if operation > OperationTensorMatMulRightTransposeQ4_1 {
		err = fmt.Errorf("device: unsupported matrix multiplication operation: %d", operation)
		return err
	}

	dimensions.leftRows = leftRows
	dimensions.leftCols = leftCols
	dimensions.rightRows = rightRows
	dimensions.rightCols = rightCols
	dimensions.resultRows = resultRows
	dimensions.resultCols = resultCols
	if operation >= OperationTensorMatMulFloat16 {
		if err = s.validateMixedMatMulBuffers(left, right, result, dimensions, operation); err != nil {
			return err
		}
	} else {
		if err = s.validateMatMulCounts(left, right, result, dimensions); err != nil {
			return err
		}
		if operation >= OperationTensorMatMul {
			var inner, cols = rightRows, rightCols
			if operation == OperationTensorMatMulRightTranspose {
				inner, cols = cols, inner
			}
			if leftRows != resultRows || leftCols != inner || cols != resultCols ||
				left.count == 0 || right.count == 0 || result.count == 0 ||
				left.count > math.MaxUint32 || right.count > math.MaxUint32 || result.count > math.MaxUint32 {
				return errors.New("device: invalid tensor matmul dimensions or index range")
			}
		}
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
	var releaseErr error

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
		if releaseErr = s.runtime.backend.releaseScope(s.handle); releaseErr != nil {
			releaseErr = fmt.Errorf("device: release command scope: %w", releaseErr)
			err = errors.Join(err, releaseErr)
		}
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

func (s *Scope) validateMixedMatMulBuffers(
	left,
	right,
	result *Buffer,
	dimensions matMulDimensions,
	operation Operation,
) (err error) {
	var (
		leftCount   = uint64(dimensions.leftRows) * uint64(dimensions.leftCols)
		rightCount  = uint64(dimensions.rightRows) * uint64(dimensions.rightCols)
		resultCount = uint64(dimensions.resultRows) * uint64(dimensions.resultCols)
		inner       = dimensions.rightRows
		cols        = dimensions.rightCols
	)
	if operation == OperationTensorMatMulRightTransposeFloat16 ||
		operation == OperationTensorMatMulRightTransposeBFloat16 ||
		operation == OperationTensorMatMulRightTransposeQ8_0 ||
		operation == OperationTensorMatMulRightTransposeQ4_0 ||
		operation == OperationTensorMatMulRightTransposeQ4_1 {
		inner, cols = cols, inner
	}
	if dimensions.leftRows != dimensions.resultRows || dimensions.leftCols != inner || cols != dimensions.resultCols ||
		leftCount == 0 || rightCount == 0 || resultCount == 0 ||
		leftCount > math.MaxUint32 || rightCount > math.MaxUint32 || resultCount > math.MaxUint32 {
		return errors.New("device: invalid mixed-precision tensor matmul dimensions or index range")
	}
	var expectedRightBytes uint64
	switch operation {
	case OperationTensorMatMulRightTransposeQ8_0:
		if inner%32 != 0 {
			return errors.New("device: Q8_0 tensor matmul inner dimension must be a multiple of 32")
		}
		expectedRightBytes = (rightCount / 32) * 34
	case OperationTensorMatMulRightTransposeQ4_0:
		if inner%32 != 0 {
			return errors.New("device: Q4_0 tensor matmul inner dimension must be a multiple of 32")
		}
		expectedRightBytes = (rightCount / 32) * 18
	case OperationTensorMatMulRightTransposeQ4_1:
		if inner%32 != 0 {
			return errors.New("device: Q4_1 tensor matmul inner dimension must be a multiple of 32")
		}
		expectedRightBytes = (rightCount / 32) * 20
	default:
		expectedRightBytes = rightCount * 2
	}
	if left.count != leftCount || right.count != 0 || right.bytes != expectedRightBytes || result.count != resultCount {
		return fmt.Errorf(
			"device: mixed-precision matrix multiplication buffer length mismatch: left=%d/%d rightBytes=%d/%d result=%d/%d",
			left.count,
			leftCount,
			right.bytes,
			expectedRightBytes,
			result.count,
			resultCount,
		)
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
