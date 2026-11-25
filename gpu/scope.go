package gpu

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"

	"github.com/itsmontoya/neuralnetwork/internal/device"
)

func newScope(device *deviceState, commands commandBuffer) (out *Scope) {
	var state scopeState
	state.device = device
	state.commands = commands
	state.leases = make(map[*tensorStorage]tensorAccess)
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
	leases     map[*tensorStorage]tensorAccess
	kvAppends  []*kvAppendState
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
	var dstStorage = destination.state.storage
	if dstStorage.dtype != Float32 {
		return state.reject("fill", fmt.Errorf("%w: fill requires float32 tensor, got %s", ErrInvalid, dstStorage.dtype))
	}
	if dstStorage.count > math.MaxUint32 {
		return state.reject("fill", fmt.Errorf("%w: element count exceeds the Metal fill index range", ErrInvalid))
	}
	if err = state.acquire(destination); err != nil {
		return state.reject("fill", err)
	}
	if err = state.commands.encodeFill(dstStorage.buffer, value); err != nil {
		state.status = scopePoisoned
		return fmt.Errorf("gpu: encode fill: %w", err)
	}
	return nil
}

// Copy records a full transfer between distinct tensors of identical shape and
// DType on this device. An earlier write in the same scope can initialize source.
func (s *Scope) Copy(source, destination *Tensor) (err error) {
	var state *scopeState
	if state, err = s.beginEncoding(); err != nil {
		return err
	}
	defer state.mutex.Unlock()

	if source == nil || source.state == nil || destination == nil || destination.state == nil {
		return state.reject("copy", fmt.Errorf("%w: missing source or destination", ErrInvalid))
	}
	var (
		srcStorage = source.state.storage
		dstStorage = destination.state.storage
	)
	if srcStorage.dtype != dstStorage.dtype {
		return state.reject("copy", fmt.Errorf("%w: tensor dtypes differ (%s vs %s)", ErrInvalid, srcStorage.dtype, dstStorage.dtype))
	}
	if !slices.Equal(source.state.shape, destination.state.shape) {
		return state.reject("copy", fmt.Errorf("%w: tensor shapes differ", ErrInvalid))
	}
	if err = state.acquire(destination, source); err != nil {
		return state.reject("copy", err)
	}
	if err = state.commands.encodeCopy(srcStorage.buffer, dstStorage.buffer); err != nil {
		state.status = scopePoisoned
		return fmt.Errorf("gpu: encode copy: %w", err)
	}
	return nil
}

// GatherRows records a full copy from source[R,C] into destination[len(rows),C].
// Supported DType combinations are matching dtypes (Float32→Float32,
// Float16→Float16, BFloat16→BFloat16) and dequantizing gathers into Float32
// (Float16→Float32, BFloat16→Float32). Each row index is zero-based, and
// repeated indices are allowed. The indices are consumed during recording.
// Source must be initialized or written earlier in this scope. Destination
// must be distinct and on the same device.
func (s *Scope) GatherRows(source *Tensor, rows []int64, destination *Tensor) (err error) {
	var state *scopeState
	if state, err = s.beginEncoding(); err != nil {
		return err
	}
	defer state.mutex.Unlock()

	if source == nil || source.state == nil || len(source.state.shape) != 2 ||
		destination == nil || destination.state == nil || len(destination.state.shape) != 2 {
		return state.reject("gather rows", fmt.Errorf("%w: operands must be rank-two tensors", ErrInvalid))
	}
	var (
		srcStorage = source.state.storage
		dstStorage = destination.state.storage
		srcDType   = srcStorage.dtype
		dstDType   = dstStorage.dtype
	)
	if !srcDType.Valid() || !dstDType.Valid() || srcDType.IsQuantized() || dstDType.IsQuantized() ||
		(srcDType != dstDType && !(dstDType == Float32 && srcDType.is16Bit())) {
		return state.reject("gather rows", fmt.Errorf("%w: unsupported gather dtype conversion %s to %s", ErrInvalid, srcDType, dstDType))
	}
	var (
		sourceRows = source.state.shape[0]
		columns    = source.state.shape[1]
	)
	if len(rows) == 0 || destination.state.shape[0] != int64(len(rows)) || destination.state.shape[1] != columns {
		return state.reject("gather rows", fmt.Errorf("%w: destination must match the index count and source width", ErrInvalid))
	}
	if srcDType != dstDType && uint64(columns) > math.MaxUint32 {
		return state.reject("gather rows", fmt.Errorf("%w: column count exceeds the Metal mixed gather index range", ErrInvalid))
	}
	for i, row := range rows {
		if row < 0 || row >= sourceRows {
			return state.reject("gather rows", fmt.Errorf("%w: row index %d at offset %d is outside [0,%d)", ErrInvalid, row, i, sourceRows))
		}
	}
	if err = state.acquire(destination, source); err != nil {
		return state.reject("gather rows", err)
	}
	if err = state.commands.encodeGatherRows(srcStorage.buffer, dstStorage.buffer, uint64(columns), rows, srcDType, dstDType); err != nil {
		state.status = scopePoisoned
		return fmt.Errorf("gpu: encode gather rows: %w", err)
	}
	return nil
}

// Add records an elementwise FP32 sum. Operands must have identical shapes,
// contain at most 2³²−1 elements each, and belong to the scope's device. Inputs
// may share an allocation and must be initialized or written earlier in this
// scope. Destination must be distinct from the inputs and is fully overwritten.
// Addition follows FP32 arithmetic, including NaN for opposing infinities.
// NaN payloads, signed zero and subnormal preservation are unspecified.
func (s *Scope) Add(left, right, destination *Tensor) (err error) {
	return s.elementwise("add", device.ElementwiseAdd, destination, left, right)
}

// Mul records an elementwise FP32 product with the shape and ownership rules
// of Add. NaNs propagate, and zero times infinity produces NaN.
func (s *Scope) Mul(left, right, destination *Tensor) (err error) {
	return s.elementwise("mul", device.ElementwiseMul, destination, left, right)
}

// Sub records an elementwise FP32 difference (left - right) with the shape and
// ownership rules of Add. Subtraction follows IEEE-754 FP32 arithmetic.
func (s *Scope) Sub(left, right, destination *Tensor) (err error) {
	return s.elementwise("sub", device.ElementwiseSub, destination, left, right)
}

// Div records an elementwise FP32 quotient (left / right) with the shape and
// ownership rules of Add. Division follows IEEE-754 FP32 arithmetic, including
// ±Inf for nonzero division by zero and NaN for 0/0 or Inf/Inf.
func (s *Scope) Div(left, right, destination *Tensor) (err error) {
	return s.elementwise("div", device.ElementwiseDiv, destination, left, right)
}

// Scale records input * scale in FP32 into destination with the shape and
// ownership rules of SiLU. Scale may be any float32 value (including zero,
// ±Inf, or NaN), following IEEE-754 multiplication semantics.
func (s *Scope) Scale(input, destination *Tensor, scale float32) (err error) {
	var state *scopeState
	if state, err = s.beginEncoding(); err != nil {
		return err
	}
	defer state.mutex.Unlock()

	if input == nil || input.state == nil || destination == nil || destination.state == nil {
		return state.reject("scale", fmt.Errorf("%w: missing input or destination", ErrInvalid))
	}
	var (
		srcStorage = input.state.storage
		dstStorage = destination.state.storage
	)
	if srcStorage.dtype != Float32 {
		return state.reject("scale", fmt.Errorf("%w: scale requires float32 tensors, got %s", ErrInvalid, srcStorage.dtype))
	}
	if dstStorage.dtype != Float32 {
		return state.reject("scale", fmt.Errorf("%w: scale requires float32 tensors, got %s", ErrInvalid, dstStorage.dtype))
	}
	if !slices.Equal(input.state.shape, destination.state.shape) {
		return state.reject("scale", fmt.Errorf("%w: tensor shapes differ", ErrInvalid))
	}
	if dstStorage.count > math.MaxUint32 {
		return state.reject("scale", fmt.Errorf("%w: tensor element count exceeds the Metal scale index range", ErrInvalid))
	}
	if err = state.acquire(destination, input); err != nil {
		return state.reject("scale", err)
	}
	if err = state.commands.encodeScale(srcStorage.buffer, dstStorage.buffer, scale); err != nil {
		state.status = scopePoisoned
		return fmt.Errorf("gpu: encode scale: %w", err)
	}
	return nil
}

// AddBroadcast records an elementwise FP32 sum using ONNX/NumPy right-aligned
// multidirectional broadcasting for ranks 1 through 6. Operands must be
// Float32 on the scope's device with at most 2³²−1 elements each, and
// destination must have the exact broadcast output shape without aliasing
// either input (including through Tensor.View).
func (s *Scope) AddBroadcast(left, right, destination *Tensor) (err error) {
	return s.broadcast("add broadcast", device.BroadcastAdd, device.ElementwiseAdd, left, right, destination)
}

// SubBroadcast records an elementwise FP32 difference (left - right) with the
// multidirectional broadcasting and ownership rules of AddBroadcast.
func (s *Scope) SubBroadcast(left, right, destination *Tensor) (err error) {
	return s.broadcast("sub broadcast", device.BroadcastSub, device.ElementwiseSub, left, right, destination)
}

// MulBroadcast records an elementwise FP32 product with the multidirectional
// broadcasting and ownership rules of AddBroadcast.
func (s *Scope) MulBroadcast(left, right, destination *Tensor) (err error) {
	return s.broadcast("mul broadcast", device.BroadcastMul, device.ElementwiseMul, left, right, destination)
}

// DivBroadcast records an elementwise FP32 quotient (left / right) with the
// multidirectional broadcasting and ownership rules of AddBroadcast.
func (s *Scope) DivBroadcast(left, right, destination *Tensor) (err error) {
	return s.broadcast("div broadcast", device.BroadcastDiv, device.ElementwiseDiv, left, right, destination)
}

const maxBroadcastRank = 6

func (s *Scope) broadcast(
	name string,
	operation device.BroadcastOperation,
	contiguousOp device.ElementwiseOperation,
	left, right, destination *Tensor,
) (err error) {
	var state *scopeState
	if state, err = s.beginEncoding(); err != nil {
		return err
	}
	defer state.mutex.Unlock()

	if left == nil || left.state == nil || right == nil || right.state == nil || destination == nil || destination.state == nil {
		return state.reject(name, fmt.Errorf("%w: missing operand", ErrInvalid))
	}
	var (
		leftStorage  = left.state.storage
		rightStorage = right.state.storage
		dstStorage   = destination.state.storage
	)
	if err = requireFloat32(name, left, right, destination); err != nil {
		return state.reject(name, err)
	}
	if leftStorage.count > math.MaxUint32 || rightStorage.count > math.MaxUint32 || dstStorage.count > math.MaxUint32 {
		return state.reject(name, fmt.Errorf("%w: tensor element count exceeds the Metal broadcast index range", ErrInvalid))
	}
	var (
		expectedShape [maxBroadcastRank]int64
		expectedRank  int
		spec          device.BroadcastSpec
		contiguous    bool
	)
	if expectedShape, expectedRank, spec, contiguous, err = buildBroadcastSpec(left.state.shape, right.state.shape); err != nil {
		return state.reject(name, err)
	}
	if !slices.Equal(destination.state.shape, expectedShape[:expectedRank]) {
		return state.reject(name, fmt.Errorf("%w: destination shape %v must match broadcast shape %v", ErrInvalid, destination.state.shape, expectedShape[:expectedRank]))
	}
	if err = state.acquire(destination, left, right); err != nil {
		return state.reject(name, err)
	}
	if contiguous {
		if err = state.commands.encodeElementwise(leftStorage.buffer, rightStorage.buffer, dstStorage.buffer, contiguousOp); err != nil {
			state.status = scopePoisoned
			return fmt.Errorf("gpu: encode %s: %w", name, err)
		}
		return nil
	}
	if err = state.commands.encodeBroadcast(leftStorage.buffer, rightStorage.buffer, dstStorage.buffer, operation, spec); err != nil {
		state.status = scopePoisoned
		return fmt.Errorf("gpu: encode %s: %w", name, err)
	}
	return nil
}

func buildBroadcastSpec(leftShape, rightShape []int64) (outShape [maxBroadcastRank]int64, outRank int, spec device.BroadcastSpec, contiguous bool, err error) {
	var (
		leftRank  = len(leftShape)
		rightRank = len(rightShape)
	)
	if leftRank == 0 || leftRank > maxBroadcastRank || rightRank == 0 || rightRank > maxBroadcastRank {
		return outShape, 0, spec, false, fmt.Errorf("%w: broadcast operands must have rank 1..%d, got %d and %d",
			ErrInvalid, maxBroadcastRank, leftRank, rightRank)
	}
	outRank = max(leftRank, rightRank)

	var (
		leftRaw  [maxBroadcastRank]uint64
		rightRaw [maxBroadcastRank]uint64
		stride   uint64 = 1
	)
	for i := leftRank - 1; i >= 0; i-- {
		var outIdx = outRank - leftRank + i
		if leftShape[i] == 1 {
			leftRaw[outIdx] = 0
		} else {
			leftRaw[outIdx] = stride
			stride *= uint64(leftShape[i])
		}
	}
	stride = 1
	for i := rightRank - 1; i >= 0; i-- {
		var outIdx = outRank - rightRank + i
		if rightShape[i] == 1 {
			rightRaw[outIdx] = 0
		} else {
			rightRaw[outIdx] = stride
			stride *= uint64(rightShape[i])
		}
	}

	var total uint64 = 1
	for i := range outRank {
		var (
			lDim int64 = 1
			rDim int64 = 1
		)
		if i >= outRank-leftRank {
			lDim = leftShape[i-(outRank-leftRank)]
		}
		if i >= outRank-rightRank {
			rDim = rightShape[i-(outRank-rightRank)]
		}
		if lDim != rDim && lDim != 1 && rDim != 1 {
			return outShape, 0, spec, false, fmt.Errorf("%w: incompatible broadcast shapes %v and %v", ErrInvalid, leftShape, rightShape)
		}
		var outDim = max(lDim, rDim)
		outShape[i] = outDim
		if total > math.MaxUint32/uint64(outDim) {
			return outShape, 0, spec, false, fmt.Errorf("%w: broadcast element count exceeds the Metal index range", ErrInvalid)
		}
		total *= uint64(outDim)
	}

	if slices.Equal(leftShape, rightShape) {
		return outShape, outRank, spec, true, nil
	}

	var (
		dims         [maxBroadcastRank]uint64
		leftStrides  [maxBroadcastRank]uint64
		rightStrides [maxBroadcastRank]uint64
		rank         int
	)
	for i := range outRank {
		if outShape[i] == 1 {
			continue
		}
		var (
			dim = uint64(outShape[i])
			ls  = leftRaw[i]
			rs  = rightRaw[i]
		)
		if rank > 0 && leftStrides[rank-1] == ls*dim && rightStrides[rank-1] == rs*dim {
			dims[rank-1] *= dim
			leftStrides[rank-1] = ls
			rightStrides[rank-1] = rs
			continue
		}
		dims[rank] = dim
		leftStrides[rank] = ls
		rightStrides[rank] = rs
		rank++
	}

	if rank == 0 || (rank == 1 && leftStrides[0] == 1 && rightStrides[0] == 1) {
		return outShape, outRank, spec, true, nil
	}

	spec.Rank = uint32(rank)
	spec.Count = uint32(total)
	var outStride uint64 = 1
	for i := rank - 1; i >= 0; i-- {
		spec.Dims[i] = uint32(dims[i])
		spec.OutStrides[i] = uint32(outStride)
		spec.LeftStrides[i] = uint32(leftStrides[i])
		spec.RightStrides[i] = uint32(rightStrides[i])
		outStride *= dims[i]
	}
	return outShape, outRank, spec, false, nil
}

// SiLU records input times its logistic sigmoid, evaluated in FP32 with a
// stable sigmoid for negative inputs. It uses the shape and ownership rules
// of Add. NaN propagates, positive infinity yields positive infinity, and
// negative infinity times its zero sigmoid yields NaN.
func (s *Scope) SiLU(input, destination *Tensor) (err error) {
	return s.elementwise("silu", device.ElementwiseSiLU, destination, input)
}

// SwiGLU records SiLU(gate) times up in one elementwise FP32 dispatch. It uses
// the shape, ownership and arithmetic rules of SiLU and Mul.
func (s *Scope) SwiGLU(gate, up, destination *Tensor) (err error) {
	return s.elementwise("swiglu", device.ElementwiseSwiGLU, destination, gate, up)
}

// ReLU records max(input, 0) in FP32 while propagating NaN. It uses the shape
// and ownership rules of SiLU.
func (s *Scope) ReLU(input, destination *Tensor) (err error) {
	return s.elementwise("relu", device.ElementwiseReLU, destination, input)
}

// Sigmoid records the numerically stable FP32 logistic sigmoid
// 1 / (1 + exp(-x)). It uses the shape and ownership rules of SiLU. Positive
// infinity yields 1, negative infinity yields 0, and NaN propagates.
func (s *Scope) Sigmoid(input, destination *Tensor) (err error) {
	return s.elementwise("sigmoid", device.ElementwiseSigmoid, destination, input)
}

// Tanh records the numerically stable FP32 hyperbolic tangent, saturating to
// ±1 for large finite magnitudes and ±Inf while propagating NaN. It uses the
// shape and ownership rules of SiLU.
func (s *Scope) Tanh(input, destination *Tensor) (err error) {
	return s.elementwise("tanh", device.ElementwiseTanh, destination, input)
}

// Softmax records a numerically stable FP32 softmax along the last axis
// (len(shape)-1). Operands must be Float32 with identical shapes, contain at
// most 2³²−1 elements, belong to the scope's device, and not alias each other
// (including through Tensor.View). For finite inputs, an axis of length 1
// yields 1.0; rows containing NaN, +Inf, or all -Inf produce NaN across the
// row, while finite rows with -Inf entries map -Inf to 0.
func (s *Scope) Softmax(input, destination *Tensor) (err error) {
	return s.softmax("softmax", input, destination, 0, true)
}

// SoftmaxAxis records a numerically stable FP32 softmax along axis, where
// 0 <= axis < len(input.Shape()). It uses the shape, dtype, ownership, and
// edge-case semantics of Softmax.
func (s *Scope) SoftmaxAxis(input, destination *Tensor, axis int) (err error) {
	return s.softmax("softmax", input, destination, axis, false)
}

func (s *Scope) softmax(name string, input, destination *Tensor, axis int, useLastAxis bool) (err error) {
	var state *scopeState
	if state, err = s.beginEncoding(); err != nil {
		return err
	}
	defer state.mutex.Unlock()

	if input == nil || input.state == nil || destination == nil || destination.state == nil {
		return state.reject(name, fmt.Errorf("%w: missing input or destination", ErrInvalid))
	}
	var (
		srcStorage = input.state.storage
		dstStorage = destination.state.storage
	)
	if srcStorage.dtype != Float32 {
		return state.reject(name, fmt.Errorf("%w: %s requires float32 tensors, got %s", ErrInvalid, name, srcStorage.dtype))
	}
	if dstStorage.dtype != Float32 {
		return state.reject(name, fmt.Errorf("%w: %s requires float32 tensors, got %s", ErrInvalid, name, dstStorage.dtype))
	}
	if !slices.Equal(input.state.shape, destination.state.shape) {
		return state.reject(name, fmt.Errorf("%w: tensor shapes differ", ErrInvalid))
	}
	if dstStorage.count > math.MaxUint32 {
		return state.reject(name, fmt.Errorf("%w: tensor element count exceeds the Metal softmax index range", ErrInvalid))
	}
	var (
		shape = input.state.shape
		rank  = len(shape)
	)
	if useLastAxis {
		axis = rank - 1
	}
	if axis < 0 || axis >= rank {
		return state.reject(name, fmt.Errorf("%w: softmax axis %d out of range for rank %d", ErrInvalid, axis, rank))
	}
	var (
		outer    uint64 = 1
		axisSize        = uint64(shape[axis])
		inner    uint64 = 1
	)
	for i := range axis {
		outer *= uint64(shape[i])
	}
	for i := axis + 1; i < rank; i++ {
		inner *= uint64(shape[i])
	}
	if err = state.acquire(destination, input); err != nil {
		return state.reject(name, err)
	}
	if err = state.commands.encodeSoftmax(srcStorage.buffer, dstStorage.buffer, uint32(outer), uint32(axisSize), uint32(inner)); err != nil {
		state.status = scopePoisoned
		return fmt.Errorf("gpu: encode %s: %w", name, err)
	}
	return nil
}

func (s *Scope) elementwise(name string, operation device.ElementwiseOperation, destination *Tensor, inputs ...*Tensor) (err error) {
	var state *scopeState
	if state, err = s.beginEncoding(); err != nil {
		return err
	}
	defer state.mutex.Unlock()

	if destination == nil || destination.state == nil {
		return state.reject(name, fmt.Errorf("%w: missing destination", ErrInvalid))
	}
	var dstStorage = destination.state.storage
	if dstStorage.dtype != Float32 {
		return state.reject(name, fmt.Errorf("%w: %s requires float32 tensors, got %s", ErrInvalid, name, dstStorage.dtype))
	}
	if dstStorage.count > math.MaxUint32 {
		return state.reject(name, fmt.Errorf("%w: tensor element count exceeds the Metal elementwise index range", ErrInvalid))
	}
	for _, input := range inputs {
		if input == nil || input.state == nil {
			return state.reject(name, fmt.Errorf("%w: missing input", ErrInvalid))
		}
		if input.state.storage.dtype != Float32 {
			return state.reject(name, fmt.Errorf("%w: %s requires float32 tensors, got %s", ErrInvalid, name, input.state.storage.dtype))
		}
		if !slices.Equal(input.state.shape, destination.state.shape) {
			return state.reject(name, fmt.Errorf("%w: tensor shapes differ", ErrInvalid))
		}
	}
	if err = state.acquire(destination, inputs...); err != nil {
		return state.reject(name, err)
	}

	var second allocation
	if len(inputs) == 2 {
		second = inputs[1].state.storage.buffer
	}
	if err = state.commands.encodeElementwise(inputs[0].state.storage.buffer, second, dstStorage.buffer, operation); err != nil {
		state.status = scopePoisoned
		return fmt.Errorf("gpu: encode %s: %w", name, err)
	}
	return nil
}

// MatMul records destination[M,N] = left[M,K] × right[K,N]. All tensors must
// be contiguous row-major matrices with positive dimensions and at most
// 2³²−1 elements each. Left and destination must be Float32; right may be
// Float32, Float16, or BFloat16. Use M=1 for a single-row projection. Operands
// must be on this device. The inputs may share an allocation. They must already
// be initialized or written by an earlier operation in the same scope.
// Destination is fully overwritten and must be distinct from both inputs.
//
// Each dot product accumulates in FP32 and may use fused multiply-add. For
// Float32 × Float32, accumulation proceeds in contraction order. NaNs
// propagate, and infinite operands follow IEEE arithmetic, including NaN for
// zero times infinity or addition of opposite infinities. NaN payloads, signed
// zero and subnormal preservation are unspecified.
func (s *Scope) MatMul(left, right, destination *Tensor) (err error) {
	return s.matMul(left, right, destination, false)
}

// MatMulRightTranspose records destination[M,N] = left[M,K] × right[N,K]ᵀ
// without moving or transposing the stored right operand. It uses the same
// row-major output, ownership, shape limits, and FP32 accumulation semantics as
// MatMul, and supports Float32, Float16, BFloat16, Q8_0, and Q4_0 right weights
// (with K % 32 == 0 required for Q8_0 and Q4_0).
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
		if operand.state.storage.count > math.MaxUint32 {
			return state.reject("matmul", fmt.Errorf("%w: tensor element count exceeds the Metal matmul index range", ErrInvalid))
		}
	}
	var (
		leftStorage  = left.state.storage
		rightStorage = right.state.storage
		dstStorage   = destination.state.storage
	)
	if leftStorage.dtype != Float32 {
		return state.reject("matmul", fmt.Errorf("%w: matmul requires float32 left operand, got %s", ErrInvalid, leftStorage.dtype))
	}
	if !rightStorage.dtype.Valid() {
		return state.reject("matmul", fmt.Errorf("%w: matmul right operand has unsupported dtype %s", ErrInvalid, rightStorage.dtype))
	}
	if rightStorage.dtype.IsQuantized() && !rightTransposed {
		return state.reject("matmul", fmt.Errorf("%w: %s weights require MatMulRightTranspose", ErrInvalid, rightStorage.dtype))
	}
	if dstStorage.dtype != Float32 {
		return state.reject("matmul", fmt.Errorf("%w: matmul requires float32 destination, got %s", ErrInvalid, dstStorage.dtype))
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
	if rightStorage.dtype.IsQuantized() && k%QuantBlockSize != 0 {
		return state.reject("matmul", fmt.Errorf("%w: %s inner dimension %d must be a multiple of %d", ErrInvalid, rightStorage.dtype, k, QuantBlockSize))
	}
	if err = state.acquire(destination, left, right); err != nil {
		return state.reject("matmul", err)
	}
	if err = state.commands.encodeMatMul(leftStorage.buffer, rightStorage.buffer, dstStorage.buffer,
		uint32(m), uint32(n), uint32(k), rightTransposed, rightStorage.dtype); err != nil {
		state.status = scopePoisoned
		return fmt.Errorf("gpu: encode matmul: %w", err)
	}
	return nil
}

// RMSNorm records input / sqrt(mean(input²) + epsilon) times weight for
// each row of the last axis. Input and destination must be Float32 with
// matching shapes; weight is a rank-one Float32, Float16, or BFloat16 vector
// matching that axis. Variance accumulates in FP32. Epsilon must be finite
// and positive. Destination is fully written and must be distinct from both inputs.
func (s *Scope) RMSNorm(input, weight, destination *Tensor, epsilon float32) (err error) {
	var state *scopeState
	if state, err = s.beginEncoding(); err != nil {
		return err
	}
	defer state.mutex.Unlock()

	if input == nil || input.state == nil || input.state.storage.count > math.MaxUint32 ||
		!hasShape(destination, input.state.shape...) {
		return state.reject("RMSNorm", fmt.Errorf("%w: invalid input or destination shape or index range", ErrInvalid))
	}
	var width = input.state.shape[len(input.state.shape)-1]
	if !hasShape(weight, width) || !(epsilon > 0 && epsilon <= math.MaxFloat32) {
		return state.reject("RMSNorm", fmt.Errorf("%w: expected a matching last-axis weight and finite positive epsilon", ErrInvalid))
	}
	if err = requireFloat32("RMSNorm", input, destination); err != nil {
		return state.reject("RMSNorm", err)
	}
	var operation device.AttentionMathOperation
	switch weight.state.storage.dtype {
	case Float32:
		operation = device.TensorRMSNorm
	case Float16:
		operation = device.TensorRMSNormFloat16
	case BFloat16:
		operation = device.TensorRMSNormBFloat16
	default:
		return state.reject("RMSNorm", fmt.Errorf("%w: RMSNorm weight has unsupported dtype %s", ErrInvalid, weight.state.storage.dtype))
	}
	if err = state.acquire(destination, input, weight); err != nil {
		return state.reject("RMSNorm", err)
	}
	var rootEpsilon = float32(math.Sqrt(float64(epsilon)))
	var params = [6]uint32{uint32(width), math.Float32bits(rootEpsilon)}
	return state.encodeAttentionMath("RMSNorm", operation, input, weight, destination, nil, params, uint32(input.state.storage.count/uint64(width)))
}

// RotatePairs applies (x0*cos - x1*sin, x1*cos + x0*sin) to adjacent FP32
// feature pairs within each head. Input and destination are token-major
// [rows, heads*headWidth] or [rows, heads, headWidth] with an even head width.
// The matching coefficient tables [positions, headWidth/2] start at absolute
// tableStart and broadcast over heads. The first input row is at position.
func (s *Scope) RotatePairs(input, cosine, sine, destination *Tensor, headWidth, position, tableStart int64) (err error) {
	var state *scopeState
	if state, err = s.beginEncoding(); err != nil {
		return err
	}
	defer state.mutex.Unlock()

	var rows, heads = headRows(input, headWidth)
	if rows == 0 || headWidth%2 != 0 || !hasShape(destination, input.state.shape...) {
		return state.reject("rotate pairs", fmt.Errorf("%w: invalid packed head shape, width or destination", ErrInvalid))
	}
	if cosine == nil || cosine.state == nil || len(cosine.state.shape) != 2 ||
		cosine.state.storage.count > math.MaxUint32 || cosine.state.shape[1] != headWidth/2 ||
		!hasShape(sine, cosine.state.shape...) {
		return state.reject("rotate pairs", fmt.Errorf("%w: invalid cosine or sine table shape", ErrInvalid))
	}
	if err = requireFloat32("rotate pairs", input, cosine, sine, destination); err != nil {
		return state.reject("rotate pairs", err)
	}
	var tableRows = cosine.state.shape[0]
	if tableStart < 0 || position < tableStart || tableStart > math.MaxInt64-(tableRows-1) {
		return state.reject("rotate pairs", fmt.Errorf("%w: invalid absolute coefficient positions", ErrInvalid))
	}
	var offset = position - tableStart
	if offset > tableRows || rows > tableRows-offset {
		return state.reject("rotate pairs", fmt.Errorf("%w: input positions are outside the coefficient table", ErrInvalid))
	}
	if err = state.acquire(destination, input, cosine, sine); err != nil {
		return state.reject("rotate pairs", err)
	}
	var params = [6]uint32{uint32(headWidth / 2), uint32(heads), uint32(offset)}
	return state.encodeAttentionMath("rotate pairs", device.TensorRotatePairs, input, cosine, destination, sine, params, uint32(input.state.storage.count/2))
}

// AppendKV stages positive consecutive rows at the cache's committed position.
// Key and value inputs are token-major [rows, heads*width] or [rows, heads, width].
// The cache owns a disjoint staging slot for the existing prefix and new rows.
// CausalAttention in this scope can use those rows in encode order. Other
// scopes see the committed prefix until the returned update is published.
func (s *Scope) AppendKV(cache *KVCache, keys, values *Tensor, position int64) (update *KVAppend, err error) {
	var state *scopeState
	if state, err = s.beginEncoding(); err != nil {
		return nil, err
	}
	defer state.mutex.Unlock()

	var (
		appendState *kvAppendState
		history     *Tensor
		destination *Tensor
	)
	if appendState, history, destination, err = state.reserveAppend(cache, keys, values, position); err != nil {
		return nil, state.reject("append KV", err)
	}
	var (
		rowWidth = appendState.cache.heads * appendState.cache.width
		active   allocation
	)
	if history != nil {
		active = history.state.storage.buffer
	}
	if err = state.commands.encodeKVAppend(active, keys.state.storage.buffer, values.state.storage.buffer, destination.state.storage.buffer,
		uint32(appendState.cache.capacity*rowWidth), uint32(position*rowWidth), appendState.cache.dtype); err != nil {
		state.status = scopePoisoned
		return nil, fmt.Errorf("gpu: encode append KV: %w", err)
	}
	return &KVAppend{state: appendState}, nil
}

func (s *scopeState) reserveAppend(cache *KVCache, keys, values *Tensor, position int64) (update *kvAppendState, history, dest *Tensor, err error) {
	var state *kvCacheState
	if state, err = s.cacheState(cache); err != nil {
		return nil, nil, nil, err
	}
	var (
		rows, keyHeads        = headRows(keys, state.width)
		valueRows, valueHeads = headRows(values, state.width)
	)
	if rows == 0 || keyHeads != state.heads || valueRows != rows || valueHeads != state.heads ||
		position < 0 || position > state.capacity || rows > state.capacity-position {
		return nil, nil, nil, fmt.Errorf("%w: invalid append shape or cache bounds", ErrInvalid)
	}
	if err = requireFloat32("append KV", keys, values); err != nil {
		return nil, nil, nil, err
	}

	s.device.mutex.Lock()
	defer s.device.mutex.Unlock()
	if state.closed {
		return nil, nil, nil, ErrClosed
	}
	if state.pending != nil {
		return nil, nil, nil, ErrBusy
	}
	if position != state.length {
		return nil, nil, nil, fmt.Errorf("%w: append position must equal committed KV length", ErrInvalid)
	}
	dest = state.slots[1-state.active]
	var sources = []*Tensor{keys, values}
	if position > 0 {
		history = state.slots[state.active]
		sources = append(sources, history)
	}
	if err = s.acquireLocked(dest, sources...); err != nil {
		return nil, nil, nil, err
	}
	update = &kvAppendState{cache: state, scope: s, length: position + rows}
	state.pending = update
	s.kvAppends = append(s.kvAppends, update)
	return update, history, dest, nil
}

// CausalAttention records scaled dot-product attention on absolute query
// positions starting at position. Queries and destination use matching
// token-major head shapes as in RotatePairs, with any positive head width.
// The number of query heads must be a multiple of the number of cache heads.
// Each consecutive group of query heads shares one KV head. Only populated
// cache rows at or before the query position contribute. An empty cache
// yields zeros. Use finite inputs with representable FP32 products and dot sums.
func (s *Scope) CausalAttention(queries *Tensor, cache *KVCache, destination *Tensor, position int64) (err error) {
	var state *scopeState
	if state, err = s.beginEncoding(); err != nil {
		return err
	}
	defer state.mutex.Unlock()
	var cacheState *kvCacheState
	if cacheState, err = state.cacheState(cache); err != nil {
		return state.reject("causal attention", err)
	}
	var rows, heads = headRows(queries, cacheState.width)
	if rows == 0 || heads%cacheState.heads != 0 || !hasShape(destination, queries.state.shape...) ||
		position < 0 || position > math.MaxInt64-(rows-1) {
		return state.reject("causal attention", fmt.Errorf("%w: invalid query shape, groups, destination or positions", ErrInvalid))
	}
	if err = requireFloat32("causal attention", queries, destination); err != nil {
		return state.reject("causal attention", err)
	}
	var (
		history *Tensor
		length  int64
	)
	if history, length, err = state.acquireAttention(cacheState, queries, destination); err != nil {
		return state.reject("causal attention", err)
	}
	if length == 0 {
		if err = state.commands.encodeFill(destination.state.storage.buffer, 0); err != nil {
			state.status = scopePoisoned
			return fmt.Errorf("gpu: encode empty causal attention: %w", err)
		}
		return nil
	}
	var (
		plane     = cacheState.capacity * cacheState.heads * cacheState.width
		params    = [6]uint32{uint32(cacheState.width), uint32(heads), uint32(cacheState.heads), uint32(plane), uint32(length), uint32(min(position, length))}
		operation = device.TensorCausalAttention
	)
	if cacheState.dtype == Float16 {
		operation = device.TensorCausalAttentionFloat16
	}
	return state.encodeAttentionMath("causal attention", operation, queries, history, destination, nil, params, uint32(rows*heads))
}

func (s *scopeState) cacheState(cache *KVCache) (state *kvCacheState, err error) {
	if cache == nil || cache.state == nil {
		return nil, ErrClosed
	}
	if cache.state.device != s.device {
		return nil, fmt.Errorf("%w: KV cache belongs to another device", ErrInvalid)
	}
	return cache.state, nil
}

func (s *scopeState) acquireAttention(cache *kvCacheState, queries, destination *Tensor) (history *Tensor, length int64, err error) {
	s.device.mutex.Lock()
	defer s.device.mutex.Unlock()
	if cache.closed {
		return nil, 0, ErrClosed
	}
	var slot = cache.active
	length = cache.length
	if cache.pending != nil && cache.pending.scope == s && !cache.pending.discarded {
		slot = 1 - slot
		length = cache.pending.length
	}

	var sources = []*Tensor{queries}
	if length > 0 {
		history = cache.slots[slot]
		sources = append(sources, history)
	}
	if err = s.acquireLocked(destination, sources...); err != nil {
		return nil, 0, err
	}
	return history, length, nil
}

func (s *scopeState) encodeAttentionMath(name string, operation device.AttentionMathOperation, first, second, target, extra *Tensor, params [6]uint32, threads uint32) (err error) {
	var additional allocation
	if extra != nil {
		additional = extra.state.storage.buffer
	}
	if err = s.commands.encodeAttentionMath(operation, first.state.storage.buffer, second.state.storage.buffer, target.state.storage.buffer, additional, params, threads); err != nil {
		s.status = scopePoisoned
		return fmt.Errorf("gpu: encode %s: %w", name, err)
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
		return state.finish(scopeDiscarded, err, false)
	}
	if len(state.leases) == 0 {
		return fmt.Errorf("%w: scope has no operations", ErrInvalid)
	}
	if err = state.commands.Commit(); err != nil {
		return state.finish(scopeDiscarded, fmt.Errorf("gpu: submit scope: %w", err), false)
	}

	state.status = scopeSubmitted
	return nil
}

// Wait observes a submitted scope's terminal result. A successful observation
// makes staged KV appends eligible for explicit publication. Cancellation
// while work is outstanding leaves leases held for a later Wait or Close.
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
		state.finish(scopeFinished, completion.err, true)
	}
	return state.result
}

// Close discards unsubmitted work or drains a submission. If Close observes
// the command result first, it returns any execution error and discards
// staged KV appends. Close is idempotent.
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
			err = state.finish(scopeFinished, completion.err, false)
		}
	}

	defer state.mutex.Unlock()
	switch state.status {
	case scopeClosed:
		return nil
	case scopeRecording, scopePoisoned:
		err = state.finish(scopeDiscarded, nil, false)
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

func requireFloat32(operation string, tensors ...*Tensor) (err error) {
	for _, tensor := range tensors {
		if tensor == nil || tensor.state == nil {
			continue
		}
		if tensor.state.storage.dtype != Float32 {
			return fmt.Errorf("%w: %s requires float32 tensors, got %s", ErrInvalid, operation, tensor.state.storage.dtype)
		}
	}
	return nil
}

// acquire validates every operand before installing any new leases.
func (s *scopeState) acquire(destination *Tensor, sources ...*Tensor) (err error) {
	s.device.mutex.Lock()
	defer s.device.mutex.Unlock()
	return s.acquireLocked(destination, sources...)
}

func (s *scopeState) acquireLocked(destination *Tensor, sources ...*Tensor) (err error) {
	var target = destination.state
	if err = s.validateAccess(target, writeAccess); err != nil {
		return err
	}
	for _, source := range sources {
		if source.state.storage == target.storage {
			return fmt.Errorf("%w: destination aliases a source", ErrInvalid)
		}
		if err = s.validateAccess(source.state, readAccess); err != nil {
			return err
		}
	}

	var targetStorage = target.storage
	if s.leases[targetStorage] == readAccess {
		targetStorage.readers--
	}
	targetStorage.writer = s
	s.leases[targetStorage] = writeAccess
	for _, source := range sources {
		var sourceStorage = source.state.storage
		if s.leases[sourceStorage] == 0 {
			sourceStorage.readers++
			s.leases[sourceStorage] = readAccess
		}
	}
	return nil
}

func (s *scopeState) validateAccess(t *tensorState, access tensorAccess) (err error) {
	var storage = t.storage
	if storage.device != s.device {
		return fmt.Errorf("%w: tensor belongs to another device", ErrInvalid)
	}
	if t.closed {
		return ErrClosed
	}
	if storage.hostAccess || (storage.writer != nil && storage.writer != s) {
		return ErrBusy
	}
	if access == writeAccess && (storage.readers > 1 || (storage.readers == 1 && s.leases[storage] != readAccess)) {
		return ErrBusy
	}
	if access == readAccess && !storage.initialized && storage.writer != s {
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
func (s *scopeState) finish(status scopeStatus, commandErr error, observed bool) (err error) {
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
	for storage, access := range s.leases {
		if access == writeAccess {
			if status == scopeFinished {
				storage.initialized = commandErr == nil
			}
			storage.writer = nil
		} else {
			storage.readers--
		}
		storage.releaseIfIdle()
	}
	for _, update := range s.kvAppends {
		update.finish(observed && err == nil)
	}
	s.kvAppends = nil
	s.leases = nil
	return err
}
