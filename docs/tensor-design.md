# Inference tensor and kernel contract

Status: NN-01 design contract for the additive NN-02 through NN-12 work.
The NN-02 storage APIs are implemented; see the [storage handoff](handoffs/NN-02.md).
The NN-03 shared shape APIs are implemented; see the [shape handoff](handoffs/NN-03.md).
The NN-04 arithmetic APIs are implemented; see the [arithmetic handoff](handoffs/NN-04.md).
The remaining signatures are proposed APIs until their implementation tasks
document the shipped subset and tests.

## Boundary and compatibility

Add `github.com/itsmontoya/neuralnetwork/tensor` for concrete, contiguous,
row-major CPU values and forward numerical kernels. Keep storage and kernels in
one package so kernels can use private typed storage without publishing mutable
slices or introducing a backend interface. Use one file per primary type and
keep every method with its type. Function-only files may group related kernels.

This is the separate repository-wide tensor decision requested by NN-01. The
tensor exclusions in the [v1 scope](v1-scope-and-api.md),
[CNN design](cnn-design.md), and [RNN design](rnn-design.md) describe their
matrix-based milestones. Those contracts remain intact. The tensor package is
an additive inference backend; it does not replace `matrix.Matrix`,
`layer.Layer`, `model.Sequential`, training caches, optimizer parameters,
dataset containers, or JSON/binary model formats. Existing positive-size matrix
construction and matrix division's zero-denominator error remain unchanged.

The separate `github.com/itsmontoya/onnx` module owns wire decoding, IR/opset
selection, operator defaults, automatic padding, symbolic/unknown dimensions,
graph validation/planning, named values, scheduling, sessions, limits, and
diagnostics. This library accepts resolved mathematical arguments. No ONNX
protobuf, domain, schema, operator-version, or runtime type enters these APIs;
there is no import from this library back into that runtime. `Sequential` does
not become a graph executor.

The local runtime roadmap proposes IR 8/opset 13 and ONNX v1.16.2 as a starting
reference, but RT-01 has not pinned a release or completed its handoff. This
document establishes the library contract without claiming that the runtime
profile or reference conformance already exists.

## Storage APIs (NN-02)

```go
type DataType uint8

const (
	Invalid DataType = iota
	Float32
	Int64
)

func New(dataType DataType, shape []int64) (out *Tensor, err error)
func FromFloat32(shape []int64, values []float32) (out *Tensor, err error)
func FromInt64(shape []int64, values []int64) (out *Tensor, err error)

type Tensor struct { /* private shape, type, storage, and storage identity */ }

func (t *Tensor) Validate() (err error)
func (t *Tensor) DataType() (dataType DataType)
func (t *Tensor) Shape() (shape []int64)
func (t *Tensor) Rank() (rank int)
func (t *Tensor) Len() (elements int64)
func (t *Tensor) ByteSize() (bytes int64)
func (t *Tensor) Float32Values() (values []float32, err error)
func (t *Tensor) Int64Values() (values []int64, err error)
func (t *Tensor) Clone() (out *Tensor, err error)
```

`New` allocates zero-filled owned storage of exactly the selected type. Typed
constructors copy both shape and values and require an exact element count.
There is no generic `any` constructor, unsafe byte reinterpretation, public
backing slice, implicit type conversion, or borrowed-buffer constructor.
Typed accessors return copies and reject the other data type, including for
empty tensors. `Shape` also returns a copy. `Clone` copies shape and payload.
An integer such as `16777217` never passes through `float32`.

Dimensions are concrete nonnegative `int64` values. A nil or empty shape is
rank zero and has one element. A shape containing zero has zero elements.
`[0, 3]` is concrete, not unknown metadata. Row-major order makes the final
dimension vary fastest. There are no public strides or metadata-only views.

Validate dimensions, checked element products, type byte width, byte counts,
and conversion to host `int` before allocating the numeric payload. Reject a
dimension that cannot fit a host index even when another dimension is zero.
Inspect all dimensions before treating a product as zero, so zero cannot hide
an invalid negative dimension. Shape planning and later stride/offset arithmetic
must also reject unrepresentable intermediate values before use. Empty values
need not materialize otherwise unused nonempty iteration spaces.

Checked arithmetic prevents integer wraparound; it does not promise that every
representable allocation fits physical memory. Runtime allocation budgets must
be checked before calling constructors, using `ValidateShape`. Do not recover
arbitrary allocation/programmer panics and report success.

The zero `Tensor` value and a nil `*Tensor` are invalid, distinct from a valid
scalar or empty tensor. `Validate`, typed accessors, and `Clone` return errors
for either. Non-error metadata methods return `Invalid`, nil shape, and zero
rank/count/bytes for invalid receivers. Callers must use `Validate` or
`DataType` when that distinction matters.

## Ownership, failure, and concurrency

All `Into` functions read their inputs and fully overwrite an existing,
caller-owned destination with the exact required shape/type. They never resize
or retain that destination. Callers allocate it using a shape planner and
`New`. Input values, shape arguments, configuration values, and statistics are
never changed. Reusing a destination is explicit; a later unrelated call cannot
overwrite an earlier result. Values remain valid independently of the call or
runtime session that produced them.

The destination must not share storage with any input, even for an elementwise
operation, identity transform, empty result, or zero scaling coefficient.
Repeated inputs and read-only input/input aliases are allowed. A shallow Go
copy of a `Tensor` shares its private storage identity and is treated as an
alias; use `Clone` for independent storage. Alias validation uses that identity,
not just pointer equality or a first-element address (which fails for empties).
The private identity must use a nonzero-size allocation so distinct empty
tensors cannot receive coalesced addresses for zero-size tokens.
No API creates partially overlapping storage. Metadata-only aliases remain
deferred until a separate explicit read-only/lifetime contract is designed.

Published initializers and shared inputs are read-only by contract: callers
must not use them as output destinations or mutate them while readers exist.
Private slices prevent mutation through constructors/accessors; the package
does not add a frozen-value type or a mutable slice escape hatch. A runtime
that shares constants must retain this distinction when assigning destinations.

Before the first destination write, validate all tensors, types, dimensions,
axes/indices, derived sizes, configuration, and alias rules. In particular,
Gather validates every index first. A returned validation error leaves the
destination unchanged. Copying matrix adapters compute into local matrices
before copying back, so their returned errors also preserve the destination.
Valid floating-point domain errors produce documented NaN/infinity values,
not a partial write followed by a validation error. Errors have `tensor`
context and identify the offending argument; underlying operational errors
are wrapped with `%w`. No new public error sentinel taxonomy is required here.

Inputs may be read concurrently, including shared immutable weights.
Concurrent calls must have disjoint destinations and no destination overlapping
another call's inputs. Reading a destination during a write requires caller
synchronization. Kernels have no hidden mutable global scratch or training state.
Scratch belongs to the invocation. There is no public workspace API in this
first contract: introduce one in NN-13 only if measured runtime costs justify
it, with session/call ownership, size planning, and a copying fallback.
These synchronous kernels do not accept a context or promise cancellation
inside a kernel; the runtime must state that granularity until a later bounded
cancellation extension is implemented.

## Shared shape planning (NN-03 and each kernel task)

```go
func ValidateShape(shape []int64, dataType DataType) (elements, bytes int64, err error)
func NormalizeAxis(axis int64, rank int) (normalized int, err error)
func BroadcastShape(left, right []int64) (shape []int64, err error)
func ValidateBroadcastTo(source, destination []int64) (err error)
```

These public helpers serve runtime planning without allocating numeric payloads.
They validate concrete dimensions and checked host-representable shape arithmetic.
`ValidateShape` checks the selected type; the two broadcasting helpers check
float32 payload sizes for their numerical callers. Explicit stride requests
reject unrepresentable suffix products even for an empty shape; kernels with
empty outputs can skip unused stride planning after validating their arguments.
Returned shapes are independent copies. Private shared helpers provide shape
equality, contiguous strides, checked sums/products and offsets; kernels must
reuse them rather than duplicate unchecked arithmetic. Each later task owns its
operation-specific shape planners listed below. Shape planners also check the
relevant output element/byte size (both supported types for type-preserving
operations, float32 for numerical operations).

`NormalizeAxis` accepts `-rank <= axis < rank` and rejects every axis for rank
zero. Multi-axis operations normalize all axes before rejecting duplicates.
Flatten is a distinct split-axis case accepting `-rank <= axis <= rank`;
for a scalar only zero is valid. Unsqueeze axes refer to the final output rank.
Explicit empty axis lists mean no axes; they do not select an ONNX default.

Multidirectional broadcasting aligns trailing dimensions, treating absent
leading dimensions as one. Each pair must be equal or contain a one; choose the
other dimension when one occurs, so `0` with `1` produces `0`, while `0` with
`2` is invalid. Scalars broadcast to any concrete shape. Unidirectional
`ValidateBroadcastTo` additionally requires source rank no greater than target
rank and must produce exactly the target. It must never enlarge a Gemm output.

## Arithmetic and matrix products (NN-04, NN-05)

```go
func AddInto(left, right, destination *Tensor) (err error)
func SubInto(left, right, destination *Tensor) (err error)
func MulInto(left, right, destination *Tensor) (err error)
func DivInto(left, right, destination *Tensor) (err error)

func MatMulShape(left, right []int64) (shape []int64, err error)
func MatMulInto(left, right, destination *Tensor) (err error)

type GemmConfig struct {
	TransposeA bool
	TransposeB bool
	Alpha      float32
	Beta       float32
}

func GemmShape(left, right []int64, config GemmConfig) (shape []int64, err error)
func GemmInto(left, right, bias, destination *Tensor, config GemmConfig) (err error)
```

All these kernels require float32. Arithmetic uses `BroadcastShape` without
expanding either complete input. Div uses floating division: nonzero divided
by signed zero produces signed infinity, zero divided by zero produces NaN,
and NaN/infinity follow IEEE floating arithmetic. It cannot blindly reuse the
matrix division API, which rejects zero denominators. Exact NaN payloads are
not promised; fixtures must distinguish NaN, signed infinity, and signed zero.

NN-04 pins its selected arithmetic fixtures to ONNX 1.16.2/NumPy 1.26.4,
using IR 8/opset 13 reference graphs without finalizing RT-01's runtime profile.
Nonempty matching shapes reuse matrix Add/Sub/Mul through copying adapters;
broadcasting and Div use direct float32 arithmetic. All shape/type/identity and
index-plan validation precedes destination writes, including empty results.
The checked-in fixtures and generator are documented in
[`tensor/testdata/README.md`](../tensor/testdata/README.md).

MatMul rejects rank-zero inputs, promotes a left vector to `[1, K]` and a right
vector to `[K, 1]`, broadcasts batch prefixes, checks equal contraction lengths,
and removes the promoted result axes. Vector-vector output is a scalar.
Zero-length contractions write additive zeros without constructing a zero-size
matrix. Empty output shapes are valid after all compatibility checks.

Gemm requires rank-two A/B, applies the explicit transpose flags, and computes
`Alpha * op(A) * op(B) + Beta * bias`. Nil bias means no bias term; a nonnil
bias may be a scalar, vector, or matrix that passes `ValidateBroadcastTo` for
the `[M, N]` output. Validate a supplied bias even when Beta is zero. GemmShape
plans A/B only; callers separately validate bias broadcasting. The config has
no schema defaults: its zero value means Alpha=Beta=0 with no transposition.
The runtime supplies any operator-default values explicitly.

For compatible nonempty float32 rank-two blocks, use copying adapters around
`matrix.FromSlice`, `matrix.New`, `MatMulInto`, and the existing left/right
transpose variants. For both-transposed operands, a local materialized transpose
can feed a supported variant. Batched MatMul reuses this block strategy. Copy
results back only after adapter success. No public matrix bridge is required.
Contraction accumulation initially follows the existing float32 matrix path;
SIMD build choices may change rounding. Use operation-specific relative/absolute
float32 tolerances rather than bitwise product equality, and document the
actual tolerance and fixtures with NN-05. Empty contractions remain exact zero.

## Activations (NN-06)

```go
func ReLUInto(input, destination *Tensor) (err error)
func SigmoidInto(input, destination *Tensor) (err error)
func TanhInto(input, destination *Tensor) (err error)
func SoftmaxInto(input, destination *Tensor, axis int64) (err error)
```

Activations preserve float32 shape, including scalars for the three elementwise
functions. Reuse the existing elementwise implementations through copying
matrix adapters for nonempty values (a `[1, elementCount]` matrix is sufficient).
Do not instantiate layers or call backward methods. Empty elementwise values
are valid no-ops after validation. Existing ReLU maps NaN and negative zero to
positive zero; preserve and test that behavior rather than claiming stronger
ONNX special-value agreement. Sigmoid uses the existing stable sign split;
Tanh uses the existing float32 helper. Preserve existing activation/backward APIs.

Softmax normalizes exactly one axis, including a nonfinal axis; rank zero has
no valid axis. Compute by logical outer/axis/inner indices with stable maximum
subtraction and local scratch, reusing row softmax only when it has identical
semantics. An empty output, including an axis of length zero, succeeds without
writes after axis validation. Finite slices normalize stably. A slice containing
NaN, positive infinity, or only negative infinities may yield NaNs under the
straight maximum-subtraction formula; NN-06 must record pinned reference cases
and its exact behavior. Do not invent a uniform distribution or claim portable
special-value conformance without those fixtures.

## Type-preserving transforms and indexing (NN-07, NN-08)

```go
func ReshapeShape(input, target []int64) (shape []int64, err error)
func FlattenShape(input []int64, axis int64) (shape []int64, err error)
func TransposeShape(input, permutation []int64) (shape []int64, err error)
func SqueezeShape(input, axes []int64) (shape []int64, err error)
func UnsqueezeShape(input, axes []int64) (shape []int64, err error)
func ReshapeInto(input, destination *Tensor) (err error)
func FlattenInto(input, destination *Tensor, axis int64) (err error)
func TransposeInto(input, destination *Tensor, permutation []int64) (err error)
func SqueezeInto(input, destination *Tensor, axes []int64) (err error)
func UnsqueezeInto(input, destination *Tensor, axes []int64) (err error)

func ConcatShape(inputs [][]int64, axis int64) (shape []int64, err error)
func GatherShape(data, indices []int64, axis int64) (shape []int64, err error)
func ConcatInto(inputs []*Tensor, destination *Tensor, axis int64) (err error)
func GatherInto(data, indices, destination *Tensor, axis int64) (err error)
```

All transforms preserve float32 or int64 storage and copy into disjoint owned
destinations, even when only shape metadata changes. ReshapeInto takes its
concrete target from the destination shape; ReshapeShape checks equal element
counts. Negative/inferred dimensions and ONNX's special reshape-zero policy do
not enter this API. Flatten produces `[product(prefix), product(suffix)]` at
the split axis, using an empty product of one at both endpoints. Products are
checked even when another part is empty.

Transpose requires an explicit full permutation, normalized against input rank,
with no duplicates. Nil/empty permutation is valid only for rank zero; runtime
default reversal is resolved by the caller. Squeeze removes exactly the listed
unit axes; omitted-axis ONNX policy is resolved outside this package. Unsqueeze
inserts unit axes normalized against final output rank. Empty axis lists are
identity copies. Scalar transitions and zero dimensions retain their exact
element counts and typed row-major order.

Concat requires a nonempty list, the same type/rank for every input, equal
non-axis dimensions, and a checked sum along its normalized axis; scalars have
no concat axis. Gather requires int64 indices and float32/int64 data of positive
rank. Its output is `data[:axis] + indices.Shape() + data[axis+1:]`. Normalize
negative index values against the selected dimension and validate every index
before writes, even when another output dimension makes the result empty.
Scalar, multidimensional, repeated, and empty indices are supported. Gathering
from an axis of size zero is valid only when the indices tensor is empty.

## Convolution and pooling (NN-09, NN-10)

```go
type Conv2DConfig struct {
	Strides   [2]int64
	Dilations [2]int64
	Pads      [4]int64
	Groups    int64
}

func Conv2DShape(input, weights []int64, config Conv2DConfig) (shape []int64, err error)
func Conv2DInto(input, weights, bias, destination *Tensor, config Conv2DConfig) (err error)

type Pool2DConfig struct {
	Kernel           [2]int64
	Strides          [2]int64
	Dilations        [2]int64
	Pads             [4]int64
	CeilMode         bool
	CountIncludePad  bool
}

func Pool2DShape(input []int64, config Pool2DConfig) (shape []int64, err error)
func GlobalAveragePoolShape(input []int64) (shape []int64, err error)
func MaxPool2DInto(input, destination *Tensor, config Pool2DConfig) (err error)
func AveragePool2DInto(input, destination *Tensor, config Pool2DConfig) (err error)
func GlobalAveragePoolInto(input, destination *Tensor) (err error)
```

These APIs use float32 NCHW rank-four values. Strides, dilations, kernel extents,
and groups are positive; pads are nonnegative in `[top, left, bottom, right]`
order. Configs are explicit mathematical values, with no implicit ones or
`auto_pad`. Their zero values are invalid. NN-09/NN-10 share checked spatial
geometry helpers with their planners, including effective dilated kernel size,
padded extent, floor/ceil output size, and bounds for sampled coordinates.
Clamp a mathematically nonpositive output extent to zero; reject arithmetic
overflow before using it. NN-09/NN-10 document any additional unsupported edge
cases explicitly and provide reference tests for accepted geometry.

Convolution is cross-correlation with weights `[M, C/Groups, kH, kW]`, optional
`[M]` bias, `C % Groups == 0`, and `M % Groups == 0`. Channels, M, and kernel
extents are positive; zero batch/spatial extents may produce empty outputs or
padded zero contributions with bias. Dynamic weights work identically to
constants. Reuse the existing Conv2D loop structure where appropriate, accounting
for the different matrix layout `[C*kH*kW, M]`; never reinterpret ONNX-layout
weight bytes as that matrix. Avoid optimizer parameters, gradient buffers,
argmax history, or backward input caches. Packing is optional future work.

Max pooling supports dilation and values only, with padded samples excluded
from maxima (negative infinity is the initial identity). There is no indices
destination or storage-order argument: runtimes must reject requests for indices
or nondefault storage order. `CountIncludePad` is ignored for max pooling.
Average pooling requires Dilations=`[1, 1]`; divisor selection includes declared
padding only when CountIncludePad is true. Samples beyond declared padding in
a trailing ceil window do not enlarge that divisor. NN-10 must pin and test
the behavior of an otherwise accepted window with zero contributing samples.
Ceil mode removes trailing windows that begin at or beyond input extent plus
begin pad. Global averaging produces `[N, C, 1, 1]`; a zero spatial reduction
requires an explicit documented empty-reduction rule in NN-10.

## Inference normalization (NN-11)

```go
func BatchNormalizationInto(
	input, scale, bias, mean, variance, destination *Tensor,
	epsilon float32,
) (err error)
```

Require float32 input of rank at least two, with `[C]` parameter tensors matching
the channel axis. Apply `scale * (x - mean) / sqrt(variance + epsilon) + bias`
per channel over `[N, C, ...]`, including rank-two feature inputs and rank-four
NCHW input. Epsilon must be finite and nonnegative. Negative variance plus
epsilon follows floating sqrt domain behavior rather than mutating statistics
or silently clamping. Empty inputs are valid after parameter validation.
There is no training flag, momentum, running-statistics update, optimizer state,
or backward cache. The runtime independently enforces its selected schema ranks,
parameter requirements, and permitted output count. Test the overlapping subset
against both existing normalization layers in evaluation mode.

## Reductions and additional math (NN-12)

```go
func ReductionShape(input, axes []int64, keepDimensions bool) (shape []int64, err error)
func ReduceSumInto(input, destination *Tensor, axes []int64, keepDimensions bool) (err error)
func ReduceMeanInto(input, destination *Tensor, axes []int64, keepDimensions bool) (err error)
func ReduceMaxInto(input, destination *Tensor, axes []int64, keepDimensions bool) (err error)
func ExpInto(input, destination *Tensor) (err error)
func LogInto(input, destination *Tensor) (err error)
func SqrtInto(input, destination *Tensor) (err error)
func PowInto(left, right, destination *Tensor) (err error)
```

All require float32. Reduction axes are explicit, normalized, and distinct;
keepDimensions replaces reduced axes with ones, otherwise removes them.
Reducing every axis of a nonscalar yields a scalar; an empty axis list is an
identity copy, including a scalar input. Runtime schema defaults and
attribute-versus-input axis choices are outside this API. `PowInto` uses
multidirectional broadcasting, and Exp/Log/Sqrt preserve input shape.

Use the ordinary floating math contract, including negative sqrt/log yielding
NaN and log(0) yielding negative infinity; document signed-zero and nonfinite
fixtures. Reduction identities and zero-count mean/max must be selected against
the pinned reference in NN-12 and explicitly documented there. A permitted
empty output must still pass all argument validation. Unsupported empty
reduction cases return an error before destination writes. Do not imply that
these primitives alone provide transformer support or approximate GELU.

## Proposed usage

These examples show the exact boundary for the tasks that will implement it.
They are illustrative until the corresponding storage/product APIs land.

```go
// Scalar uses a rank-zero shape and exactly one float32 value.
scalar, err := tensor.FromFloat32(nil, []float32{3.5})
if err != nil {
	return err
}

// Empty contains zero values with a concrete, preserved rank-two shape.
empty, err := tensor.New(tensor.Float32, []int64{0, 3})
if err != nil {
	return err
}

// Shape values retain integer precision; this is an int64 tensor of shape [3].
shapeVector, err := tensor.FromInt64([]int64{3}, []int64{2, 0, 16777217})
if err != nil {
	return err
}
```

```go
func multiply() (values []float32, err error) {
	var (
		left, right, output *tensor.Tensor
		shape               []int64
	)
	if left, err = tensor.FromFloat32([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}); err != nil {
		return nil, err
	}
	if right, err = tensor.FromFloat32([]int64{3, 2}, []float32{7, 8, 9, 10, 11, 12}); err != nil {
		return nil, err
	}
	if shape, err = tensor.MatMulShape(left.Shape(), right.Shape()); err != nil {
		return nil, err
	}
	if output, err = tensor.New(tensor.Float32, shape); err != nil {
		return nil, err
	}
	if err = tensor.MatMulInto(left, right, output); err != nil {
		return nil, err
	}
	values, err = output.Float32Values() // [58, 64, 139, 154], shape [2, 2].
	return values, err
}
```

## Implementation and runtime handoff

Implement NN-02 through NN-12 in backlog order using these signatures and
update this document when a reviewed implementation requires a narrow contract
clarification. NN-02 provides checked allocation/ownership; NN-03 provides the
shared shape/index machinery; later tasks add only their own planners/kernels.
Every task supplies explicit scalar/empty/error/alias tests where applicable,
reference numerical fixtures, and a tracked handoff with exact shipped APIs.

For RT-01, the required additions are typed storage, concrete shape planners,
and disjoint-destination forward kernels. Runtime metadata must keep unknown
and symbolic dimensions separate from a concrete zero, and runtime budgets must
include owned copies, numeric destinations, bridge matrices, and call-local
scratch. No public API provides a cancellation check inside a kernel yet.
RT-01 must pin reference tools/schemas before its own conformance claims;
NN-04 onward must record the actual reference version and generated fixture
provenance, or explicitly identify hand-calculated fixtures and remaining gaps.

Reference semantics to audit include versioned
[MatMul-13](https://onnx.ai/onnx/operators/onnx__MatMul.html#matmul-13),
[Gemm-13](https://onnx.ai/onnx/operators/onnx__Gemm.html#gemm-13), and
[Softmax-13](https://onnx.ai/onnx/operators/onnx__Softmax.html#softmax-13).
The current documentation site also displays newer operator versions; use the
selected version section and pin fixture sources rather than treating the
current site version as the runtime profile.

Zero-copy matrix interoperability, borrowed views, public reusable workspaces,
constant-weight packing, and host/device coherence changes are optional later
optimization. NN-13 requires a working RT-12 runtime and before/after MLP and
shape-heavy graph measurements. That dependency is currently unavailable; do
not invent benchmarks or expose mutable initializer/device storage to claim
completion. Keep the copying bridge until evidence supports a narrower change.
