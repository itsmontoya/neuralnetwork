# Tensor reference fixtures

`arithmetic.json` contains 40 original, deterministic float32 Add/Sub/Mul/Div
cases evaluated by ONNX 1.16.2's reference evaluator with IR 8 and standard-domain
opset 13. These fixtures pin this library test source; RT-01 still owns the
runtime's eventual compatibility profile. The generator is original project
code and does not copy upstream test data.

Generation used Python 3.9.6, `onnx==1.16.2`, `numpy==1.26.4`, and
`protobuf==4.25.3` in a temporary virtual environment:

```sh
/usr/bin/python3 -m venv /private/tmp/neuralnetwork-onnx-reference
/private/tmp/neuralnetwork-onnx-reference/bin/python -m pip install --only-binary=:all: onnx==1.16.2 numpy==1.26.4 protobuf==4.25.3
/private/tmp/neuralnetwork-onnx-reference/bin/python tensor/testdata/generate_arithmetic.py
```

Normal Go tests read the checked-in JSON offline. Shapes are compared exactly.
Every finite result, infinity, and signed zero is compared by float32 bits;
NaNs are compared by classification, without promising payload or sign.
The literal special-value pairs exercise both the matching-shape matrix adapter
(32 values, including SIMD-sized paths) and broadcasting.

Pinned upstream sources are the ONNX v1.16.2
[Add](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_add.py),
[Sub](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_sub.py),
[Mul](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_mul.py), and
[Div](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_div.py)
reference implementations. They select NumPy's corresponding operations;
float32 division uses `numpy.divide` and does not reject zero denominators.
The ONNX project is licensed under
[Apache-2.0](https://github.com/onnx/onnx/blob/v1.16.2/LICENSE).

The fixtures cover the selected arithmetic cases only; they do not establish
general ONNX conformance, integer arithmetic, or later opset compatibility.

`products.json` adds 49 original NN-05 MatMul/Gemm cases generated with the same
environment:

```sh
/private/tmp/neuralnetwork-onnx-reference/bin/python tensor/testdata/generate_products.py
```

The generator uses IR 8/opset 13 graphs and alternates runtime-fed and initializer
weights. Go tests use the same owned tensor API for both; weights can change
between calls. Cases cover vector promotion, unequal batch ranks, broadcast
prefixes, empty products, all Gemm transpose flags, scalar/vector/row/column/full
biases, omitted bias, and nonfinite scaling behavior. SHA-256 of this fixture is
`ed354ee8723c3a5519248575bbe55b848abb7b58f6c1164a1279ccb60f731322`.

Product values use `abs(got-want) <= 1e-5 + 1e-5*abs(want)` for finite values,
NaN classification, and exact infinity signs. Existing matrix kernels accumulate
in float32, so their rounding may differ from NumPy's product implementation.
Empty MatMul contractions separately require exact positive zero. Nonempty
product signed zeros and NaN payloads are not a cross-backend guarantee.

Pinned upstream product implementations are
[MatMul](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_matmul.py)
and [Gemm](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_gemm.py).
Gemm's reference skips a supplied bias term when Beta equals zero, including
negative zero; the library follows this result behavior while still validating
the bias. Alpha scales the computed product even when Alpha is zero, and a nil
bias ignores Beta. Fixtures explicitly cover nonfinite values in these cases.

Fixture SHA-256:
`b01790cce2772a2037a9c14d5d93778552dc3df0292fbef973eb5fc580519833`.

`activations.json` adds 38 original NN-06 ReLU/Sigmoid/Tanh/Softmax cases:

```sh
/private/tmp/neuralnetwork-onnx-reference/bin/python tensor/testdata/generate_activations.py
```

The same pinned environment evaluates IR 8/opset 13 graphs. Cases cover scalar
and rank-one/two/three elementwise inputs, first/middle/final and negative
Softmax axes, large logits, singleton axes, empty shapes, and separate
noncontiguous slices containing NaNs or infinities. The fixture SHA-256 is
`32289003d230f1ef4b845ecb029b69d4f8c1ce07c2a4100a7d3c5b879256ccac`.

Finite values use `abs(got-want) <= 1e-6 + 1e-6*abs(want)`. Signed zeros and
infinity signs are exact; NaNs compare by classification. Existing stable
float32 activation formulas may round differently from the NumPy reference.
These are fixture tolerances, not an error bound for every possible input.

The tests explicitly assert two reference differences. Existing library ReLU
maps NaN to positive zero, whereas the pinned NumPy maximum returns NaN. The
fixture retains that NaN reference value and the test asserts the library's
compatibility result. Softmax over an empty axis returns an empty tensor in
the library; the pinned evaluator raises a zero-size maximum-reduction error.
Two fixtures record this rejection in `reference_error` and require a
successful empty library result. No case is skipped or counted as agreement
with that rejected reference behavior. Empty values with a nonempty reduction
axis are also tested against successful reference results.

The pinned sources are
[ReLU](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_relu.py),
[Sigmoid](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_sigmoid.py),
[Tanh](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_tanh.py), and
[Softmax](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_softmax.py).
Softmax subtracts the slice maximum and normalizes exponentials. A slice with
NaN, positive infinity, or only negative infinities produces NaNs throughout;
finite slices with some negative infinities assign those elements zero.
The tests verify that special values do not affect other logical slices.

`transforms.json` adds 54 original NN-07 reshape/flatten/transpose/squeeze/
unsqueeze cases for float32 and int64 values:

```sh
/private/tmp/neuralnetwork-onnx-reference/bin/python tensor/testdata/generate_transforms.py
```

The same pinned environment generates exact shapes and values. Integer JSON
fields decode directly into Go int64 values, including both extrema and values
above `2^24`. Float32 results compare all bits exactly, including NaN payloads,
infinities, and signed zero, because these transforms perform no floating math.
The fixture SHA-256 is
`bd22d4f2c3b62711815aa9f475bc7b3ac408df59f2c1ed4896e86aaad0ff1284`.

Fifty cases also agree with ONNX 1.16.2 ReferenceEvaluator IR 8/opset 13 graphs.
Permutations are normalized before passing to the reference; scalar transpose
has the equivalent omitted permutation. The four remaining cases explicitly
exercise the library's resolved mathematical boundary with NumPy 1.26.4:

- Two reshapes move a literal zero to another axis. Their concrete targets use
  `numpy.reshape`; the runtime must already have resolved ONNX-13's special
  zero-copy-dimension and inferred-dimension rules.
- Two empty-prefix Flatten cases use the two checked concrete products and
  `numpy.reshape`. The pinned ONNX reference's inferred `-1` reshape rejects
  them; `reference_error` retains that rejection. Tests assert the documented
  library empty shape instead of claiming ONNX evaluator agreement.

The pinned ONNX sources delegate to NumPy's reshape, transpose, squeeze, and
expand-dims implementations:
[Reshape](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_reshape.py),
[Flatten](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_flatten.py),
[Transpose](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_transpose.py),
[Squeeze](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_squeeze.py), and
[Unsqueeze](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_unsqueeze.py).
Empty axes are explicit identity transforms in the library; the runtime resolves
omitted-axis schema defaults separately. No transform returns borrowed storage.

`indexing.json` adds 48 original NN-08 Concat/Gather cases for both storage types:

```sh
/private/tmp/neuralnetwork-onnx-reference/bin/python tensor/testdata/generate_indexing.py
```

The same pinned environment generates all expected values. Shapes, int64 values,
and float32 bits compare exactly, including large integers, both integer extrema,
NaN payloads, infinities, and signed zero. The fixture SHA-256 is
`2a99fa5b3ae99842c9b6f29c6b6e63f37aab0371f739a0dcbda99d734440f282`.

Cases cover feature-map/shape-vector concatenation, negative axes, empty inputs,
scalar/multidimensional/repeated/negative/empty Gather indices, and empty data
prefixes/suffixes. Forty-two cases agree with ONNX 1.16.2 ReferenceEvaluator
using IR 8/opset 13 and an independent NumPy result. Six empty-indices Gather
cases use NumPy 1.26.4 `take` for the specified output shape. The pinned Gather
implementation returns `[0]` for every empty indices tensor, losing the data
prefix, multidimensional indices shape, and suffix. Each fixture records that
actual result in `reference_shape` and explains the difference; tests assert
the contracted shape instead of counting evaluator agreement or skipping it.

The versioned [Gather-13 definition](https://onnx.ai/onnx/operators/onnx__Gather.html#gather-13)
specifies how the data and indices ranks form the output. Pinned primary sources
are [Gather](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_gather.py)
and [Concat](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_concat.py).
They otherwise delegate numerical selection/copying to NumPy. Library validation
rejects unsupported types, scalar data/concat axes, malformed axes/shapes, and
every out-of-range index before writing, including when the result is empty.

`convolution.json` adds 46 original NN-09 float32 NCHW Conv cases:

```sh
/private/tmp/neuralnetwork-onnx-reference/bin/python tensor/testdata/generate_convolution.py
```

The same pinned environment generates expected values with an independent
vectorized NumPy padded-window/einsum contraction. Forty cases also agree
with ONNX 1.16.2 ReferenceEvaluator IR 8/opset 13 (Conv schema 11). Cases cover
ordinary/grouped/depthwise convolution and channel multipliers, asymmetric
padding, independent strides/dilations, omitted bias, multiple batches,
initializer/runtime weights, fractional accumulation, boundary kernels, and
empty spatial inputs with bias-only padded results. Numerical comparisons use
`atol=rtol=1e-5`; shapes are exact. Fixture SHA-256:
`039b09171de2da1ad0e66b8b80b2a3b8a71f8eb47f6085fece3d16fe0b5973f2`.

Six empty/clamped-output cases retain the evaluator's actual exception in
`reference_error`: empty batches, empty grouped batches, and a kernel producing
negative spatial geometry, each with/without bias. Their expected values use
the contracted nonnegative concrete output shape and NumPy, rather than
counting rejected evaluator calls as agreement. No test is skipped.

Pinned primary sources are the
[Conv reference](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_conv.py)
and [schema definitions](https://github.com/onnx/onnx/blob/v1.16.2/onnx/defs/nn/old.cc).
The versioned [Conv-11 definition](https://onnx.ai/onnx/operators/onnx__Conv.html#conv-11)
specifies the NCHW/filter layout and mathematical attributes. Library padding
skips absent samples; these finite fixtures do not independently cover NaN or
infinity interactions with padding. The runtime owns automatic padding and
schema-specific restrictions.

`pooling.json` adds 65 original NN-10 MaxPool/AveragePool/GlobalAveragePool
cases. The independent NumPy oracle uses broadcast coordinate grids, advanced
indexing, and reductions, rather than the Go kernel's loops. It records concrete
output shapes, float32 bits, and exact per-window average divisors.

```sh
/private/tmp/neuralnetwork-onnx-reference/bin/python -m pip install --only-binary=:all: onnxruntime==1.19.2
/private/tmp/neuralnetwork-onnx-reference/bin/python tensor/testdata/generate_pooling.py
```

The existing Python 3.9.6 / ONNX 1.16.2 / NumPy 1.26.4 / protobuf 4.25.3
environment also runs ONNX Runtime 1.19.2 with CPUExecutionProvider, one
intra/inter-op thread, and graph optimizations disabled. These tools are only
for fixture regeneration; ordinary Go tests read JSON offline. Neither ONNX
Runtime nor a new Go dependency is needed to build or run the library.
Numerical comparisons use `atol=rtol=1e-5`, NaN classification, and exact
infinity signs; shapes and divisor counts are exact. Finite max-zero tie signs
and NaN payloads are not cross-backend promises. Fixture SHA-256:
`f413621f5e94f9073450584fd0db265ad2a41d695f28d7b9a30e40c644fa70a6`.

NN-10 corrects an NN-01 draft assumption: CountIncludePad uses the full kernel
area even for a partial ceil window beyond declared padding. For input values
1 through 9 shaped `[1,1,3,3]`, kernel `[2,2]`, stride `[2,2]`, no padding,
and ceil mode, both executed references give average outputs
`[3, 2.25, 3.75, 2.25]` when including padding, with divisors `[4,4,4,4]`.
Excluding padding gives `[3,4.5,7.5,9]` with divisors `[4,2,2,1]`.
For `[1,1,2,2]` containing 1 through 4 with a `[3,3]` kernel and `[2,2]`
stride, ceil mode gives `10/9` with padding included and `10/4` without.
These literal outcomes and expected divisors are asserted in Go tests.

Forty-eight cases agree with ONNX ReferenceEvaluator, IR 8/opset 13
(MaxPool-12, AveragePool-11, GlobalAveragePool-1). Seventeen retain its exact
errors or differing shapes/value bits and use the contracted NumPy result:
five ceil-shape differences, nine empty/all-padding/oversized-window errors,
and three NaN/no-valid-sample max differences. No case is skipped.

The same fixtures retain actual ONNX Runtime outputs or errors: 40 agreements,
four differences, and 21 rejections. The four differences are precise:

- Input `[1,1,3,3]`, kernel/stride `[3,3]`, pads `[0,0,2,2]`, ceil mode:
  the task requires output `[1,1,1,1]`; ORT returns `[1,1,2,2]` for max and
  both average divisor modes, retaining windows that start on end padding.
- Input `[1,1,1,1]`, kernel/dilation `[2,2]`, pads `[1,1,1,1]`:
  dilation samples no actual input. The library's max identity is negative
  infinity; ORT returns the lowest finite float32.

ORT rejects the other 21 documented cases because pads must be smaller than
kernel extents, only the batch dimension may be zero, or a raw output extent
is negative. The library's mathematical backend permits those shapes and
clamps nonpositive output extents; the runtime adapter still owns schema/profile
restrictions. All-padding averages excluding padding and global averages over
zero spatial extents produce NaN. Max/average propagate actual input NaNs;
the pinned ONNX common pooling path instead filters them in some cases.

Pinned primary sources are [common pooling](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_pool_common.py),
[max pooling](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_max_pool.py),
[global average pooling](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_global_average_pool.py),
and ONNX Runtime's [pool dispatch](https://github.com/microsoft/onnxruntime/blob/v1.19.2/onnxruntime/core/providers/cpu/nn/pool.cc)
and [MLAS pooling](https://github.com/microsoft/onnxruntime/blob/v1.19.2/onnxruntime/core/mlas/lib/pooling.cpp).
The versioned [MaxPool-12](https://onnx.ai/onnx/operators/onnx__MaxPool.html#maxpool-12)
and [AveragePool-11](https://onnx.ai/onnx/operators/onnx__AveragePool.html#averagepool-11)
definitions supply the mathematical attributes. ONNX is Apache-2.0 licensed;
[ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/v1.19.2/LICENSE) is MIT licensed.
The original generator does not copy upstream implementation or fixture data.

`normalization.json` adds 31 original NN-11 stored-statistics batch-normalization
fixtures. Independent NumPy broadcasting computes
`scale * (x - mean) / sqrt(variance + epsilon) + bias` without reproducing the
Go channel-index loop. Inputs, parameters, epsilon, and expected results retain
float32 bit patterns. Models use IR 8/opset 13, BatchNormalization-9, and one
output. The same pinned Python/ONNX/NumPy/protobuf/ORT environment records actual
outputs from both reference implementations:

```sh
/private/tmp/neuralnetwork-onnx-reference/bin/python tensor/testdata/generate_normalization.py
```

Ordinary Go tests require no Python or runtime library. Finite comparisons use
`atol=rtol=1e-5`; NaNs compare by classification and infinity signs exactly.
Fixture SHA-256:
`4788058268769cb709bfc099aa137e5417d9dd26c5391c19d4de090fa0e6b8bc`.

Thirty cases agree with executed ONNX Runtime 1.19.2 CPU results, using one
intra/inter-op thread and disabled graph optimizations. Its sole difference is
zero variance with zero epsilon: input `[-1,0,1]`, scale 1, bias/mean/variance 0
produces three NaNs in ORT, while the literal formula gives `[-Inf,NaN,+Inf]`.
The kernel preserves the stated expression and records this numerical limit.

ONNX 1.16.2 ReferenceEvaluator agrees on eight cases and differs on 23.
Its BatchNormalization-9 wrapper updates means/variances using momentum even
when only one inference output is requested. The kernel instead uses the
supplied statistics unchanged, as required by the inference contract. For
`[3,5]` shaped `[2,1]`, scale 2, bias 0.5, mean 1, variance 4, and epsilon 0,
NumPy and ORT give `[2.5,4.5]`; the evaluator with default momentum 0.9 gives
approximately `[2.2675757,4.3470764]`. Exact actual shapes/bits, result
classification, and explicit/defaulted momentum metadata are retained; no
disagreement is silently counted as agreement or skipped.

Cases include ranks two through six, multiple batches/channels, learned
statistics, nondefault/zero/negative-zero epsilon, empty dimensions, negative
variance domains, and nonfinite inputs/parameters. The library does not expose
momentum, training mode, or running-statistics updates. Runtime schema/rank/output
restrictions remain adapter responsibilities. Pinned primary sources are the
[normalization reference](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_batch_normalization.py),
[schema definitions](https://github.com/onnx/onnx/blob/v1.16.2/onnx/defs/nn/old.cc),
and [ORT CPU normalization](https://github.com/microsoft/onnxruntime/blob/v1.19.2/onnxruntime/core/providers/cpu/nn/batch_norm.cc).
The versioned [BatchNormalization-9 definition](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html#batchnormalization-9)
supplies the inference boundary. ONNX is Apache-2.0 licensed; ONNX Runtime is
MIT licensed. The original generator does not copy upstream implementation or
fixture data. These fixtures make no general backend-equivalence claim.

`reductions_math.json` adds 99 original NN-12 ReduceSum/ReduceMean/ReduceMax,
Exp/Log/Sqrt, and broadcast Pow cases:

```sh
/private/tmp/neuralnetwork-onnx-reference/bin/python tensor/testdata/generate_reductions_math.py
```

Generation uses the same pinned Python 3.9.6, ONNX 1.16.2, NumPy 1.26.4, and
protobuf 4.25.3. NumPy reductions and vectorized unary/power expressions form
independent expectations rather than reproducing Go's coordinate loops. All
99 cases agree with executed ONNX ReferenceEvaluator models; exact actual
shapes/bits and selected opsets are retained. SHA-256:
`165660321fe5c25f792562b57b547165b1ff9474352826e102e3295562d0a0fc`.

Reduction models use IR 8/opset 18 so every operator can express explicit empty
axes as identity with `noop_with_empty_axes=1`. Identity models omit the
optional axes input; all nonempty axis lists are supplied as int64 tensors.
The library always receives resolved mathematical axes and does not select
omitted-axis defaults. These fixtures do not expand the initial runtime
profile or implement its versioned reduction adapters. Exp/Log/Sqrt/Pow models
use IR 8/opset 13. Ordinary Go tests need only the checked-in JSON.

Cases cover reordered/negative/multiple/all/no axes, keepdims on/off, scalar
identity, zero-count reductions, empty outputs, noncontiguous logical slices,
NaNs/infinities, math domains, signed zeros, and multidirectional/scalar Pow.
Zero-count identities are sum positive zero, mean NaN, and max negative
infinity. The pinned max reference explicitly supplies this empty identity;
NumPy expectations use `initial=-inf`. Literal Go tests independently assert
these values and multi-axis results. Empty-axis identity copies preserve all
bits. Nonempty max zero-tie signs and NaN payloads are not cross-backend
promises. Finite comparisons use `atol=rtol=1e-5`, infinity signs and unary/Pow
zero signs compare exactly, and NaNs compare by classification. Float32
row-major sum/mean accumulation may round differently from NumPy's reduction
algorithm; the selected tolerances are fixture bounds, not universal guarantees.

Pinned primary implementations are
[ReduceSum](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_reduce_sum.py),
[ReduceMean](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_reduce_mean.py),
[ReduceMax](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_reduce_max.py),
[Exp](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_exp.py),
[Log](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_log.py),
[Sqrt](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_sqrt.py), and
[Pow](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_pow.py).
ONNX is Apache-2.0 licensed. The original generator copies no upstream code or
fixture data, and these selected cases do not establish general compatibility.
