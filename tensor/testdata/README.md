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
