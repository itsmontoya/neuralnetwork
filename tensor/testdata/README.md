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
