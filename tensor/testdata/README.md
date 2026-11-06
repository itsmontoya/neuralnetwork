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

Fixture SHA-256:
`b01790cce2772a2037a9c14d5d93778552dc3df0292fbef973eb5fc580519833`.
