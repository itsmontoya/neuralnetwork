"""Regenerate original NN-12 fixtures with pinned mathematical references."""

import json
import warnings
from pathlib import Path

import google.protobuf
import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def bits(value):
    return np.asarray(value, dtype=np.float32).reshape(-1).view(np.uint32).tolist()


def evaluate(name, operation, shape, values, axes=(), keep=False,
             right_shape=None, right_values=None):
    x = np.asarray(values, dtype=np.float32).reshape(shape)
    feeds = {"input": x}
    attributes = {}
    inputs = ["input"]
    opset = 13
    if operation.startswith("Reduce"):
        # Opset 18 represents the explicit empty-axis identity for every
        # reduction. The library has no ONNX omitted-axis/default policy.
        opset = 18
        if axes:
            feeds["axes"] = np.asarray(axes, dtype=np.int64)
            inputs.append("axes")
        attributes = {"keepdims": int(keep), "noop_with_empty_axes": 1}
        if not axes:
            expected = x.copy()
        elif operation == "ReduceSum":
            expected = np.sum(x, axis=tuple(axes), keepdims=keep, dtype=np.float32)
        elif operation == "ReduceMean":
            expected = np.mean(x, axis=tuple(axes), keepdims=keep, dtype=np.float32)
        else:
            # Initial -Inf expresses the pinned ONNX float32 empty identity.
            expected = np.max(x, axis=tuple(axes), keepdims=keep, initial=-np.inf)
    elif operation == "Pow":
        feeds["right"] = np.asarray(right_values, dtype=np.float32).reshape(right_shape)
        inputs.append("right")
        expected = np.power(x, feeds["right"])
    else:
        expected = {"Exp": np.exp, "Log": np.log, "Sqrt": np.sqrt}[operation](x)
    expected = np.asarray(expected, dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node(operation, inputs, ["output"], **attributes)], name,
        [helper.make_tensor_value_info(key, TensorProto.INT64 if key == "axes" else TensorProto.FLOAT,
                                       list(value.shape)) for key, value in feeds.items()],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, ir_version=8,
                              opset_imports=[helper.make_opsetid("", opset)])
    result, error, actual_shape, actual_bits = "agree", "", [], []
    try:
        actual = ReferenceEvaluator(model).run(None, feeds)[0]
        actual_shape, actual_bits = list(actual.shape), bits(actual)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-5, atol=1e-5, equal_nan=True):
            result = "different"
    except (ValueError, RuntimeError, TypeError, IndexError, ZeroDivisionError) as failure:
        result, error = "error", str(failure)
    return {
        "name": name, "operation": operation, "opset": opset,
        "input_shape": list(shape), "input_bits": bits(x),
        "axes": list(axes), "keep_dimensions": keep,
        "axes_input_present": "axes" in feeds,
        "right_shape": [] if right_shape is None else list(right_shape),
        "right_bits": [] if right_shape is None else bits(feeds["right"]),
        "output_shape": list(expected.shape), "output_bits": bits(expected),
        "reference_result": result, "reference_error": error,
        "reference_shape": actual_shape, "reference_bits": actual_bits,
    }


assert onnx.__version__ == "1.16.2", onnx.__version__
assert np.__version__ == "1.26.4", np.__version__
assert google.protobuf.__version__ == "4.25.3", google.protobuf.__version__

fixtures = []
with warnings.catch_warnings(), np.errstate(all="ignore"):
    warnings.simplefilter("ignore", RuntimeWarning)
    values = (np.arange(24, dtype=np.float32) - np.float32(11.5)) / np.float32(4)
    for operation in ("ReduceSum", "ReduceMean", "ReduceMax"):
        for axes in ([0], [1], [-1], [0, 2], [-1, -3], [2, 0]):
            for keep in (False, True):
                fixtures.append(evaluate(f"{operation} axes{axes} keep{keep}", operation,
                                         [2, 3, 4], values, axes, keep))
        for axes in ([0, 1, 2], []):
            for keep in (False, True):
                fixtures.append(evaluate(f"{operation} all or identity {axes} keep{keep}", operation,
                                         [2, 3, 4], values, axes, keep))
        fixtures.extend([
            evaluate(f"{operation} vector", operation, [5], [-4, -1, 0.5, 2, 8], [-1]),
            evaluate(f"{operation} scalar identity", operation, [], [-0.0]),
            evaluate(f"{operation} all empty to scalar", operation, [2, 0, 3], [], [2, 0, 1]),
            evaluate(f"{operation} nonfinite rows", operation, [2, 4],
                     [np.inf, 1, -np.inf, np.nan, -0.0, 0.0, -2, 2], [1]),
            evaluate(f"{operation} noncontiguous special slices", operation, [2, 2, 3],
                     [1, np.inf, -np.inf, 3, 2, -np.inf, 4, np.nan, 6, 8, 10, 12], [1]),
            evaluate(f"{operation} all negative infinity", operation, [2, 2], [-np.inf] * 4, [1]),
        ])
        for keep in (False, True):
            fixtures.append(evaluate(f"{operation} empty reduction keep{keep}", operation,
                                     [2, 0, 3], [], [1], keep))
            fixtures.append(evaluate(f"{operation} empty output keep{keep}", operation,
                                     [0, 2, 3], [], [2], keep))
    for operation in ("Exp", "Log", "Sqrt"):
        fixtures.extend([
            evaluate(f"{operation} scalar signed zero", operation, [], [-0.0]),
            evaluate(f"{operation} domains and nonfinite", operation, [11],
                     [-np.inf, -4, -1, -0.0, 0.0, 0.25, 1, 4, 100, np.inf, np.nan]),
            evaluate(f"{operation} higher rank", operation, [2, 2, 2],
                     np.linspace(0.1, 2.1, 8, dtype=np.float32)),
            evaluate(f"{operation} empty", operation, [2, 0, 3], []),
        ])
    pairs = [(-0.0, -3), (-0.0, 3), (0.0, -1), (0.0, 2), (-2, 0.5), (-2, 3),
             (-2, 4), (-np.inf, -3), (-np.inf, 3), (np.inf, 0), (np.nan, 0),
             (1, np.nan), (-1, np.inf), (2, np.inf), (0.5, -np.inf), (np.nan, 2)] * 2
    fixtures.extend([
        evaluate("Pow row broadcast", "Pow", [2, 3], [1, 2, 3, 4, 5, 6],
                 right_shape=[3], right_values=[2, 3, -1]),
        evaluate("Pow multidirectional", "Pow", [2, 1, 2], [-2, 0.25, 4, -1],
                 right_shape=[1, 3, 1], right_values=[0, 0.5, 3]),
        evaluate("Pow scalar base", "Pow", [], [2], right_shape=[2, 3], right_values=[-3, -2, -1, 0, 1, 3]),
        evaluate("Pow scalar exponent domains", "Pow", [2, 3], [-4, -1, -0.0, 0, 1, 4],
                 right_shape=[], right_values=[0.5]),
        evaluate("Pow matching specials", "Pow", [32], [pair[0] for pair in pairs],
                 right_shape=[32], right_values=[pair[1] for pair in pairs]),
        evaluate("Pow broadcast NaN exponent", "Pow", [6], [-1, -0.0, 0, 1, 2, np.nan],
                 right_shape=[], right_values=[np.nan]),
        evaluate("Pow empty", "Pow", [2, 0, 3], [], right_shape=[3], right_values=[2, 3, 4]),
        evaluate("Pow empty zero with one", "Pow", [0, 1], [], right_shape=[1, 3], right_values=[2, 3, 4]),
        evaluate("Pow scalars", "Pow", [], [-2], right_shape=[], right_values=[3]),
    ])

path = Path(__file__).with_name("reductions_math.json")
path.write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(fixtures)} cases")
print({result: sum(case["reference_result"] == result for case in fixtures)
       for result in ("agree", "different", "error")})
for case in fixtures:
    if case["reference_result"] != "agree":
        print(case["name"], case["reference_result"], case["reference_error"])
