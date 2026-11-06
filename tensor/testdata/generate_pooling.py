"""Generate NN-10 sampled-window fixtures with pinned NumPy and ONNX.

Development only; ordinary Go tests read pooling.json offline.
"""

import json
import warnings
from pathlib import Path

import google.protobuf
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException


def bits(values):
    return np.asarray(values, dtype=np.float32).reshape(-1).view(np.uint32).tolist()


def output_dimension(size, kernel, stride, dilation, begin, end, ceil):
    numerator = size + begin + end - ((kernel - 1) * dilation + 1)
    result = max(0, (-((-numerator) // stride) if ceil else numerator // stride) + 1)
    if ceil:
        result = min(result, (size + begin + stride - 1) // stride)
    return result


def numpy_pool(x, operation, kernel, strides, dilations, pads, ceil, include):
    if operation == "GlobalAveragePool":
        return np.mean(x, axis=(2, 3), keepdims=True, dtype=np.float32), [x.shape[2] * x.shape[3]]
    oh, ow = [output_dimension(x.shape[i + 2], kernel[i], strides[i],
                              dilations[i], pads[i], pads[i + 2], ceil)
              for i in range(2)]
    shape = (x.shape[0], x.shape[1], oh, ow)
    if np.prod(shape) == 0:
        return np.empty(shape, dtype=np.float32), []

    # Broadcast coordinate grids and advanced indexing build independent windows.
    rows = np.arange(oh)[:, None] * strides[0] + np.arange(kernel[0]) * dilations[0] - pads[0]
    cols = np.arange(ow)[:, None] * strides[1] + np.arange(kernel[1]) * dilations[1] - pads[1]
    valid_rows, valid_cols = (rows >= 0) & (rows < x.shape[2]), (cols >= 0) & (cols < x.shape[3])
    valid = valid_rows[:, None, :, None] & valid_cols[None, :, None, :]
    if x.shape[2] == 0 or x.shape[3] == 0:
        windows = np.zeros(shape + tuple(kernel), dtype=np.float32)
    else:
        windows = x[:, :, np.clip(rows, 0, x.shape[2] - 1)[:, None, :, None],
                    np.clip(cols, 0, x.shape[3] - 1)[None, :, None, :]]
    if operation == "MaxPool":
        return np.max(np.where(valid, windows, np.float32(-np.inf)), axis=(-2, -1)), []

    totals = np.sum(np.where(valid, windows, np.float32(0)), axis=(-2, -1), dtype=np.float32)
    if include:
        counts = np.full((oh, ow), kernel[0] * kernel[1], dtype=np.int64)
    else:
        counts = np.sum(valid, axis=(-2, -1))
    return np.divide(totals, counts.astype(np.float32)), counts.reshape(-1).tolist()


def evaluate(name, operation, shape, kernel=(2, 2), strides=(1, 1),
             dilations=(1, 1), pads=(0, 0, 0, 0), ceil=False, include=False,
             values=None):
    count = int(np.prod(shape, dtype=np.int64))
    x = (-(np.arange(count, dtype=np.float32) % 13 + 1) / np.float32(7)
         if values is None else np.array(values, dtype=np.float32)).reshape(shape)
    attrs = {}
    if operation != "GlobalAveragePool":
        attrs = {"kernel_shape": list(kernel), "strides": list(strides),
                 "pads": list(pads), "ceil_mode": int(ceil)}
        if operation == "MaxPool":
            attrs["dilations"] = list(dilations)
        else:
            attrs["count_include_pad"] = int(include)
    graph = helper.make_graph(
        [helper.make_node(operation, ["input"], ["output"], **attrs)], name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(shape))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 13)])
    output, divisors = numpy_pool(x, operation, kernel, strides, dilations, pads, ceil, include)
    reference, error, difference, actual_shape, actual_bits = "onnx", "", "", [], []
    try:
        actual = ReferenceEvaluator(model).run(None, {"input": x})[0]
        actual_shape, actual_bits = list(actual.shape), bits(actual)
        if actual.shape != output.shape:
            reference = "numpy"
            difference = f"output shape {list(actual.shape)} differs from contracted shape {list(output.shape)}"
        elif not np.allclose(actual, output, rtol=1e-5, atol=1e-5, equal_nan=True):
            reference = "numpy"
            difference = "values differ from sampled-window identity/NaN rules; exact reference bits retained"
    except (ValueError, RuntimeError, IndexError, ZeroDivisionError) as exception:
        reference, error = "numpy", str(exception)

    ort_result, ort_error, ort_difference, ort_shape, ort_bits = "agree", "", "", [], []
    try:
        session = ort.InferenceSession(model.SerializeToString(), ORT_OPTIONS, providers=["CPUExecutionProvider"])
        ort_actual = session.run(None, {"input": x})[0]
        ort_shape, ort_bits = list(ort_actual.shape), bits(ort_actual)
        if ort_actual.shape != output.shape:
            ort_result = "different"
            ort_difference = f"output shape {list(ort_actual.shape)} differs from contracted shape {list(output.shape)}"
        elif not np.allclose(ort_actual, output, rtol=1e-5, atol=1e-5, equal_nan=True):
            ort_result = "different"
            ort_difference = "no-valid-sample max uses lowest finite float instead of negative infinity; exact bits retained"
    except RuntimeException as exception:
        ort_result, ort_error = "error", str(exception)
    return {
        "name": name, "operation": operation, "input_shape": list(shape),
        "input_bits": bits(x), "kernel": list(kernel), "strides": list(strides),
        "dilations": list(dilations), "pads": list(pads), "ceil_mode": ceil,
        "count_include_pad": include, "output_shape": list(output.shape),
        "output_bits": bits(output), "reference": reference,
        "expected_divisors": divisors,
        "reference_error": error, "reference_difference": difference,
        "reference_shape": actual_shape, "reference_bits": actual_bits,
        "ort_result": ort_result, "ort_error": ort_error,
        "ort_difference": ort_difference, "ort_shape": ort_shape, "ort_bits": ort_bits,
    }


assert onnx.__version__ == "1.16.2", onnx.__version__
assert np.__version__ == "1.26.4", np.__version__
assert google.protobuf.__version__ == "4.25.3", google.protobuf.__version__
assert ort.__version__ == "1.19.2", ort.__version__
ORT_OPTIONS = ort.SessionOptions()
ORT_OPTIONS.intra_op_num_threads = 1
ORT_OPTIONS.inter_op_num_threads = 1
ORT_OPTIONS.log_severity_level = 4
ORT_OPTIONS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

cases = [
    ("rectangular batches channels", [2, 2, 4, 5], {"kernel": [2, 3]}),
    ("negative padded", [1, 1, 2, 3], {"pads": [1, 1, 1, 1]}),
    ("odd floor", [1, 1, 5, 5], {"strides": [2, 2]}),
    ("odd ceil partial beyond declared pads", [1, 1, 3, 3], {"strides": [2, 2], "ceil": True, "values": list(range(1, 10))}),
    ("ceil removes starts on end pads", [1, 1, 3, 3],
     {"kernel": [3, 3], "strides": [3, 3], "pads": [0, 0, 2, 2], "ceil": True}),
    ("ceil negative numerator", [1, 1, 2, 2], {"kernel": [3, 3], "strides": [2, 2], "ceil": True, "values": [1, 2, 3, 4]}),
    ("asymmetric pads", [1, 2, 3, 4], {"kernel": [2, 3], "pads": [1, 2, 0, 1]}),
    ("unequal strides", [2, 1, 5, 6], {"kernel": [2, 3], "strides": [2, 1], "pads": [0, 1, 1, 0]}),
    ("all padding windows", [1, 1, 1, 1], {"pads": [2, 2, 2, 2]}),
    ("empty batch", [0, 2, 3, 4], {}),
    ("empty channels", [1, 0, 3, 4], {}),
    ("empty height", [1, 1, 0, 3], {"kernel": [1, 1]}),
    ("padded empty height", [1, 1, 0, 3], {"pads": [2, 0, 1, 0]}),
    ("oversized floor kernel", [1, 1, 1, 2], {"kernel": [3, 3]}),
    ("stride beyond extent", [1, 1, 2, 3], {"kernel": [1, 1], "strides": [5, 5]}),
    ("ceil clips many end-pad starts", [1, 1, 3, 3],
     {"strides": [2, 3], "pads": [0, 0, 8, 7], "ceil": True}),
    ("NaN input", [1, 1, 2, 3], {"values": [1, np.nan, 3, 4, 5, 6]}),
    ("infinite input", [1, 1, 2, 2], {"values": [-np.inf, np.inf, 2, -3]}),
]
fixtures = []
with warnings.catch_warnings(), np.errstate(all="ignore"):
    warnings.simplefilter("ignore", RuntimeWarning)
    for name, shape, attributes in cases:
        fixtures.append(evaluate(name + " max", "MaxPool", shape, **attributes))
        for include in (False, True):
            fixtures.append(evaluate(name + (" avg include" if include else " avg exclude"),
                                     "AveragePool", shape, include=include, **attributes))
    for name, shape, attributes in [
        ("dilated max", [2, 2, 5, 6], {"kernel": [2, 3], "dilations": [2, 2],
                                      "strides": [2, 1], "pads": [1, 1, 0, 2]}),
        ("dilated ceil max", [1, 1, 4, 5], {"dilations": [2, 3], "strides": [2, 2],
                                          "pads": [0, 1, 1, 0], "ceil": True}),
        ("dilation misses all input", [1, 1, 1, 1], {"dilations": [2, 2], "pads": [1, 1, 1, 1]}),
    ]:
        fixtures.append(evaluate(name, "MaxPool", shape, **attributes))
    for name, shape, values in [
        ("global batches channels", [2, 3, 3, 5], None),
        ("global singleton", [1, 2, 1, 1], None),
        ("global empty height", [1, 2, 0, 3], None),
        ("global empty spatial", [2, 1, 0, 0], None),
        ("global empty batch", [0, 2, 3, 4], None),
        ("global empty channels", [1, 0, 3, 4], None),
        ("global NaN", [1, 1, 1, 2], [np.nan, 1]),
        ("global infinities", [1, 1, 1, 2], [np.inf, -np.inf]),
    ]:
        fixtures.append(evaluate(name, "GlobalAveragePool", shape, values=values))

path = Path(__file__).with_name("pooling.json")
path.write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(fixtures)} cases; {sum(f['reference'] != 'onnx' for f in fixtures)} explicit NumPy references")
print("ORT 1.19.2:", {result: sum(f["ort_result"] == result for f in fixtures) for result in ("agree", "different", "error")})
for fixture in fixtures:
    if fixture["reference"] != "onnx":
        print(fixture["name"], fixture["reference_error"] or fixture["reference_difference"])
