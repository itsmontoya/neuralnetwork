"""Generate NN-11 stored-statistics fixtures with pinned independent tools.

Development only; ordinary Go tests read normalization.json offline.
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
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument, RuntimeException


def bits(values):
    return np.asarray(values, dtype=np.float32).reshape(-1).view(np.uint32).tolist()


def evaluate(name, shape, epsilon=np.float32(0.125), values=None,
             scale=None, bias=None, mean=None, variance=None, momentum=None):
    channels = shape[1]
    count = int(np.prod(shape, dtype=np.int64))
    x = (np.linspace(-4, 7, count, dtype=np.float32)
         if values is None else np.asarray(values, dtype=np.float32)).reshape(shape)
    parameters = [np.asarray(value, dtype=np.float32) if value is not None
                  else np.resize(np.asarray(default, dtype=np.float32), channels)
                  for value, default in [
                      (scale, [2, -1.5, 0.25, 3]),
                      (bias, [0.5, -2, 1.25, -0.75]),
                      (mean, [1, -2.5, 0.75, 4]),
                      (variance, [4, 9, 0.25, 2]),
                  ]]
    feeds = dict(zip(["input", "scale", "bias", "mean", "variance"], [x] + parameters))
    attributes = {"epsilon": float(epsilon)}
    if momentum is not None:
        attributes["momentum"] = momentum
    graph = helper.make_graph(
        [helper.make_node("BatchNormalization", list(feeds), ["output"], **attributes)],
        name,
        [helper.make_tensor_value_info(key, TensorProto.FLOAT, list(value.shape))
         for key, value in feeds.items()],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 13)])

    # Independent broadcasting evaluates the stated inference expression,
    # without the Go loop's explicit channel/spatial index calculation.
    parameter_shape = (1, channels) + (1,) * (len(shape) - 2)
    s, b, m, v = [value.reshape(parameter_shape) for value in parameters]
    expected = (s * (x - m) / np.sqrt(v + np.float32(epsilon)) + b).astype(np.float32)

    reference_result, reference_error, reference_shape, reference_bits = "agree", "", [], []
    try:
        actual = ReferenceEvaluator(model).run(None, feeds)[0]
        reference_shape, reference_bits = list(actual.shape), bits(actual)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-5, atol=1e-5, equal_nan=True):
            reference_result = "different"
    except (ValueError, RuntimeError, IndexError, ZeroDivisionError) as error:
        reference_result, reference_error = "error", str(error)

    ort_result, ort_error, ort_shape, ort_bits = "agree", "", [], []
    try:
        session = ort.InferenceSession(model.SerializeToString(), ORT_OPTIONS, providers=["CPUExecutionProvider"])
        actual = session.run(None, feeds)[0]
        ort_shape, ort_bits = list(actual.shape), bits(actual)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-5, atol=1e-5, equal_nan=True):
            ort_result = "different"
    except (Fail, InvalidArgument, RuntimeException) as error:
        ort_result, ort_error = "error", str(error)

    return {
        "name": name, "input_shape": list(shape), "input_bits": bits(x),
        "scale_bits": bits(parameters[0]), "bias_bits": bits(parameters[1]),
        "mean_bits": bits(parameters[2]), "variance_bits": bits(parameters[3]),
        "epsilon_bits": bits([epsilon])[0], "output_bits": bits(expected),
        "momentum_explicit": momentum is not None,
        "momentum": 0.9 if momentum is None else momentum,
        "reference_result": reference_result, "reference_error": reference_error,
        "reference_shape": reference_shape, "reference_bits": reference_bits,
        "ort_result": ort_result, "ort_error": ort_error,
        "ort_shape": ort_shape, "ort_bits": ort_bits,
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

fixtures = []
with warnings.catch_warnings(), np.errstate(all="ignore"):
    warnings.simplefilter("ignore", RuntimeWarning)
    for shape in ([1, 1], [3, 3], [2, 3, 4], [2, 3, 2, 4]):
        for epsilon in (np.float32(0), np.float32(1e-5), np.float32(0.125)):
            fixtures.append(evaluate(f"shape {shape} epsilon {epsilon}", shape, epsilon))
    for shape in ([2, 3, 2, 1, 2], [1, 2, 1, 2, 1, 3], [2, 1, 1, 1]):
        fixtures.append(evaluate(f"higher rank or singleton {shape}", shape))
    for shape in ([0, 3], [2, 0], [0, 2, 3, 4], [1, 0, 3, 4], [2, 3, 0, 4], [1, 2, 3, 0, 1]):
        fixtures.append(evaluate(f"empty {shape}", shape))
    fixtures.extend([
        evaluate("literal learned statistics", [2, 1], np.float32(0),
                 values=[3, 5], scale=[2], bias=[0.5], mean=[1], variance=[4]),
        evaluate("zero variance and zero epsilon", [3, 1], np.float32(0),
                 values=[-1, 0, 1], scale=[1], bias=[0], mean=[0], variance=[0]),
        evaluate("negative sqrt domain", [2, 2], np.float32(0), variance=[-1, 4]),
        evaluate("epsilon offsets negative variance", [2, 2], np.float32(0.5), variance=[-0.25, 4]),
        evaluate("nonfinite input", [1, 2, 3], values=[-np.inf, 0, np.inf, np.nan, -0.0, 2]),
        evaluate("nonfinite parameters", [2, 4], scale=[1, np.inf, np.nan, 0],
                 bias=[0, -np.inf, 1, np.inf], mean=[np.inf, 2, 0, -np.inf],
                 variance=[np.inf, 4, np.nan, -np.inf]),
        evaluate("negative zero epsilon", [2, 2], np.float32(-0.0)),
    ])
    for momentum in (0.0, 0.5, 1.0):
        fixtures.append(evaluate(f"inference momentum attribute {momentum}", [2, 2], momentum=momentum))

path = Path(__file__).with_name("normalization.json")
path.write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(fixtures)} cases")
for prefix in ("reference", "ort"):
    print(prefix, {result: sum(case[prefix + "_result"] == result for case in fixtures)
                   for result in ("agree", "different", "error")})
for case in fixtures:
    if case["ort_result"] != "agree":
        print(case["name"], "ORT", case["ort_result"], case["ort_error"])
