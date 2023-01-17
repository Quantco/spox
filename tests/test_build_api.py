import numpy as np
import onnxruntime

from spox import Tensor, argument, build
from spox.opset.ai.onnx import v17 as op


def test_simple_build():
    x, y = argument(Tensor(float, ())), argument(Tensor(float, ()))
    z = op.add(x, op.mul(y, op.cast(op.const(2.0), to=float)))
    model = build({"x": x, "y": y}, {"z": z})
    assert onnxruntime.InferenceSession(model.SerializeToString()).run(
        None, {"x": np.array(7.0), "y": np.array(5.0)}
    ) == np.array(17.0)
