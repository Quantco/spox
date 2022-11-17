import numpy as np

import spox.opset.ai.onnx.ml.v3 as op_ml
from spox._graph import arguments
from spox._type_system import Tensor


def test_linear_regressor_inference():
    (x,) = arguments(x=Tensor(np.float64, ("N", 5)))
    y = op_ml.linear_regressor(x)
    assert y.type == Tensor(np.float32, ("N", 5))


def test_linear_regressor_inference_vector():
    (x,) = arguments(x=Tensor(np.float64, ("N",)))
    y = op_ml.linear_regressor(x)
    assert y.type == Tensor(np.float32, (1, "N"))


def test_linear_regressor_inference_scalar():
    (x,) = arguments(x=Tensor(np.float64, ()))
    y = op_ml.linear_regressor(x)
    assert y.type == Tensor(np.float32, (1, 1))
