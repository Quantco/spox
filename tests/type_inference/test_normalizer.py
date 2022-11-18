import numpy as np

import spox.opset.ai.onnx.ml.v3 as op_ml
from spox._graph import arguments
from spox._type_system import Tensor


def test_normalizer_inference():
    (x,) = arguments(x=Tensor(np.float64, ("N", 5)))
    y = op_ml.normalizer(x)
    assert y.type == Tensor(np.float64, ("N", 5))
