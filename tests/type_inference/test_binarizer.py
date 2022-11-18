import numpy as np

import spox.opset.ai.onnx.ml.v3 as op_ml
from spox._graph import arguments
from spox._type_system import Tensor


def test_binarizer_inference():
    (x,) = arguments(x=Tensor(np.float64, (None, 5, "N")))
    y = op_ml.binarizer(x)
    assert y.type == Tensor(np.float64, (None, 5, "N"))
