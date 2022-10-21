import numpy as np

from steelix import arguments, Tensor
import steelix.opset.ai.onnx.ml.v3 as op_ml


def test_binarizer_inference():
    (x,) = arguments(x=Tensor(np.float64, (None, 5, "N")))
    y = op_ml.binarizer(x)
    assert y.type == Tensor(np.float64, (None, 5, "N"))
