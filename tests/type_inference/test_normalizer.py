import numpy as np

from steelix import arguments, Tensor
import steelix.opset.ai.onnx.ml.v3 as op_ml


def test_normalizer_inference():
    (x,) = arguments(x=Tensor(np.float64, ("N", 5)))
    y = op_ml.normalizer(x)
    assert y.type == Tensor(np.float64, ("N", 5))
