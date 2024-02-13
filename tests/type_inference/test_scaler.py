import numpy as np
import pytest

import spox.opset.ai.onnx.ml.v3 as op_ml
from spox._graph import arguments
from spox._standard import InferenceError
from spox._type_system import Tensor


def test_scaler_inference():
    (x,) = arguments(x=Tensor(np.float64, ("N", 5)))
    y = op_ml.scaler(x, offset=[0.1, 0.2, 0.3, 0.4, 0.5], scale=[2.5])
    assert y.type == Tensor(np.float32, ("N", 5))


def test_scaler_inference_fails_mismatched_lengths():
    (x,) = arguments(x=Tensor(np.float64, ("N", 3)))
    with pytest.raises(InferenceError):
        op_ml.scaler(x, offset=[0.0, 0.1], scale=[1.0])
