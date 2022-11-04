import numpy as np

import steelix.opset.ai.onnx.ml.v3 as op_ml
from steelix import Tensor, arguments


def test_tree_ensemble_regressor_inference():
    (x,) = arguments(x=Tensor(np.float64, ("N", 5)))
    y = op_ml.tree_ensemble_regressor(x, n_targets=3)
    assert y.type == Tensor(np.float32, ("N", 3))
