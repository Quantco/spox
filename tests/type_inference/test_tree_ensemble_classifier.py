from typing import Any

import numpy as np

import steelix.opset.ai.onnx.ml.v3 as op_ml
from steelix._graph import arguments
from steelix._type_system import Tensor


def test_tree_ensemble_regressor_inference_str():
    (x,) = arguments(x=Tensor(np.float32, ("N", 5)))
    yz: Any = op_ml.tree_ensemble_classifier(
        x, classlabels_strings=["a", "b"], class_ids=[1, 2, 3]
    )
    assert yz.Y.type == Tensor(np.str_, ("N",))
    assert yz.Z.type == Tensor(np.float32, ("N", 3))


def test_tree_ensemble_regressor_inference_int():
    (x,) = arguments(x=Tensor(np.float32, ("N", 5)))
    yz: Any = op_ml.tree_ensemble_classifier(
        x, classlabels_int64s=[1, 2], class_ids=[1, 2, 3]
    )
    assert yz.Y.type == Tensor(np.int64, ("N",))
    assert yz.Z.type == Tensor(np.float32, ("N", 3))
