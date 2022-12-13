import numpy as np

import spox.opset.ai.onnx.ml.v3 as op_ml
from spox._graph import arguments
from spox._type_system import Tensor


def test_tree_ensemble_regressor_inference_str():
    (x,) = arguments(x=Tensor(np.float32, ("N", 5)))
    y, z = op_ml.tree_ensemble_classifier(
        x, classlabels_strings=["a", "b"], class_ids=[1, 2, 3]
    )
    assert y.type == Tensor(np.str_, ("N",))
    assert z.type == Tensor(np.float32, ("N", 3))


def test_tree_ensemble_regressor_inference_int():
    (x,) = arguments(x=Tensor(np.float32, ("N", 5)))
    y, z = op_ml.tree_ensemble_classifier(
        x, classlabels_int64s=[1, 2], class_ids=[1, 2, 3]
    )
    assert y.type == Tensor(np.int64, ("N",))
    assert z.type == Tensor(np.float32, ("N", 3))
