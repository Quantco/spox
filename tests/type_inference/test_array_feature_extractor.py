import numpy as np

import spox.opset.ai.onnx.ml.v3 as op_ml
from spox._graph import arguments
from spox._type_system import Tensor


def test_array_feature_extractor_inference():
    (x, y) = arguments(
        x=Tensor(
            np.str_,
            (
                None,
                5,
                "N",
            ),
        ),
        y=Tensor(np.int64, (3,)),
    )
    r = op_ml.array_feature_extractor(x, y)
    assert r.type == Tensor(np.str_, (None, 5, 3))


def test_array_feature_extractor_1d_special_case():
    (x, y) = arguments(x=Tensor(np.str_, (5,)), y=Tensor(np.int64, (7,)))
    r = op_ml.array_feature_extractor(x, y)
    assert r.type == Tensor(np.str_, (1, 7))
