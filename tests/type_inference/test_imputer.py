import numpy as np
import pytest

import spox.opset.ai.onnx.ml.v3 as op_ml
from spox._graph import arguments
from spox._standard import InferenceError
from spox._type_system import Tensor


def test_imputer_inference():
    (x,) = arguments(x=Tensor(np.int64, (None, 5, "N")))
    y = op_ml.imputer(x, imputed_value_int64s=[999], replaced_value_int64=-1)
    assert y.type == Tensor(np.int64, (None, 5, "N"))


def test_imputer_inference_with_n_feature():
    (x,) = arguments(x=Tensor(np.int64, (None, 5, "N", 3)))
    y = op_ml.imputer(
        x, imputed_value_int64s=[999, 9999, 99999], replaced_value_int64=-1
    )
    assert y.type == Tensor(np.int64, (None, 5, "N", 3))


def test_imputer_inference_with_n_feature_mismatch():
    (x,) = arguments(x=Tensor(np.int64, (None, 5, "N", 4)))
    with pytest.raises(InferenceError):
        op_ml.imputer(
            x, imputed_value_int64s=[999, 9999, 99999], replaced_value_int64=-1
        )
