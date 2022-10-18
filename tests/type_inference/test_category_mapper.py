import numpy as np
import pytest

from steelix import arguments, Tensor
import steelix.opset.ai.onnx.ml.v3 as op_ml


@pytest.mark.parametrize("T,S", [(np.int64, np.str_), (np.str_, np.int64)])
def test_array_feature_encoder_inference(T, S):
    (x,) = arguments(x=Tensor(T, (5, "N")))
    y = op_ml.category_mapper(x, cats_int64s=(0, 1, 2), cats_strings=("a", "b", "c"))
    assert y.type == Tensor(S, (5, "N"))
