import numpy as np
import pytest

import spox.opset.ai.onnx.ml.v3 as op_ml
from spox._graph import arguments
from spox._type_system import Tensor


@pytest.mark.parametrize("T,S", [(np.int64, np.str_), (np.str_, np.int64)])
def test_category_mapper_inference(T, S):
    (x,) = arguments(x=Tensor(T, (5, "N")))
    y = op_ml.category_mapper(x, cats_int64s=(0, 1, 2), cats_strings=("a", "b", "c"))
    assert y.type == Tensor(S, (5, "N"))
