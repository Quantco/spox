import numpy as np
import pytest

import spox.opset.ai.onnx.ml.v3 as op_ml
from spox._graph import arguments
from spox._type_system import Tensor


@pytest.mark.parametrize(
    "keys_type,keys_kwarg,keys",
    [
        (np.float32, "keys_floats", (-0.5, 0.0, 0.5)),
        (np.int64, "keys_int64s", (1, 2, 3)),
        (np.str_, "keys_strings", ("x", "y", "z")),
    ],
)
@pytest.mark.parametrize(
    "values_type,values_kwarg,values",
    [
        (np.float32, "values_floats", (0.0, 0.5, 1.0)),
        (np.int64, "values_int64s", (0, 1, 2)),
        (np.str_, "values_strings", ("a", "b", "c")),
    ],
)
def test_label_encoder_inference(
    keys_type, keys_kwarg, keys, values_type, values_kwarg, values
):
    (x,) = arguments(x=Tensor(keys_type, (None, 5, "N")))
    y = op_ml.label_encoder(x, **{keys_kwarg: keys, values_kwarg: values})
    assert y.type == Tensor(values_type, (None, 5, "N"))
