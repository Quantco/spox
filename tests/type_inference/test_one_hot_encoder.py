import numpy as np

import spox.opset.ai.onnx.ml.v3 as op_ml
from spox._graph import arguments, results
from spox._type_system import Tensor


def test_one_hot_encoder_inference():
    """Test type and shape inference for ``OneHotEncoder``."""
    (str_cats, int_cats) = arguments(
        str_cats=Tensor(np.str_, ("N", 5)), int_cats=Tensor(np.int64, ("N", 5))
    )
    str_encoded = op_ml.one_hot_encoder(str_cats, cats_strings=["a", "b", "c"])
    int_encoded = op_ml.one_hot_encoder(int_cats, cats_int64s=[0, 1, 2, 3])
    assert str_encoded.type == Tensor(np.float32, ("N", 5, 3))
    assert int_encoded.type == Tensor(np.float32, ("N", 5, 4))


def test_one_hot_encoder_runs(onnx_helper):
    """Test custom type and shape inference and correct runtime result of the ``OneHotEncoder``."""
    (str_cats,) = arguments(str_cats=Tensor(np.str_, ("N",)))
    encoded = op_ml.one_hot_encoder(str_cats, cats_strings=np.array(["a", "b", "c"]))
    graph = results(encoded=encoded)
    onnx_helper.assert_close(
        onnx_helper.run(graph, "encoded", str_cats=["a", "b", "c", "z"]),
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )
