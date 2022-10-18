import numpy as np

import steelix
import steelix.opset.ai.onnx.ml.v3 as op_ml


def test_one_hot_encoder_inference():
    """Test type and shape inference for ``OneHotEncoder``."""
    (str_cats, int_cats) = steelix.arguments(str_cats=steelix.Tensor(np.str_, ("N", 5)), int_cats=steelix.Tensor(np.str_, ("N", 5)))
    str_encoded = op_ml.one_hot_encoder(str_cats, cats_strings=["a", "b", "c"])
    int_encoded = op_ml.one_hot_encoder(int_cats, cats_int64s=[0, 1, 2, 3])
    assert str_encoded.unwrap_tensor() == ("N", 5, 3)
    assert int_encoded.unwrap_tensor() == ("N", 5, 4)


def test_one_hot_encoder_runs(onnx_helper):
    """Test custom type and shape inference and correct runtime result of the ``OneHotEncoder``."""
    (str_cats,) = steelix.arguments(str_cats=steelix.Tensor(np.str_, ("N",)))
    encoded = op_ml.one_hot_encoder(str_cats, cats_strings=np.array(["a", "b", "c"]))
    graph = steelix.results(encoded=encoded)
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
