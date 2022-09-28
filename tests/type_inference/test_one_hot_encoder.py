import numpy as np

import steelix
import steelix.opset.ai.onnx.ml.v3 as op_ml


def test_one_hot_encoder_custom_type_inference(onnx_helper):
    """Test custom type and shape inference of the ``OneHotEncoder``."""
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
