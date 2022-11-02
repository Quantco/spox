import numpy as np

import steelix.opset.ai.onnx.ml.v3 as op_ml
from steelix import _type_system as _type_system
from steelix._graph import arguments, results


def test_one_hot_encoder_custom_type_inference(onnx_helper):
    """Test custom type and shape inference of the ``OneHotEncoder``."""
    (str_cats,) = arguments(str_cats=_type_system.Tensor(np.str_, ("N",)))
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
