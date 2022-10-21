import numpy as np

from steelix import arguments, Tensor
import steelix.opset.ai.onnx.ml.v3 as op_ml


def test_array_feature_extractor_inference():
    (x, y) = arguments(x=Tensor(np.str_, (None, 5, "N",)), y=Tensor(np.int64, (None, 5)))
    r = op_ml.array_feature_extractor(x, y)
    assert r.type == Tensor(np.str_, (None, 5))


def test_array_feature_extractor_inference_untyped():
    # We are not trying to do more complicated guessing in type inference for now.
    (x, y) = arguments(x=Tensor(np.str_, None), y=Tensor(np.int64, (None, 5)))
    r = op_ml.array_feature_extractor(x, y)
    assert r.type is None
