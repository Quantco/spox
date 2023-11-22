import pytest

import spox.opset.ai.onnx.v17 as op17
import spox.opset.ai.onnx.v18 as op18
import spox.opset.ai.onnx.v19 as op19
import spox.opset.ai.onnx.v20 as op20
from spox._graph import arguments
from spox._standard import InferenceError
from spox._type_system import Tensor


def test_compress_inference():
    x, y = arguments(x=Tensor(float, ("N", "M")), y=Tensor(bool, (None,)))
    assert op.compress(x, y).unwrap_tensor() == Tensor(float, (None,))
    assert op.compress(x, y, axis=0).unwrap_tensor() == Tensor(float, (None, "M"))
    assert op.compress(x, y, axis=1).unwrap_tensor() == Tensor(float, ("N", None))


@pytest.mark.parametrize("op", [op17, op18, op19, op20])
def test_compress_inference_checks_bool_cond(op):
    (x,) = arguments(x=Tensor(float, ("N", "M")))
    with pytest.raises(InferenceError):
        op.compress(x, op.const(123)).unwrap_tensor()
    with pytest.raises(InferenceError):
        op.compress(x, op.const("abc")).unwrap_tensor()
