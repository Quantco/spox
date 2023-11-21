import pytest

import spox.opset.ai.onnx.v17 as op
from spox._graph import arguments
from spox._standard import InferenceError
from spox._type_system import Tensor


def test_one_hot_inference():
    x, y, z = arguments(
        x=Tensor(int, ("N", "M")), y=Tensor(int, ()), z=Tensor(float, (2,))
    )
    assert op.one_hot(x, y, z).unwrap_tensor() == Tensor(float, ("N", "M", None))
    assert op.one_hot(x, y, z, axis=0).unwrap_tensor() == Tensor(
        float, (None, "N", "M")
    )
    assert op.one_hot(x, y, z, axis=1).unwrap_tensor() == Tensor(
        float, ("N", None, "M")
    )
    assert op.one_hot(x, y, z, axis=-1).unwrap_tensor() == Tensor(
        float, ("N", "M", None)
    )
    assert op.one_hot(x, y, z, axis=-2).unwrap_tensor() == Tensor(
        float, ("N", None, "M")
    )


def test_one_hot_inference_checks_axis_in_range():
    x, y, z = arguments(
        x=Tensor(int, ("N", "M")), y=Tensor(int, ()), z=Tensor(float, (2,))
    )
    with pytest.raises((InferenceError, RuntimeError), match="InferenceError"):
        assert op.one_hot(x, y, z, axis=-4)
    with pytest.raises((InferenceError, RuntimeError), match="InferenceError"):
        assert op.one_hot(x, y, z, axis=3)
