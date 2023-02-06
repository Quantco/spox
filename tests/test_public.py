import numpy as np
import onnx
import onnxruntime
import pytest

from spox import Tensor, argument, build, inline
from spox.opset.ai.onnx import v17 as op


@pytest.fixture
def simple_model():
    x, y = argument(Tensor(float, ())), argument(Tensor(float, ()))
    z = op.add(x, op.mul(y, op.cast(op.const(2.0), to=float)))
    return build({"x": x, "y": y}, {"z": z})


def test_simple_build(simple_model):
    assert onnxruntime.InferenceSession(simple_model.SerializeToString()).run(
        None, {"x": np.array(7.0), "y": np.array(5.0)}
    ) == np.array(17.0)


def test_simple_inline(simple_model):
    a, b = argument(Tensor(float, ())), argument(Tensor(float, ()))
    (c,) = inline(simple_model)(a, y=b).values()
    model = build({"a": a, "b": b}, {"c": c})
    assert onnxruntime.InferenceSession(model.SerializeToString()).run(
        None, {"a": np.array(7.0), "b": np.array(5.0)}
    ) == np.array(17.0)


def test_simple_inline_with_defaults(simple_model):
    model = onnx.ModelProto()
    model.ParseFromString(simple_model.SerializeToString())
    model.graph.initializer.append(
        onnx.helper.make_tensor("x", onnx.TensorProto.DOUBLE, (), [0.0])
    )
    b = argument(Tensor(float, ()))
    (c,) = inline(model)(y=b).values()
    model = build({"b": b}, {"c": c})
    assert onnxruntime.InferenceSession(model.SerializeToString()).run(
        None, {"b": np.array(5.0)}
    ) == np.array(10.0)


def test_simple_inline_types(simple_model):
    a = argument(Tensor(float, ()))
    (r,) = inline(simple_model)(a, a).values()
    assert r.type == Tensor(float, ())


def test_simple_inline_bad_types(simple_model):
    a = argument(Tensor(float, ()))
    b = argument(Tensor(float, ("N", 3)))
    c = argument(Tensor(int, ()))
    with pytest.raises(TypeError):
        inline(simple_model)(b, a)
    with pytest.raises(TypeError):
        inline(simple_model)(a, c)
