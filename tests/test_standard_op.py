import re

import numpy
import pytest

import spox.opset.ai.onnx.ml.v3 as ml
import spox.opset.ai.onnx.v17 as op
from spox._exceptions import InferenceError
from spox._graph import arguments
from spox._type_system import Tensor


def test_basic_inference():
    a, b = arguments(a=Tensor(numpy.float32, ("N",)), b=Tensor(numpy.float32, (2, "N")))
    assert op.add(a, b).type == Tensor(numpy.float32, (2, "N"))


def test_variadic_input_inference():
    typ = Tensor(numpy.float32, ("N",))
    a, b, c = arguments(a=typ, b=typ, c=typ)
    assert op.max([a, b, c]).type == typ


def test_variadic_output_inference():
    (x,) = arguments(x=Tensor(numpy.float32, (3, "N")))
    x1, x2, x3 = op.split(x, op.const([1, 1, 1]), outputs_count=3)
    assert x1.type == x2.type == x3.type == Tensor(numpy.float32, (1, "N"))


def test_optional_input_inference():
    (x,) = arguments(x=Tensor(numpy.float32, ("N",)))
    assert op.clip(x, max=op.constant(value_float=1.0)).type == Tensor(
        numpy.float32, ("N",)
    )
    assert op.clip(x, min=op.constant(value_float=0.0)).type == Tensor(
        numpy.float32, ("N",)
    )
    assert op.clip(
        x, min=op.constant(value_float=0.0), max=op.constant(value_float=1.0)
    ).type == Tensor(numpy.float32, ("N",))


def test_function_body_inference():
    a, b = arguments(a=Tensor(numpy.float32, ("N",)), b=Tensor(numpy.float32, ("N",)))
    assert op.greater_or_equal(a, b).type == Tensor(numpy.bool_, ("N",))


def test_inference_fails():
    a, b = arguments(a=Tensor(numpy.float32, (2,)), b=Tensor(numpy.float32, (3,)))
    with pytest.raises((InferenceError, RuntimeError), match="InferenceError"):
        op.add(a, b)


def test_inference_validation_fails():
    a, b = arguments(a=Tensor(numpy.float32, (2,)), b=Tensor(numpy.float64, (2,)))
    with pytest.raises((InferenceError, RuntimeError), match="InferenceError"):
        op.add(a, b)


def test_multiple_outputs():
    x, k = arguments(
        a=Tensor(numpy.float32, ("N", "M", "K")), b=Tensor(numpy.int64, (1,))
    )
    values, indices = op.top_k(x, k)
    assert values.unwrap_type()._subtype(Tensor(numpy.float32, ("N", "M", None)))
    assert indices.unwrap_type()._subtype(Tensor(numpy.int64, ("N", "M", None)))


def test_passing_wrong_type():
    with pytest.raises(TypeError):
        ml.label_encoder(
            X=op.constant(value_ints=["a"]),  # type: ignore
            keys_int64s=[0],
            values_strings=["a"],
            default_string="?",
        )

    with pytest.raises(TypeError):
        ml.label_encoder(
            op.constant(value_ints=[0]),
            keys_int64s=["a"],  # type: ignore
            values_strings=["a"],
            default_string="?",
        )

    with pytest.raises(AttributeError, match=re.escape("has no attribute 'encode'")):
        ml.label_encoder(
            op.constant(value_ints=[0]),
            keys_int64s=[0],
            values_strings=[0],  # type: ignore
            default_string="?",
        )

    with pytest.raises(AttributeError, match=re.escape("has no attribute 'encode'")):
        ml.label_encoder(
            op.constant(value_ints=[0]),
            keys_int64s=[0],
            values_strings=["a"],
            default_string=0,  # type: ignore
        )
