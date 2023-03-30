import numpy
import pytest

from spox._graph import arguments, results
from spox._type_system import Tensor


def test_explicit_unspecified_optional(op, onnx_helper):
    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    r = op.clip(x, min=None, max=op.constant(value_float=0.0))
    graph = results(r=r)
    onnx_helper.assert_close(
        onnx_helper.run(graph, "r", x=numpy.array([-1, 1, 2], numpy.float32)),
        [-1, 0, 0],
    )


def test_unspecified_optional(op, onnx_helper):
    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    r = op.clip(x, max=op.constant(value_float=1.0))
    r = op.clip(r, min=op.constant(value_float=-1.0))
    graph = results(r=r)
    onnx_helper.assert_close(
        onnx_helper.run(graph, "r", x=numpy.array([-3, -1, 1, 2], numpy.float32)),
        [-1, -1, 1, 1],
    )


def test_variadic_no_input_list_mutation(op, onnx_helper):
    a, b = op.const([1]), op.const([2])
    ins = [a, b]
    concat = op.concat(ins, axis=0)
    ins[1] = b
    assert list(concat._op.inputs) == [a, b]


def test_variadic_no_attr_mutation_array(op, onnx_helper):
    a = numpy.array([1])
    x = op.constant(value=a)
    a[0] = 0
    assert list(x._op.attrs.value.value) == [1]


def test_variadic_no_attr_mutation_list(op, onnx_helper):
    a = [1]
    x = op.constant(value_ints=a)
    a[0] = 0
    assert list(x._op.attrs.value_ints.value) == [1]


def test_const_float_warns(op):
    with pytest.warns(DeprecationWarning):
        op.const(1.0)
    with pytest.warns(DeprecationWarning):
        op.const([1.0, 2.0, 3.0])


def test_deprecated_raises(op):
    (x,) = arguments(x=Tensor(float, (None,)))
    s = op.const(numpy.array([2.0], numpy.float32))
    with pytest.warns(DeprecationWarning):
        y = op.upsample(x, s)
    graph = results(y=y).with_arguments(x)
    with pytest.raises(Exception):
        graph.to_onnx_model()
