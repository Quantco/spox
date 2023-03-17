import itertools
from typing import Any, List

import numpy
import pytest

from spox._graph import arguments, results
from spox._type_system import Tensor

TESTED_CONST_ROWS: List[List[Any]] = [
    [0, 1, 2],
    [0.0, 1.0, 2.0],
    [numpy.float16(3.14), numpy.float16(5.3)],
    ["abc", "def"],
    [True, False],
]


def assert_expected_const(var, value):
    numpy.testing.assert_equal(var._get_value(), numpy.array(value))
    assert var.unwrap_tensor().dtype.type == numpy.array(value).dtype.type
    assert var.unwrap_tensor().shape == numpy.array(value).shape


@pytest.mark.parametrize("value", itertools.chain.from_iterable(TESTED_CONST_ROWS))
def test_const_scalar(op, value):
    assert_expected_const(op.const(value), value)


@pytest.mark.parametrize("row", TESTED_CONST_ROWS)
def test_const_iter(op, row):
    assert_expected_const(op.const(row), row)


@pytest.mark.parametrize("row", TESTED_CONST_ROWS)
def test_const_matrix(op, row):
    assert_expected_const(op.const([row, row]), [row, row])


def test_const_float(op):
    assert op.const(3.14).type.dtype.type == numpy.float64
    assert op.const(numpy.float32(3.14)).type.dtype.type == numpy.float32
    assert op.const(numpy.float64(3.14)).type.dtype.type == numpy.float64


def test_old_const_float(op):
    assert op._const(3.14).type.dtype.type == numpy.float32
    assert op._const(numpy.float32(3.14)).type.dtype.type == numpy.float32
    assert op._const(numpy.float64(3.14)).type.dtype.type == numpy.float64


def test_explicit_unspecified_optional(op, onnx_helper):
    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    r = op.clip(x, min=None, max=op.const(0.0, numpy.float32))
    graph = results(r=r)
    onnx_helper.assert_close(
        onnx_helper.run(graph, "r", x=numpy.array([-1, 1, 2], numpy.float32)),
        [-1, 0, 0],
    )


def test_unspecified_optional(op, onnx_helper):
    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    r = op.clip(x, max=op.const(1.0, numpy.float32))
    r = op.clip(r, min=op.const(-1.0, numpy.float32))
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


def test_const_float_warns(op):
    with pytest.warns(DeprecationWarning):
        op.const(1.0)
    with pytest.warns(DeprecationWarning):
        op.const([1.0, 2.0, 3.0])
