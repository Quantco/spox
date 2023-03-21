import operator

import numpy
import numpy as np
import pytest

from spox import Tensor
from spox._future import operator_overloading
from spox._graph import arguments, results
from spox.opset.ai.onnx import v17 as op


def test_no_overloads_by_default():
    with pytest.raises(TypeError):
        op.const(2) + op.const(2)


@pytest.mark.parametrize(
    "bin_op",
    [operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv],
)
@pytest.mark.parametrize("x", [1.5, 3, 5, [[2, 3]]])
@pytest.mark.parametrize("y", [0.5, 1, 2, [[1], [1.5]]])
def test_arithmetic(bin_op, x, y):
    x, y = np.array(x), np.array(y)
    with operator_overloading(op, type_promotion=True):
        assert (bin_op(op.const(x), op.const(y))._get_value() == bin_op(x, y)).all()


def test_arithmetic_mismatching_types():
    with operator_overloading(op, type_promotion=False):
        with pytest.raises(TypeError):
            op.const(1) + op.const(np.float32(2.0))


def test_var_neg():
    with operator_overloading(op):
        assert (-op.const(2))._get_value() == np.array(-2)


BITS_0 = np.array([False, True, False, True])
BITS_1 = np.array([False, False, True, True])


@pytest.mark.parametrize("bin_op", [operator.and_, operator.or_, operator.xor])
def test_logic(bin_op):
    with operator_overloading(op):
        assert (
            bin_op(op.const(BITS_0), op.const(BITS_1))._get_value()
            == bin_op(BITS_0, BITS_1)
        ).all()


def test_var_invert():
    with operator_overloading(op):
        assert (
            (~op.const(BITS_0))._get_value() == np.array([True, False, True, False])
        ).all()


def test_var_operator_constant_promotion():
    x = op.const(5)
    with operator_overloading(op):
        assert ((x + 1) / 2)._get_value() == np.array(3.0)


def test_var_reflected_operators():
    with operator_overloading(op):
        assert (2 / (1 + op.const(1)))._get_value() == np.array(1.0)


def test_var_operators_context_manager(onnx_helper):
    x, y = arguments(x=Tensor(np.float32, ()), y=Tensor(np.float32, ()))
    with operator_overloading(op):
        r = x / y + y
    graph = results(r=r)
    assert (
        onnx_helper.run(
            graph, "r", x=np.array(5, np.float32), y=np.array(2, np.float32)
        )
        == 5 / 2 + 2
    )


def test_var_operators_decorator(onnx_helper):
    x, y = arguments(x=Tensor(np.float32, ()), y=Tensor(np.float32, ()))

    @operator_overloading(op)
    def inner():
        return x / y + y

    graph = results(r=inner())

    assert (
        onnx_helper.run(
            graph, "r", x=np.array(5, np.float32), y=np.array(2, np.float32)
        )
        == 5 / 2 + 2
    )


def test_bad_operator_types():
    with pytest.raises(TypeError):
        with operator_overloading(op):
            op.const(1) + ...


def test_constant_float_to_int():
    with pytest.raises(TypeError):
        with operator_overloading(op):
            op.const(1) + 1.5


@pytest.mark.parametrize("bin_op", [operator.add, operator.truediv])
@pytest.mark.parametrize(
    "lhs", [1, 1.0, np.int32(1), np.int64(1), np.float32(1), np.float64(1)]
)
@pytest.mark.parametrize(
    "rhs", [2, 2.0, np.int32(2), np.int64(2), np.float32(2), np.float64(2)]
)
def test_var_operator_promotion_like_numpy(bin_op, lhs, rhs):
    with operator_overloading(op, type_promotion=True):
        spox_value = bin_op(op.const(np.array([lhs])), rhs)._get_value()
        numpy_value = bin_op(np.array([lhs]), rhs)
        assert (
            spox_value.dtype == numpy_value.dtype
        ), f"{lhs!r}: {type(lhs)} | {rhs!r}: {type(rhs)}"
        assert numpy.isclose(spox_value, numpy_value).all()
