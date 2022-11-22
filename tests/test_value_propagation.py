import numpy
import onnxruntime.capi.onnxruntime_pybind11_state
import pytest

from spox import Var, _standard, _type_system, _var
from spox._graph import arguments, results
from spox._shape import Shape


@pytest.fixture(scope="function")
def enable_onnx_value_propagation():
    """Fixture for enabling ONNX Runtime value propagation for tests that use it."""
    prev = _standard._USE_ONNXRUNTIME_VALUE_PROP
    _standard._USE_ONNXRUNTIME_VALUE_PROP = True
    yield
    _standard._USE_ONNXRUNTIME_VALUE_PROP = prev


def dummy_var(typ=None, value=None):
    """Function for creating a ``var`` without an operator but with a type and value."""
    return Var(None, typ, value)  # type: ignore


def assert_equal_value(arr, expected):
    """
    Convenience function for comparing a ``var``'s propagated value and an expected one.
    Expected Types vs value types:

    - Tensor - numpy.ndarray
    - Optional - spox.var.Nothing or the underlying type
    - Sequence - list of underlying type
    """
    assert arr._value is not None, "var.value expected to be known"
    if isinstance(arr.type, _type_system.Tensor):
        expected = numpy.array(expected)
        assert arr.type.dtype.type == expected.dtype.type, "element type must match"
        assert Shape.from_simple(expected.shape) <= arr.type._shape, "shape must match"
        numpy.testing.assert_allclose(arr._value, expected)
    elif isinstance(arr.type, _type_system.Optional):
        if expected is None:
            assert (
                arr._value is _var.Nothing
            ), "value must be Nothing when optional is empty"
        else:
            assert_equal_value(dummy_var(arr.type.elem_type, arr._value), expected)
    elif isinstance(arr.type, _type_system.Sequence):
        assert isinstance(arr._value, list), "value must be list when it is a Sequence"
        assert len(arr._value) == len(expected), "sequence length must match"
        for a, b in zip(arr._value, expected):
            assert_equal_value(dummy_var(arr.type.elem_type, a), b)
    else:
        raise NotImplementedError(f"Datatype {arr.type}")


def test_sanity_no_prop(enable_onnx_value_propagation, op):
    (x,) = arguments(x=_type_system.Tensor(numpy.int64, ()))
    op.add(x, x)


def test_sanity_const(enable_onnx_value_propagation, op):
    assert_equal_value(op.const(2), numpy.int64(2))


def test_add(enable_onnx_value_propagation, op):
    assert_equal_value(op.add(op.const(2), op.const(2)), numpy.int64(4))


def test_div(enable_onnx_value_propagation, op):
    assert_equal_value(op.div(op.const(5.0), op.const(2.0)), numpy.float32(2.5))


def test_identity(enable_onnx_value_propagation, op):
    for x in [
        5,
        [1, 2, 3],
        numpy.array([[1, 2], [3, 4], [5, 6]]),
        numpy.array(0.5, dtype=numpy.float32),
    ]:
        assert_equal_value(op.const(x), x)


def test_reshape(enable_onnx_value_propagation, op):
    assert_equal_value(
        op.reshape(op.const([1, 2, 3, 4]), op.const([2, 2])), [[1, 2], [3, 4]]
    )


def test_optional(enable_onnx_value_propagation, op):
    assert_equal_value(op.optional(op.const(2.0)), numpy.float32(2.0))


def test_empty_optional(enable_onnx_value_propagation, op):
    assert_equal_value(op.optional(type=_type_system.Tensor(numpy.float32, ())), None)


def test_empty_optional_has_no_element(enable_onnx_value_propagation, op):
    assert_equal_value(
        op.optional_has_element(
            op.optional(type=_type_system.Tensor(numpy.float32, ()))
        ),
        False,
    )


def test_sequence_empty(enable_onnx_value_propagation, op):
    assert_equal_value(op.sequence_empty(dtype=numpy.float32), [])


def test_sequence_append(enable_onnx_value_propagation, op):
    emp = op.sequence_empty(dtype=numpy.int64)
    assert_equal_value(
        op.sequence_insert(op.sequence_insert(emp, op.const(2)), op.const(1)), [2, 1]
    )


def test_with_reconstruct(enable_onnx_value_propagation, op):
    a, b = arguments(
        a=_type_system.Tensor(numpy.int64, ()),
        b=_type_system.Tensor(numpy.int64, ()),
    )
    c = op.add(a, b)
    graph = (
        results(c=c).with_arguments(a, b)._with_constructor(lambda x, y: [op.add(x, y)])
    )
    assert_equal_value(
        graph._reconstruct(op.const(2), op.const(3)).requested_results["c"], 5
    )


def test_bad_reshape_raises(enable_onnx_value_propagation, op):
    op.reshape(op.const([1, 2]), op.const([2]))  # sanity
    with pytest.raises(onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException):
        op.reshape(op.const([1, 2, 3]), op.const([2]))
