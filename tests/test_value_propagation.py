import numpy
import onnx
import pytest

import spox
import spox._future
import spox.opset.ai.onnx.ml.v3 as ml
import spox.opset.ai.onnx.v17 as op
from spox import Var, _type_system
from spox._graph import arguments, results
from spox._shape import Shape
from spox._value_prop import ORTValue, PropValue


@pytest.fixture(
    params=[
        spox._future.ValuePropBackend.ONNXRUNTIME,
        spox._future.ValuePropBackend.REFERENCE,
    ]
)
def value_prop_backend(request):
    return request.param


def dummy_var(typ=None, value=None):
    """Function for creating a ``var`` without an operator but with a type and value."""
    return Var(None, typ, value)  # type: ignore


def assert_equal_value(var: Var, expected: ORTValue):
    """
    Convenience function for comparing a ``var``'s propagated value and an expected one.
    Expected Types vs value types:

    - Tensor - numpy.ndarray
    - Optional - spox.var.Nothing or the underlying type
    - Sequence - list of underlying type
    """
    assert var._value is not None, "var.value expected to be known"
    value = var._value.to_ort_value()
    if isinstance(var.type, _type_system.Tensor):
        expected = numpy.array(expected)
        assert var.type.dtype.type == expected.dtype.type, "element type must match"
        assert Shape.from_simple(expected.shape) <= var.type._shape, "shape must match"
        numpy.testing.assert_allclose(value, expected)
    elif isinstance(var.type, _type_system.Optional):
        if expected is None:
            assert value is None, "value must be Nothing when optional is empty"
        else:
            assert_equal_value(
                dummy_var(var.type.elem_type, var._value.value), expected
            )
    elif isinstance(var.type, _type_system.Sequence):
        assert isinstance(value, list), "value must be list when it is a Sequence"
        assert isinstance(
            expected, list
        ), "expected value must be list when tested type is a Sequence"
        assert len(value) == len(expected), "sequence length must match"
        for a, b in zip(value, expected):
            assert_equal_value(
                dummy_var(
                    var.type.elem_type, PropValue.from_ort_value(var.type.elem_type, a)
                ),
                b,
            )
    else:
        raise NotImplementedError(f"Datatype {var.type}")


def test_sanity_no_prop():
    (x,) = arguments(x=_type_system.Tensor(numpy.int64, ()))
    op.add(x, x)


def test_sanity_const():
    assert_equal_value(op.const(2), numpy.int64(2))


def test_add():
    assert_equal_value(op.add(op.const(2), op.const(2)), numpy.int64(4))


def test_div():
    assert_equal_value(
        op.div(op.const(numpy.float32(5.0)), op.const(numpy.float32(2.0))),
        numpy.float32(2.5),
    )


def test_identity():
    for x in [
        5,
        [1, 2, 3],
        numpy.array([[1, 2], [3, 4], [5, 6]]),
        numpy.array(0.5, dtype=numpy.float32),
    ]:
        assert_equal_value(op.const(x), x)


def test_reshape():
    assert_equal_value(
        op.reshape(op.const([1, 2, 3, 4]), op.const([2, 2])), [[1, 2], [3, 4]]
    )


def test_optional():
    assert_equal_value(op.optional(op.const(numpy.float32(2.0))), numpy.float32(2.0))


def test_empty_optional():
    assert_equal_value(op.optional(type=_type_system.Tensor(numpy.float32, ())), None)


def test_empty_optional_has_no_element():
    assert_equal_value(
        op.optional_has_element(
            op.optional(type=_type_system.Tensor(numpy.float32, ()))
        ),
        False,
    )


def test_sequence_empty():
    assert_equal_value(op.sequence_empty(dtype=numpy.float32), [])


def test_sequence_append():
    emp = op.sequence_empty(dtype=numpy.int64)
    assert_equal_value(
        op.sequence_insert(op.sequence_insert(emp, op.const(2)), op.const(1)), [2, 1]
    )


def test_with_reconstruct():
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


def test_bad_reshape_fails(caplog):
    caplog.set_level("DEBUG")
    _ = op.reshape(op.const([1, 2]), op.const([2]))  # sanity
    assert not caplog.records
    _ = op.reshape(op.const([1, 2, 3]), op.const([2]))._value
    assert any(record.levelname == "DEBUG" for record in caplog.records)


def test_give_up_silently():
    # The LabelEncoder currently has no reference implementation.
    ml.label_encoder(
        op.const(numpy.array(["foo"])),
        keys_strings=["foo"],
        values_int64s=[42],
        default_int64=-1,
    )


def test_non_ascii_characters_in_string_tensor():
    op.cast(op.constant(value_string="FööBär"), to=str)


def test_propagated_value_does_not_alias_dtype():
    # Ensures that platform-dependent dtypes aren't accidentally propagated
    x = numpy.iinfo(numpy.int64).max + 1
    # Without the explicit astype(uint64), x actually ends up being ulonglong
    assert_equal_value(op.const(x), numpy.array(x).astype(numpy.uint64))


def test_value_propagation_does_not_fail_on_unseen_opsets(value_prop_backend):
    spox._future.set_value_prop_backend(value_prop_backend)

    model_input = [onnx.helper.make_tensor_value_info("X", elem_type=8, shape=("X",))]
    model_output = [
        onnx.helper.make_tensor_value_info("y", elem_type=8, shape=("y", "max_words"))
    ]

    nodes = [
        onnx.helper.make_node(
            "RandomNode",
            inputs=["X"],
            outputs=["y"],
            domain="com.hello",
        )
    ]

    graph = onnx.helper.make_graph(
        nodes,
        "RandomNode",
        model_input,
        model_output,
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("com.hello", 1),
        ],
    )

    spox.inline(model)(X=op.const(["Test Test"], dtype=numpy.str_))
