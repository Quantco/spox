# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import ml_dtypes
import numpy as np
import onnx
import pytest

import spox
import spox._future
import spox.opset.ai.onnx.ml.v3 as ml
import spox.opset.ai.onnx.v22 as op
from spox import Optional, Sequence, Tensor, Type, Var, argument
from spox._graph import arguments, results
from spox._shape import Shape
from spox._utils import make_model
from spox._value_prop import ORTValue, PropValue, RefValue
from spox._var import _VarInfo


@pytest.fixture(
    params=[
        spox._future.ValuePropBackend.ONNXRUNTIME,
        spox._future.ValuePropBackend.REFERENCE,
    ],
    autouse=True,
)
def backend(request):
    with spox._future.value_prop_backend(request.param):
        yield request.param


def dummy_var(typ: Type, value: ORTValue):
    """Function for creating a ``var`` without an operator but with a type and value."""
    var_info = _VarInfo(None, typ)  # type: ignore
    return Var(var_info, PropValue.from_ort_value(typ, value))


def assert_equal_value(var: Var, expected: ORTValue | RefValue):
    """
    Convenience function for comparing a ``var``'s propagated value and an expected one.
    Expected Types vs value types:

    - Tensor - numpy.ndarray
    - Optional - spox.var.Nothing or the underlying type
    - Sequence - list of underlying type
    """
    assert var._value is not None, "var.value expected to be known"
    value = var._value.to_ort_value()
    if isinstance(var.type, Tensor):
        expected = np.array(expected)
        assert var.type.dtype.type == expected.dtype.type, "element type must match"
        assert Shape.from_simple(expected.shape) <= var.type._shape, "shape must match"
        if value.dtype.kind in "Uiub":
            np.testing.assert_array_equal(value, expected)
        else:
            np.testing.assert_allclose(value, expected)
    elif isinstance(var.type, Optional):
        if expected is None:
            assert value is None, "value must be Nothing when optional is empty"
        else:
            assert_equal_value(
                dummy_var(var.type.elem_type, var._value.value), expected
            )
    elif isinstance(var.type, Sequence):
        assert isinstance(value, list), "value must be list when it is a Sequence"
        assert isinstance(expected, list), (
            "expected value must be list when tested type is a Sequence"
        )
        assert len(value) == len(expected), "sequence length must match"
        for a, b in zip(value, expected):
            assert_equal_value(
                dummy_var(var.type.elem_type, a),
                b,
            )
    else:
        raise NotImplementedError(f"Datatype {var.type}")


def test_sanity_no_prop():
    (x,) = arguments(x=Tensor(np.int64, ()))
    op.add(x, x)


def test_sanity_const():
    assert_equal_value(op.const(2), np.asarray(2, np.int64))


def test_add():
    assert_equal_value(op.add(op.const(2), op.const(2)), np.asarray(4, np.int64))


def test_div():
    assert_equal_value(
        op.div(op.const(np.float32(5.0)), op.const(np.float32(2.0))),
        np.asarray(2.5, np.float32),
    )


def test_dtypes(dtype, backend):
    # Test value propagation on all supported data types across
    # non-trivial operations.
    # There is an upstream issue for the odd-looking special casing
    # for float8_e5m2: https://github.com/jax-ml/ml_dtypes/issues/216
    if backend == spox._future.ValuePropBackend.ONNXRUNTIME and (
        dtype.kind == "V" or dtype == ml_dtypes.float8_e5m2
    ):
        # onnxruntime does not properly support ml_dtypes in the models' signature
        pytest.xfail(reason="onnxruntime does not support ml_dtypes in model signature")

    arr = np.asarray([1], dtype=dtype)
    # Reshape is defined for all data types
    var = op.reshape(op.const(arr), op.const(np.asarray([1, 1], np.int64)))
    assert_equal_value(var, arr.reshape((1, 1)))


def test_reshape():
    assert_equal_value(
        op.reshape(op.const([1, 2, 3, 4]), op.const([2, 2])),
        np.asarray([[1, 2], [3, 4]], np.int64),
    )


def test_optional():
    assert_equal_value(
        op.optional(op.const(np.float32(2.0))), np.asarray(2.0, np.float32)
    )


def test_empty_optional():
    assert_equal_value(op.optional(type=Tensor(np.float32, ())), None)


def test_empty_optional_has_no_element():
    assert_equal_value(
        op.optional_has_element(op.optional(type=Tensor(np.float32, ()))),
        False,
    )


@pytest.mark.parametrize("min", [None, 2])
def test_optional_clip(min):
    min_var = None if min is None else op.const(min)
    assert_equal_value(
        op.clip(op.const([1, 2, 3]), min=min_var, max=op.const(3)),
        np.clip([1, 2, 3], a_min=min, a_max=3),
    )


def test_sequence_empty():
    assert_equal_value(op.sequence_empty(dtype=np.float32), [])


def test_sequence_append():
    emp = op.sequence_empty(dtype=np.int64)
    assert_equal_value(
        op.sequence_insert(op.sequence_insert(emp, op.const(2)), op.const(1)), [2, 1]
    )


def test_variadict_max():
    a = op.const([2, 1, 4])
    b = op.const(3)
    c = op.const([0])
    assert_equal_value(op.max([a, b, c]), [3, 3, 4])


def test_with_reconstruct():
    a, b = arguments(
        a=Tensor(np.int64, ()),
        b=Tensor(np.int64, ()),
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
        op.const(np.array(["foo"])),
        keys_strings=["foo"],
        values_int64s=[42],
        default_int64=-1,
    )


def test_non_ascii_characters_in_string_tensor():
    op.cast(op.constant(value_string="FööBär"), to=str)


def test_propagated_value_does_not_alias_dtype():
    # Ensures that platform-dependent dtypes aren't accidentally propagated
    x = np.iinfo(np.int64).max + 1
    # Without the explicit astype(uint64), x actually ends up being ulonglong
    assert_equal_value(op.const(x), np.array(x).astype(np.uint64))


def test_value_propagation_does_not_fail_on_unseen_opsets():
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

    model = make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("com.hello", 1),
        ],
    )

    spox.inline(model)(X=op.const(["Test Test"], dtype=np.str_))


def test_value_propagation_across_inline():
    def make_model() -> onnx.ModelProto:
        a = argument(Tensor(np.float64, ("N",)))
        return spox.build({"a": a}, {"b": op.add(a, a)})

    a = op.const([1.0], np.float64)
    (c,) = spox.inline(make_model())(a=a).values()

    assert_equal_value(c, np.asarray([2.0]))


def test_strings():
    x, y = op.const("foo"), op.const("bar")
    assert op.string_concat(x, y)._value.value == "foobar"  # type: ignore

    x, y = op.const(["foo"]), op.const(["bar"])
    np.testing.assert_equal(op.string_concat(x, y)._value.value, np.array(["foobar"]))  # type: ignore
