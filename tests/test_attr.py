import numpy as np
import pytest

import spox.opset.ai.onnx.v17 as op
from spox._attributes import AttrString, AttrStrings, AttrTensor


def test_bad_attribute_type_cast_fails():
    with pytest.raises(TypeError):
        # to must be an value which can be handled by `np.dtype`.
        op.cast(op.const(1), to="abc")


def test_cast_with_build_in_type():
    op.cast(op.const(1), to=str)


def test_float_instead_of_int_attr():
    with pytest.raises(TypeError):
        op.concat([op.const(1)], axis=3.14)  # type: ignore


@pytest.mark.parametrize(
    "vals, expected, field",
    [
        ([3.14], [3.14], "double_data"),
        (["foo"], [b"foo"], "string_data"),
        # Check scalar values
        (3.14, [3.14], "double_data"),
        ("foo", [b"foo"], "string_data"),
    ],
)
def test_tensor_does_not_use_raw_data(vals, field, expected):
    attr = AttrTensor(np.array(vals), "foo")
    pb = attr._to_onnx_deref()
    assert pb.t.raw_data == b""
    assert getattr(pb.t, field) == expected
    assert pb.t.dims == list(np.array(vals).shape)


def test_non_ascii_string_attr():
    AttrString("🐍", "foo")._to_onnx()
    AttrStrings(["🐍"], "foo")._to_onnx()
