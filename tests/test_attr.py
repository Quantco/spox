# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import spox.opset.ai.onnx.v17 as op
from spox._attributes import AttrString, AttrStrings


def test_bad_attribute_type_cast_fails():
    with pytest.raises(TypeError):
        # to must be an value which can be handled by `np.dtype`.
        op.cast(op.const(1), to="abc")


def test_cast_with_build_in_type():
    op.cast(op.const(1), to=str)


def test_float_instead_of_int_attr():
    with pytest.raises(TypeError):
        op.concat([op.const(1)], axis=3.14)  # type: ignore


def test_non_ascii_string_attr():
    AttrString("🐍", "foo")._to_onnx()
    AttrStrings(["🐍"], "foo")._to_onnx()
