# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import ml_dtypes
import numpy as np
import pytest

from spox._shape import Constant, Natural, Shape, ShapeError, Unknown
from spox._type_system import Tensor, Type


@pytest.fixture(params=[np.float64, np.int32, np.bool_])
def scalar_type(request):
    return request.param


@pytest.fixture(
    params=[
        (Shape((Constant(2), Constant(3))), (2, 3)),
        (Shape((Unknown("x"), Unknown("y"))), ("x", "y")),
        (Shape((Unknown(), Unknown("x"), Constant(4))), (None, "x", 4)),
        (Shape(None), None),
    ]
)
def shape_pair(request):
    return request.param


def test_from_onnx_shape_elem(shape_pair):
    if shape_pair[0] is None or shape_pair[1] is None:
        return
    for x, y in zip(shape_pair[0], shape_pair[1]):
        assert x == Natural.from_simple(y)


def test_to_onnx_shape_elem(shape_pair):
    if shape_pair[0] is None or shape_pair[1] is None:
        return
    for x, y in zip(shape_pair[0], shape_pair[1]):
        assert x.to_simple() == y


def test_from_onnx_shape(shape_pair):
    assert shape_pair[0] == Shape.from_simple(shape_pair[1])


def test_to_onnx_shape(shape_pair):
    assert shape_pair[0].to_simple() == shape_pair[1]


def test_shape_rank():
    assert Shape.from_simple((1, 2, 3)).rank == 3
    assert Shape.from_simple((1, 2, 3)).maybe_rank == 3
    assert Shape(None).maybe_rank is None
    with pytest.raises(ShapeError):
        _ = Shape(None).rank


def test_shape_indexing():
    assert Shape.from_simple(("N", "M"))[1].to_simple() == "M"
    assert Shape.from_simple((1, 2, 3))[2].to_simple() == 3
    with pytest.raises(ShapeError):
        _ = Shape(None)[1]
    assert Shape.from_simple(("X", 2, 3, "Y"))[1:3] == Shape.from_simple((2, 3))


def test_tensor_shorthand(scalar_type, shape_pair):
    rich, simple = shape_pair
    assert Tensor(scalar_type, simple)._shape == rich


def test_bad_tensor_type():
    assert Tensor(np.float32) != Tensor(np.float64)  # sanity check
    for typ in (np.object_, object, None):
        with pytest.raises(TypeError):
            Tensor(typ)


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        bool,
        np.str_,
        ml_dtypes.bfloat16,
        ml_dtypes.float8_e4m3fn,
        ml_dtypes.float8_e4m3fnuz,
        ml_dtypes.float8_e5m2,
        ml_dtypes.float8_e5m2fnuz,
        ml_dtypes.uint4,
        ml_dtypes.int4,
        ml_dtypes.float4_e2m1fn,
        ml_dtypes.float8_e8m0fnu,
    ],
)
def test_supported_dtypes(dtype):
    # Check construction, serializing and deserializing of all supported data types
    tensor = Tensor(dtype)
    assert tensor.dtype == dtype
    assert Type._from_onnx(tensor._to_onnx()) == tensor
