from typing import List, Tuple

import numpy
import onnx
import pytest

from spox._shape import Shape
from spox._type_system import Optional, Sequence, Tensor, Type
from spox._utils import dtype_to_tensor_type, tensor_type_to_dtype


def tensor_type_proto(elem_type, shape):
    return onnx.helper.make_tensor_type_proto(dtype_to_tensor_type(elem_type), shape)


def tensor_shape_proto(shape):
    proto = tensor_type_proto(numpy.float32, shape)
    return proto.tensor_type.shape if proto.tensor_type.HasField("shape") else None


@pytest.fixture
def tensor_elem_type_pairs():
    return [
        (numpy.float32, onnx.TensorProto.FLOAT),
        (numpy.float64, onnx.TensorProto.DOUBLE),
        (numpy.int32, onnx.TensorProto.INT32),
        (numpy.int64, onnx.TensorProto.INT64),
        (numpy.bool_, onnx.TensorProto.BOOL),
        (numpy.str_, onnx.TensorProto.STRING),
    ]


@pytest.fixture
def tensor_shape_pairs():
    return [
        (None, tensor_shape_proto(None)),
        ((), tensor_shape_proto(())),
    ]


@pytest.fixture
def type_pairs() -> List[Tuple[Type, onnx.TypeProto]]:
    tensor_f32 = tensor_type_proto(numpy.float32, None)
    seq_tensor_f32 = onnx.helper.make_sequence_type_proto(tensor_f32)
    opt_seq_tensor_f32 = onnx.helper.make_optional_type_proto(seq_tensor_f32)
    return [
        (Tensor(numpy.float32), tensor_f32),
        (Sequence(Tensor(numpy.float32)), seq_tensor_f32),
        (Optional(Sequence(Tensor(numpy.float32))), opt_seq_tensor_f32),
        (Tensor(numpy.int32, ("N", 1, 2)), tensor_type_proto(numpy.int32, ("N", 1, 2))),
        (Tensor(numpy.bool_, ()), tensor_type_proto(numpy.bool_, ())),
        (Tensor(numpy.str_, (None,)), tensor_type_proto(numpy.str_, (None,))),
    ]


def test_tensor_elem_type_to_onnx(tensor_elem_type_pairs):
    for first, second in tensor_elem_type_pairs:
        assert dtype_to_tensor_type(first) == second


def test_tensor_elem_type_from_onnx(tensor_elem_type_pairs):
    for first, second in tensor_elem_type_pairs:
        assert tensor_type_to_dtype(second) == first


def test_scalar_is_not_unknown_shape():
    assert Tensor(numpy.float32)._to_onnx() != Tensor(numpy.float32, ())._to_onnx()
    assert Shape.from_simple(None) != Shape.from_simple(())
    assert Shape.from_simple(None).to_onnx() != Shape.from_simple(()).to_onnx()


def test_tensor_shape_to_onnx(tensor_shape_pairs):
    for first, second in tensor_shape_pairs:
        assert Shape.from_simple(first).to_onnx() == second


def test_tensor_shape_from_onnx(tensor_shape_pairs):
    for first, second in tensor_shape_pairs:
        assert Shape.from_onnx(second) == Shape.from_simple(first)


def test_type_to_onnx(type_pairs):
    for first, second in type_pairs:
        assert first._to_onnx() == second


def test_type_from_onnx(type_pairs):
    for first, second in type_pairs:
        assert Type._from_onnx(second) == first
