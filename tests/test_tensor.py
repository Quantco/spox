import numpy
import pytest

from spox._shape import Constant, Natural, Shape, ShapeError, Unknown
from spox._type_system import Tensor


@pytest.fixture(params=[numpy.float64, numpy.int32, numpy.bool_])
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
    assert Tensor(numpy.float32) != Tensor(numpy.float64)  # sanity check
    for typ in (numpy.object_, object, None):
        with pytest.raises(TypeError):
            Tensor(typ)
