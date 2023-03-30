from typing import List

import numpy
import pytest

from spox._shape import Shape
from spox._type_system import Tensor, Type


@pytest.fixture(
    params=[
        [(), None],
        [(2, 3, 4), (2, 3, None), (None, None, None), None],
        [("x", "y", "z"), ("x", "y", None), (None, None, None), None],
    ]
)
def shape_clique(request) -> List[Shape]:
    return [Shape.from_simple(sh) for sh in request.param]


@pytest.fixture(
    params=[
        [Tensor(numpy.int32, (2, 3, 4)), Tensor(numpy.int32, None), Type()],
    ]
)
def weak_type_clique(request) -> List[Type]:
    return request.param


def test_subset_shapes(shape_clique):
    for i, first in enumerate(shape_clique):
        for second in shape_clique[i:]:
            assert first <= second
            assert second >= first


def test_subset_types(weak_type_clique):
    for i, first in enumerate(weak_type_clique):
        for second in weak_type_clique[i:]:
            assert first._subtype(second)


@pytest.mark.parametrize(
    "first,second",
    [((2, 3, 4), (2, 3, 4, None)), ((), (None,)), ((), ("x",)), ((1, 2, 3), (1, 2, 4))],
)
def test_incompatible_shapes(first, second):
    s, z = Shape.from_simple(first), Shape.from_simple(second)
    assert not (s <= z or z <= s)


@pytest.mark.parametrize(
    "first,second",
    [(Tensor(numpy.int32, (2, 3)), Tensor(numpy.int64, (2, 3)))],
)
def test_incompatible_types(first, second):
    assert not (first._subtype(second) or second._subtype(first))
