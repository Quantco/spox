from typing import Tuple

import pytest

from spox._shape import Shape


@pytest.fixture(
    params=[
        [(2, 3, 4, 5), (), (2, 3, 4, 5)],
        [(2, 3, 4, 5), (5,), (2, 3, 4, 5)],
        [(2, 3, 4, 5), (2, 1, 1, 5), (2, 3, 4, 5)],
        [(2, 3, 4, 5), (1, 3, 1, 5), (2, 3, 4, 5)],
        [("x", "y", None), ("x", "y", None), ("x", "y", None)],
        [("x", "y", None), (1, "y", 1), ("x", "y", None)],
        [("x", "y", None), (1, None), ("x", "y", None)],
        [(1, 2, 3), (None, None, None), (None, 2, 3)],
        [("x", None), (3, 2, "x", 7), (3, 2, "x", 7)],
        [("x", None), ("y", 2, "x", 7), ("y", 2, "x", 7)],
        [("x", "y", None), ("x", "z", None), ("x", None, None)],
        [("x", "y", None), ("x", "z", "z"), ("x", None, None)],
        [("x", "y", None), (1, "z", 1), ("x", None, None)],
        [("x", "y", None), (1, "z", 1), ("x", None, None)],
        [(5, 7), ("x", None), (5, 7)],
        [(1, 2, 3), ("x", "y", "z"), ("x", 2, 3)],
        [(1, 2, 3), (1, 2, "n"), (1, 2, 3)],
    ]
)
def broadcast_shapes(request):
    return tuple(Shape.from_simple(sh) for sh in request.param)


@pytest.fixture(
    params=[
        [(2, 3, 4, 5), (6,)],
        [(2, 3, 4, 5), (6,)],
        [(2, 3, 4, 5), (3, 1, 1, 5)],
        [(2, 3, 4, 5), (1, 2, 1, 5)],
        [(2, "x", "y"), (3, "y", "z")],
    ]
)
def no_broadcast_shapes(request):
    return tuple(Shape.from_simple(sh) for sh in request.param)


def test_can_broadcast_true(broadcast_shapes: Tuple[Shape, Shape, Shape]):
    first, second, _ = broadcast_shapes
    assert first.can_broadcast(second)


def test_can_broadcast_false(no_broadcast_shapes: Tuple[Shape, Shape]):
    first, second = no_broadcast_shapes
    assert not first.can_broadcast(second)


def test_broadcast(broadcast_shapes: Tuple[Shape, Shape, Shape]):
    first, second, result = broadcast_shapes
    assert first.broadcast(second) == result
