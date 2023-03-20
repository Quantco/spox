import numpy
import pytest

import spox.opset.ai.onnx.v17 as op
from spox._graph import arguments, initializer, results
from spox._internal_op import intro
from spox._type_system import Tensor


@pytest.fixture
def min_graph():
    first, second = arguments(
        first=Tensor(numpy.float32, (None,)), second=Tensor(numpy.float32, (None,))
    )
    return results(final=op.add(first, second))


@pytest.fixture
def square_graph():
    (value,) = arguments(value=Tensor(numpy.float32, (None,)))
    return results(final=op.identity(op.mul(value, value)))


@pytest.fixture
def trivial_seq_graph():
    first, second = arguments(
        first=Tensor(numpy.float32, (None,)), second=Tensor(numpy.float32, (None,))
    )
    return results(final=intro(op.add(first, second), second))


@pytest.fixture
def expanding_seq_graph():
    first, second = arguments(
        first=Tensor(numpy.float32, (None,)), second=Tensor(numpy.float32, (None,))
    )
    return results(final=intro(first, second, op.add(first, second)))


@pytest.fixture
def copy_graph():
    first, second = arguments(
        first=Tensor(numpy.float32, (None,)), second=Tensor(numpy.float64, (None,))
    )
    return results(final=first)


@pytest.fixture
def tri_graph():
    first, second, third = arguments(
        first=Tensor(numpy.float32, (2, None)),
        second=Tensor(numpy.float32, (None,)),
        third=Tensor(numpy.int32, (None,)),
    )
    return results(
        final=op.add(first, op.mul(second, op.cast(third, to=numpy.float32)))
    )


@pytest.fixture
def initializer_graph():
    normal, defaulted = arguments(
        normal=Tensor(numpy.float32, (None,)),
        defaulted=numpy.array([3, 2, 1], dtype=numpy.float32),
    )
    frozen = initializer(numpy.array([1, 10, 100], dtype=numpy.float32))
    return results(result=op.mul(op.add(normal, defaulted), frozen))


def test_minimal(onnx_helper, min_graph):
    onnx_helper.assert_close(
        onnx_helper.run(min_graph, "final", first=[1, 2, 3], second=[3, 2, 1]),
        [4, 4, 4],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            min_graph, "final", first=[1, 2, 3, 4, 5], second=[2, 4, 6, 8, 10]
        ),
        [3, 6, 9, 12, 15],
    )


def test_square(onnx_helper, square_graph):
    onnx_helper.assert_close(
        onnx_helper.run(square_graph, "final", value=[2, 3, 4]),
        [4, 9, 16],
    )


def test_trivial_seq(onnx_helper, trivial_seq_graph):
    # This will throw if ``first`` was removed from the graph due to ignored sequencing.
    onnx_helper.assert_close(
        onnx_helper.run(trivial_seq_graph, "final", first=[1, 2, 3], second=[3, 2, 1]),
        [3, 2, 1],
    )


def test_expanding_seq(onnx_helper, expanding_seq_graph):
    onnx_helper.assert_close(
        onnx_helper.run(
            expanding_seq_graph, "final", first=[1, 2, 3], second=[3, 2, 1]
        ),
        [4, 4, 4],
    )


def test_no_shape_throws():
    first, second = arguments(
        first=Tensor(numpy.float32, (None,)), second=Tensor(numpy.float32)
    )
    fun = results(final=op.add(first, second))
    with pytest.raises(ValueError):
        fun.to_onnx_model()


def test_no_type_throws():
    with pytest.raises(TypeError):
        first, second = arguments(first=Tensor(numpy.float32, (None,)), second=None)


def test_copy(onnx_helper, copy_graph):
    onnx_helper.assert_close(
        onnx_helper.run(
            copy_graph, "final", first=[1, 2, 3]
        ),  # Unused input is not in model
        [1, 2, 3],
    )


def test_triple_with_cast(onnx_helper, tri_graph):
    a = numpy.random.rand(2, 4).astype(numpy.float32)
    b = numpy.random.rand(4).astype(numpy.float32)
    c = numpy.array([1, 2, 3, 4], dtype=numpy.int32)
    onnx_helper.assert_close(
        onnx_helper.run(tri_graph, "final", first=a, second=b, third=c),
        a + b * c.astype(numpy.float32),
    )


def test_initializer(onnx_helper, initializer_graph):
    onnx_helper.assert_close(
        onnx_helper.run(initializer_graph, "result", normal=[1, 2, 3]),
        [4, 40, 400],
    )


@pytest.mark.xfail(
    reason="ONNX Runtime does not seem to support overriding initializers."
)
def test_initializer_override(onnx_helper, initializer_graph):
    onnx_helper.assert_close(
        onnx_helper.run(
            initializer_graph, "result", normal=[1, 2, 3], defaulted=[4, 3, 2]
        ),
        [5, 50, 500],
    )


def test_initializer_name_sets_correct(initializer_graph):
    proto = initializer_graph.to_onnx_model()
    assert {arg.name for arg in proto.graph.input} == {"normal", "defaulted"}
    assert "defaulted" in {arg.name for arg in proto.graph.initializer}
