import numpy
import onnx
import onnx.parser
import pytest

from steelix._graph import arguments, results
from steelix._internal_op import embedded
from steelix._type_system import Tensor


@pytest.fixture
def lin_fun_proto() -> onnx.ModelProto:
    return onnx.parser.parse_model(
        """
<
 ir_version: 8,
 opset_import: ["" : 17]
>
agraph (float[N] A, float X, float[N] B) => (float[N] C)
{
    T = Mul(A, X)
    C = Add(T, B)
}
"""
    )


@pytest.fixture
def min_graph(op, lin_fun_proto):
    first, second = arguments(
        first=Tensor(numpy.float32, (None,)), second=Tensor(numpy.float32, (None,))
    )
    (result,) = embedded(lin_fun_proto)(A=first, X=op.const(1.0), B=second).values()
    return results(final=result)


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


@pytest.fixture
def larger_graph(op, lin_fun_proto):
    first, second = arguments(
        first=Tensor(numpy.float32, (None,)), second=Tensor(numpy.float32, (None,))
    )
    (result,) = embedded(lin_fun_proto)(
        A=op.add(first, second), X=op.const(2.0), B=second
    ).values()
    return results(final=op.div(result, first))


def test_larger(onnx_helper, larger_graph):
    a, b = numpy.random.random(5).astype(numpy.float32), numpy.random.random(5).astype(
        numpy.float32
    )
    onnx_helper.assert_close(
        onnx_helper.run(larger_graph, "final", first=a, second=b), (2 * (a + b) + b) / a
    )
