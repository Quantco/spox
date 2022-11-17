import numpy

from steelix._graph import arguments, results
from steelix._type_system import Tensor


def test_explicit_unspecified_optional(op, onnx_helper):
    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    r = op.clip(x, min=None, max=op.const(0.0))
    graph = results(r=r)
    onnx_helper.assert_close(
        onnx_helper.run(graph, "r", x=numpy.array([-1, 1, 2], numpy.float32)),
        [-1, 0, 0],
    )


def test_unspecified_optional(op, onnx_helper):
    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    r = op.clip(x, max=op.const(1.0))
    r = op.clip(r, min=op.const(-1.0))
    graph = results(r=r)
    onnx_helper.assert_close(
        onnx_helper.run(graph, "r", x=numpy.array([-3, -1, 1, 2], numpy.float32)),
        [-1, -1, 1, 1],
    )
