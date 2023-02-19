import numpy

from spox._graph import arguments, results
from spox._type_system import Tensor


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


def test_variadic_no_input_list_mutation(op, onnx_helper):
    a, b = op.const([1]), op.const([2])
    ins = [a, b]
    concat = op.concat(ins, axis=0)
    ins[1] = b
    assert list(concat._op.inputs) == [a, b]
