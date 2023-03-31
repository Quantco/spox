import numpy

import spox.opset.ai.onnx.v17 as op
from spox._graph import arguments, results
from spox._type_system import Tensor


def test_explicit_unspecified_optional(onnx_helper):
    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    r = op.clip(x, min=None, max=op.constant(value_float=0.0))
    graph = results(r=r)
    onnx_helper.assert_close(
        onnx_helper.run(graph, "r", x=numpy.array([-1, 1, 2], numpy.float32)),
        [-1, 0, 0],
    )


def test_unspecified_optional(onnx_helper):
    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    r = op.clip(x, max=op.constant(value_float=1.0))
    r = op.clip(r, min=op.constant(value_float=-1.0))
    graph = results(r=r)
    onnx_helper.assert_close(
        onnx_helper.run(graph, "r", x=numpy.array([-3, -1, 1, 2], numpy.float32)),
        [-1, -1, 1, 1],
    )


def test_variadic_no_input_list_mutation(onnx_helper):
    a, b = op.const([1]), op.const([2])
    ins = [a, b]
    concat = op.concat(ins, axis=0)
    ins[1] = b
    assert list(concat._op.inputs) == [a, b]


def test_variadic_no_attr_mutation_array(onnx_helper):
    a = numpy.array([1])
    x = op.constant(value=a)
    a[0] = 0
    assert isinstance(x._op, op._Constant)
    assert x._op.attrs.value is not None
    assert list(x._op.attrs.value.value) == [1]


def test_variadic_no_attr_mutation_list(onnx_helper):
    a = [1]
    x = op.constant(value_ints=a)
    a[0] = 0
    assert isinstance(x._op, op._Constant)
    assert x._op.attrs.value_ints is not None
    assert list(x._op.attrs.value_ints.value) == [1]


def test_deprecated_schemas_removed():
    import spox.opset.ai.onnx.v17 as op17

    assert not hasattr(op17, "scatter")
    assert not hasattr(op17, "upsample")
