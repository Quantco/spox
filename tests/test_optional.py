import numpy

import spox.opset.ai.onnx.v17 as op
from spox._graph import arguments
from spox._type_system import Optional, Tensor


def test_optional_type_objects():
    (a,) = arguments(a=Tensor(numpy.int64, ()))
    assert op.optional(a).type == Optional(Tensor(numpy.int64, ()))
    assert op.optional(type=a.type).type == Optional(Tensor(numpy.int64, ()))
    assert op.optional_get_element(op.optional(a)).type == a.type
