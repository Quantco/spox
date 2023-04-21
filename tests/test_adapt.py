from dataclasses import dataclass
from typing import Iterable

import numpy
import onnx
import onnx.parser
import pytest

from spox._attributes import AttrInt64s
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._graph import arguments, results
from spox._node import OpType
from spox._public import inline
from spox._standard import StandardNode
from spox._type_system import Tensor
from spox._var import Var


@pytest.fixture
def old_squeeze() -> onnx.ModelProto:
    return onnx.parser.parse_model(
        """
<
 ir_version: 8,
 opset_import: ["" : 12]
>
agraph (float[1, N] A) => (float[N] B)
{
    B = Squeeze<axes: floats = [0]>(A)
}
"""
    )


@pytest.fixture
def old_identity() -> onnx.ModelProto:
    return onnx.parser.parse_model(
        """
<
 ir_version: 8,
 opset_import: ["" : 13]
>
agraph (float[N] A) => (float[N] B)
{
    B = Identity(A)
}
    """
    )


@pytest.fixture
def inline_old_squeeze_graph(old_squeeze):
    (data,) = arguments(
        data=Tensor(
            numpy.float32,
            (
                1,
                None,
            ),
        )
    )
    (result,) = inline(old_squeeze)(A=data).values()
    return results(final=result).with_opset(("ai.onnx", 17))


@pytest.fixture
def inline_old_identity_twice_graph(old_identity):
    (x,) = arguments(data=Tensor(numpy.float32, (None,)))
    (y,) = inline(old_identity)(A=x).values()
    (z,) = inline(old_identity)(A=y).values()
    return results(final=z).with_opset(("ai.onnx", 17))


@pytest.fixture
def old_squeeze_graph(old_squeeze):
    class Squeeze11(StandardNode):
        @dataclass
        class Attributes(BaseAttributes):
            axes: AttrInt64s

        @dataclass
        class Inputs(BaseInputs):
            data: Var

        @dataclass
        class Outputs(BaseOutputs):
            squeezed: Var

        op_type = OpType("Squeeze", "", 11)

        attrs: Attributes
        inputs: Inputs
        outputs: Outputs

    def squeeze11(_data: Var, _axes: Iterable[int]):
        return Squeeze11(
            Squeeze11.Attributes(AttrInt64s(_axes)), Squeeze11.Inputs(_data)
        ).outputs.squeezed

    (data,) = arguments(
        data=Tensor(
            numpy.float32,
            (
                1,
                None,
            ),
        )
    )
    result = squeeze11(data, [0])
    return results(final=result).with_opset(("ai.onnx", 17))


def test_adapts_inline_old_squeeze(onnx_helper, inline_old_squeeze_graph):
    onnx_helper.assert_close(
        onnx_helper.run(
            inline_old_squeeze_graph,
            "final",
            data=numpy.array([[1, 2, 3, 4]], dtype=numpy.float32),
        ),
        [1, 2, 3, 4],
    )


def test_adapts_inline_old_identity_twice(onnx_helper, inline_old_identity_twice_graph):
    onnx_helper.assert_close(
        onnx_helper.run(
            inline_old_identity_twice_graph,
            "final",
            data=numpy.array([1, 2, 3, 4], dtype=numpy.float32),
        ),
        [1, 2, 3, 4],
    )


def test_adapts_singleton_old_squeeze(onnx_helper, old_squeeze_graph):
    onnx_helper.assert_close(
        onnx_helper.run(
            old_squeeze_graph,
            "final",
            data=numpy.array([[1, 2, 3, 4]], dtype=numpy.float32),
        ),
        [1, 2, 3, 4],
    )
