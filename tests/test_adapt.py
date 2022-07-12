from typing import Iterable, Sequence

import numpy
import onnx
import onnx.parser
import pytest

from steelix.arrow import Arrow
from steelix.arrowfields import ArrowFields
from steelix.attr import Attr
from steelix.attrfields import AttrFields
from steelix.graph import arguments, results
from steelix.internal_op import embedded
from steelix.node import OpType
from steelix.standard import StandardNode
from steelix.type_system import Tensor


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
def embedded_old_squeeze_graph(op, old_squeeze):
    (data,) = arguments(
        data=Tensor(
            numpy.float32,
            (
                1,
                None,
            ),
        )
    )
    (result,) = embedded(old_squeeze)(A=data).values()
    return results(final=result).with_opset(("ai.onnx", 17))


@pytest.fixture
def old_squeeze_graph(op, old_squeeze):
    class Squeeze11(StandardNode):
        class Attributes(AttrFields):
            axes: Attr[Sequence[int]]

        class Inputs(ArrowFields):
            data: Arrow

        class Outputs(ArrowFields):
            squeezed: Arrow

        op_type = OpType("Squeeze", "", 11)

        attrs: Attributes
        inputs: Inputs
        outputs: Outputs

    def squeeze11(_data: Arrow, _axes: Iterable[int]):
        return Squeeze11(
            Squeeze11.Attributes(_axes), Squeeze11.Inputs(_data)
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


def test_adapts_embedded_old_squeeze(onnx_helper, embedded_old_squeeze_graph):
    onnx_helper.assert_close(
        onnx_helper.run(
            embedded_old_squeeze_graph,
            "final",
            data=numpy.array([[1, 2, 3, 4]], dtype=numpy.float32),
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
