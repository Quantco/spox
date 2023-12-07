from dataclasses import dataclass
from typing import Iterable

import numpy
import onnx
import onnx.parser
import pytest

import spox.opset.ai.onnx.v18 as op18
import spox.opset.ai.onnx.v19 as op19
from spox import Tensor, Var, argument, build, inline
from spox._attributes import AttrInt64s
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._graph import arguments, results
from spox._node import OpType
from spox._standard import StandardNode


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
    B = Squeeze<axes = [0]>(A)
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
            Squeeze11.Attributes(AttrInt64s(_axes, "axes")), Squeeze11.Inputs(_data)
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


def test_adapt_node_with_repeating_input_names():
    a = argument(Tensor(numpy.float32, ("N",)))
    b = op18.equal(a, a)
    c = op19.identity(a)

    build({"a": a}, {"b": b, "c": c})


def test_inline_model_custom_node_only():
    """Inline a model which only consists of a custom node.

    Such models do not import from the default domain.
    """
    domain = "foo.ai"
    node = onnx.helper.make_node("FooOp", ["a"], ["b"], domain=domain)
    value_infos_input = [
        onnx.helper.make_value_info(
            "a", onnx.helper.make_tensor_type_proto(onnx.TensorProto.STRING, ("N",))
        ),
    ]
    value_infos_output = [
        onnx.helper.make_value_info(
            "b", onnx.helper.make_tensor_type_proto(onnx.TensorProto.STRING, ("N",))
        )
    ]

    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [node],
            "graph",
            value_infos_input,
            value_infos_output,
        ),
        opset_imports=[onnx.helper.make_opsetid(domain, 1)],
    )

    # Ensure that our model is valid
    onnx.checker.check_model(model, full_check=True)

    (a,) = arguments(data=Tensor(numpy.str_, ("N",)))
    (b,) = inline(model)(a).values()

    # Add another node to the model to trigger the adaption logic
    c = op18.identity(b)
    build({"a": a}, {"c": c})


@pytest.mark.skip(
    reason="Adapting custom nodes (including their subgraphs) is currently not supported"
)
def test_inline_model_custom_node_nested(old_squeeze: onnx.ModelProto):
    """A singleton custom node with a old standard node in its attribute."""
    domain = "foo.ai"

    node = onnx.helper.make_node(
        "FooOp", ["a"], ["b"], domain=domain, **{"nested_graph": old_squeeze.graph}
    )
    value_infos_input = [
        onnx.helper.make_value_info(
            "a", onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ("N",))
        ),
    ]
    value_infos_output = [
        onnx.helper.make_value_info(
            "b", onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ("N",))
        )
    ]

    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [node],
            "graph",
            value_infos_input,
            value_infos_output,
        ),
        opset_imports=[
            onnx.helper.make_opsetid(domain, 1),
            onnx.helper.make_opsetid("", 12),
        ],
    )

    # Ensure that our model is valid
    onnx.checker.check_model(model, full_check=True)

    (a,) = arguments(data=Tensor(numpy.float32, ("N",)))
    (b,) = inline(model)(a).values()

    # Add another node to the model to trigger the adaption logic
    c = op18.identity(b)
    build({"a": a}, {"c": c})
