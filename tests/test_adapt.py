# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import onnx
import onnx.parser
import onnxruntime as ort
import pytest
from onnx.numpy_helper import to_array

import spox.opset.ai.onnx.v18 as op18
import spox.opset.ai.onnx.v19 as op19
from spox import Tensor, Var, argument, build, inline
from spox._attributes import AttrInt64s
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._future import initializer
from spox._graph import arguments, results
from spox._node import OpType
from spox._standard import StandardNode
from spox._utils import make_model
from spox._var import _VarInfo


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
            np.float32,
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
    (x,) = arguments(data=Tensor(np.float32, (None,)))
    (y,) = inline(old_identity)(A=x).values()
    (z,) = inline(old_identity)(A=y).values()
    return results(final=z).with_opset(("ai.onnx", 17))


class Squeeze11(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axes: AttrInt64s

    @dataclass
    class Inputs(BaseInputs):
        data: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        squeezed: _VarInfo

    op_type = OpType("Squeeze", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


def squeeze11(_data: Var, _axes: Iterable[int]):
    return (
        Squeeze11(
            Squeeze11.Attributes(AttrInt64s(_axes, "axes")),
            Squeeze11.Inputs(_data._var_info),
        )
        .get_output_vars()
        .squeezed
    )


@pytest.fixture
def old_squeeze_graph(old_squeeze):
    (data,) = arguments(
        data=Tensor(
            np.float32,
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
            data=np.array([[1, 2, 3, 4]], dtype=np.float32),
        ),
        [1, 2, 3, 4],
    )


def test_adapts_inline_old_identity_twice(onnx_helper, inline_old_identity_twice_graph):
    onnx_helper.assert_close(
        onnx_helper.run(
            inline_old_identity_twice_graph,
            "final",
            data=np.array([1, 2, 3, 4], dtype=np.float32),
        ),
        [1, 2, 3, 4],
    )


def test_adapts_singleton_old_squeeze(onnx_helper, old_squeeze_graph):
    onnx_helper.assert_close(
        onnx_helper.run(
            old_squeeze_graph,
            "final",
            data=np.array([[1, 2, 3, 4]], dtype=np.float32),
        ),
        [1, 2, 3, 4],
    )


def test_adapt_node_with_repeating_input_names():
    a = argument(Tensor(np.float32, ("N",)))
    b = op18.equal(a, a)
    c = op19.identity(a)

    build({"a": a}, {"b": b, "c": c})


def test_adapt_node_initializer():
    init_data = [1.0, 2.0, 3.0]

    a = argument(Tensor(np.float32, ("N",)))
    b = initializer(init_data, np.float32)
    c = op18.equal(a, b)
    d = op19.identity(a)

    model = build({"a": a}, {"b": b, "c": c, "d": d})
    np.testing.assert_allclose(to_array(model.graph.initializer[0]), init_data)


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

    model = make_model(
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

    (a,) = arguments(data=Tensor(np.str_, ("N",)))
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

    model = make_model(
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

    (a,) = arguments(data=Tensor(np.float32, ("N",)))
    (b,) = inline(model)(a).values()

    # Add another node to the model to trigger the adaption logic
    c = op18.identity(b)
    build({"a": a}, {"c": c})


def test_if_adapatation_squeeze():
    cond = argument(Tensor(np.bool_, ()))
    b = argument(Tensor(np.float32, (1,)))
    squeezed = squeeze11(b, [0])
    out = op18.if_(
        cond,
        then_branch=lambda: [squeezed],
        else_branch=lambda: [squeeze11(b, [0])],
    )
    model = build({"b": b, "cond": cond}, {"out": out[0]})

    # predict on model
    b = np.array([1.1], dtype=np.float32)
    cond = np.array(True, dtype=np.bool_)
    out = ort.InferenceSession(model.SerializeToString()).run(
        None, {"b": b, "cond": cond}
    )


def test_if_adaptation_const():
    sq = op19.const(1.1453, dtype=np.float32)
    b = argument(Tensor(np.float32, ("N",)))
    cond = op18.equal(sq, b)
    out = op18.if_(cond, then_branch=lambda: [sq], else_branch=lambda: [sq])
    model = build({"b": b}, {"out": out[0]})
    assert model.domain == "" or model.domain == "ai.onnx"
    assert (
        model.opset_import[0].domain == "ai.onnx" or model.opset_import[0].domain == ""
    )
    assert model.opset_import[0].version > 11
