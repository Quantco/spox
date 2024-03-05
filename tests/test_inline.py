from typing import Dict

import numpy
import onnx
import onnx.parser
import pytest

import spox.opset.ai.onnx.v17 as op
from spox._graph import arguments, results
from spox._inline import rename_in_graph
from spox._public import inline
from spox._type_system import Tensor
from spox._utils import from_array
from spox._var import Var


@pytest.fixture
def add_proto() -> onnx.ModelProto:
    return onnx.parser.parse_model(
        """
<
 ir_version: 8,
 opset_import: ["" : 17]
>
agraph (float[N] A, float[N] B) => (float[N] C)
{
    C = Add(A, B)
}
"""
    )


@pytest.fixture
def inc_proto() -> onnx.ModelProto:
    model = onnx.parser.parse_model(
        """
<
 ir_version: 8,
 opset_import: ["" : 17]
>
agraph (float[N] X) => (float[N] Y)
{
    Y = Add(X, One)
}
"""
    )
    model.graph.initializer.append(from_array(numpy.array(1, numpy.float32), "One"))
    return model


@pytest.fixture
def inc_proto_sparse_one(inc_proto) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.CopyFrom(inc_proto)
    del model.graph.initializer[:]
    model.graph.sparse_initializer.append(
        onnx.helper.make_sparse_tensor(
            from_array(numpy.array([1], numpy.float32), "One"),
            from_array(numpy.array([0], numpy.int64), "OneI"),
            [1],
        )
    )
    return model


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
def proj_proto() -> onnx.ModelProto:
    return onnx.parser.parse_model(
        """
<
 ir_version: 8,
 opset_import: ["" : 17]
>
agraph (double[N] a, double[N] b) => (double[N] c)
{
    c = Identity(b)
}
"""
    )


@pytest.fixture
def relu_proto() -> onnx.ModelProto:
    (x,) = arguments(c=Tensor(float, ()))
    (y,) = op.if_(
        op.less(x, op.const(0.0)),
        then_branch=lambda: (op.const(0.0),),
        else_branch=lambda: (x,),
    )
    return results(y=y).with_arguments(x).to_onnx_model()


@pytest.fixture
def min_graph(lin_fun_proto):
    first, second = arguments(
        first=Tensor(numpy.float32, (None,)), second=Tensor(numpy.float32, (None,))
    )
    (result,) = inline(lin_fun_proto)(
        A=first, X=op.constant(value_float=1.0), B=second
    ).values()
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
def larger_graph(lin_fun_proto):
    first, second = arguments(
        first=Tensor(numpy.float32, (None,)), second=Tensor(numpy.float32, (None,))
    )
    (result,) = inline(lin_fun_proto)(
        A=op.add(first, second), X=op.constant(value_float=2.0), B=second
    ).values()
    return results(final=op.div(result, first))


def test_larger(onnx_helper, larger_graph):
    a, b = (
        numpy.random.random(5).astype(numpy.float32),
        numpy.random.random(5).astype(numpy.float32),
    )
    onnx_helper.assert_close(
        onnx_helper.run(larger_graph, "final", first=a, second=b), (2 * (a + b) + b) / a
    )


@pytest.fixture
def add4_graph(add_proto):
    def add(x, y):
        return inline(add_proto)(A=x, B=y)["C"]

    vec = Tensor(numpy.float32, (None,))
    a, b, c, d = arguments(a=vec, b=vec, c=vec, d=vec)
    r = add(add(a, b), add(c, d))
    return results(r=r)


def test_repeated(onnx_helper, add4_graph):
    a = numpy.array([1.5, 1.125], numpy.float32)
    b = numpy.array([0.25, 4.5], numpy.float32)
    c = numpy.array([4.75, 2], numpy.float32)
    d = numpy.array([0.125, 3], numpy.float32)
    onnx_helper.assert_close(
        onnx_helper.run(add4_graph, "r", a=a, b=b, c=c, d=d), a + b + c + d
    )


@pytest.fixture
def inc3_graph(inc_proto):
    def inc(s):
        return inline(inc_proto)(X=s)["Y"]

    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    y = inc(inc(inc(x)))
    return results(y=y)


def test_inc3_with_initializer(onnx_helper, inc3_graph):
    x = numpy.array([1.5, 0.75], numpy.float32)
    onnx_helper.assert_close(onnx_helper.run(inc3_graph, "y", x=x), x + 3)


@pytest.fixture
def inc3_graph_sparse_init(inc_proto_sparse_one):
    def inc(s):
        return inline(inc_proto_sparse_one)(X=s)["Y"]

    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    y = inc(inc(inc(x)))
    return results(y=y)


def test_inc3_with_sparse_initializer(onnx_helper, inc3_graph_sparse_init):
    x = numpy.array([1.5, 0.75], numpy.float32)
    onnx_helper.assert_close(onnx_helper.run(inc3_graph_sparse_init, "y", x=x), x + 3)


def test_inc3_value_prop(inc_proto):
    def inc(s):
        return inline(inc_proto)(X=s)["Y"]

    assert inc(inc(inc(op.constant(value_floats=[0.0]))))._get_value() == numpy.array(
        [3.0], numpy.float32
    )


def test_proj_different_outer_name(onnx_helper, proj_proto):
    def proj(a, b):
        return inline(proj_proto)(a=a, b=b)["c"]

    x, y = arguments(x=Tensor(float, ("N",)), y=Tensor(float, ("N",)))
    graph = results(z=proj(x, y))

    onnx_helper.assert_close(
        onnx_helper.run(graph, "z", x=numpy.array([1.0]), y=numpy.array([2.0])), 2
    )


def test_proj_same_outer_name(onnx_helper, proj_proto):
    def proj(a, b):
        return inline(proj_proto)(a=a, b=b)["c"]

    x, y = arguments(a=Tensor(float, ("N",)), b=Tensor(float, ("N",)))
    graph = results(c=proj(x, y))

    onnx_helper.assert_close(
        onnx_helper.run(graph, "c", a=numpy.array([1.0]), b=numpy.array([2.0])), 2
    )


def test_proj_composed_same_name(onnx_helper, proj_proto):
    def proj(a, b):
        return inline(proj_proto)(a=a, b=b)["c"]

    x, y = arguments(a=Tensor(float, ("N",)), b=Tensor(float, ("N",)))
    graph = results(c=proj(y, proj(x, y)))

    onnx_helper.assert_close(
        onnx_helper.run(graph, "c", a=numpy.array([1.0]), b=numpy.array([2.0])), 2
    )


def test_relu_inline_subgraph(onnx_helper, relu_proto):
    (a,) = arguments(a=Tensor(float, ()))
    (b,) = inline(relu_proto)(a).values()
    graph = results(b=b).with_arguments(a)

    onnx_helper.assert_close(onnx_helper.run(graph, "b", a=numpy.array(1.0)), 1.0)
    onnx_helper.assert_close(onnx_helper.run(graph, "b", a=numpy.array(-1.0)), 0.0)


def test_symbolic_dim_stripped(add4_graph):
    x: Var
    (x,) = add4_graph.requested_results.values()
    assert x.unwrap_tensor().shape == (None,)


def _make_subgraph_example(op, node, x, c, y1, y2, y):
    return onnx.helper.make_graph(
        [
            onnx.helper.make_node(op, [x, x], [c], name=node),
            onnx.helper.make_node(
                "If",
                [c],
                [y],
                then_branch=onnx.helper.make_graph(
                    [onnx.helper.make_node("Constant", [], [y1], value_int=0)],
                    "subgraph_then_branch",
                    [],
                    [onnx.helper.make_tensor_value_info(x, onnx.TensorProto.INT64, ())],
                ),
                else_branch=onnx.helper.make_graph(
                    [onnx.helper.make_node("Constant", [], [y2], value_int=1)],
                    "subgraph_else_branch",
                    [],
                    [onnx.helper.make_tensor_value_info(x, onnx.TensorProto.INT64, ())],
                ),
            ),
        ],
        "graph",
        [onnx.helper.make_tensor_value_info(x, onnx.TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor_value_info(y, onnx.TensorProto.INT64, ())],
    )


def test_subgraph_rename():
    renames = {"X": "in", "C": "cond", "Y1": "sub1", "Y2": "sub2", "Y": "out"}
    renamed = rename_in_graph(
        _make_subgraph_example("Equal", "I", "X", "C", "Y1", "Y2", "Y"),
        lambda x: renames[x],
        rename_node=lambda x: "test",
        rename_op=lambda d, t: ("", "Less") if (d, t) == ("", "Equal") else (d, t),
    )
    expected = _make_subgraph_example(
        "Less", "test", "in", "cond", "sub1", "sub2", "out"
    )
    for i in range(len(renamed.node)):
        assert renamed.node[i] == expected.node[i], i


def _duplicate_subgraphs_to_list(
    graph_proto_: onnx.GraphProto,
) -> onnx.GraphProto:
    # Replace all graph attributes with a list of that graph twice
    # This obviously invalidates the graph, but we just want to test renaming in those subgraphs
    graph_proto = onnx.GraphProto()
    graph_proto.CopyFrom(graph_proto_)
    for node in graph_proto.node:
        for attr_proto in node.attribute:
            graph = onnx.helper.get_attribute_value(attr_proto)
            if isinstance(graph, onnx.GraphProto):
                attr_proto.Clear()
                attr_proto.type = onnx.AttributeProto.GRAPHS
                attr_proto.graphs.append(graph)
                attr_proto.graphs.append(graph)
    return graph_proto


def test_subgraph_list_rename(relu_proto):
    # This is a simple property test that ensures renaming
    # in lists of subgraphs is the same as in just subgraphs
    renames: Dict[str, str] = {}

    def example_rename(n: str) -> str:
        if n not in renames:
            renames[n] = f"{n}_{len(renames)}"
        return renames[n]

    rename_then_duplicate = _duplicate_subgraphs_to_list(
        rename_in_graph(relu_proto.graph, example_rename)
    )
    renames.clear()
    duplicate_then_rename = rename_in_graph(
        _duplicate_subgraphs_to_list(relu_proto.graph), example_rename
    )
    assert rename_then_duplicate.node == duplicate_then_rename.node
