import numpy
import pytest

from spox._graph import arguments, results
from spox._type_system import Sequence, Tensor


def test_subgraph(op, onnx_helper):
    (e,) = arguments(e=Tensor(numpy.int64, ()))
    lp, sc = op.loop(
        v_initial=[op.const([0.0])],  # Initial carry
        body=lambda iter_num, cond_in, carry_in: (
            # Stop condition: iter_num < e
            op.less(iter_num, e),
            # Carried: carry_in + iter_num
            op.add(carry_in, op.cast(iter_num, to=numpy.float32)),
            # Scanned: iter_num
            iter_num,
        ),
    )
    lp = op.reshape(lp, op.const(numpy.array([], dtype=numpy.int64)))
    sc = op.squeeze(sc, op.const([-1]))
    graph = results(lp=lp, sc=sc)
    onnx_helper.assert_close(onnx_helper.run(graph, "lp", e=numpy.array(9)), [45])
    onnx_helper.assert_close(onnx_helper.run(graph, "lp", e=numpy.array(10)), [55])
    onnx_helper.assert_close(
        onnx_helper.run(graph, "sc", e=numpy.array(5)), [0, 1, 2, 3, 4, 5]
    )


def test_subgraph_copy_result(op, onnx_helper):
    b, x, y = arguments(
        b=Tensor(numpy.bool_, ()), x=Tensor(numpy.int64, ()), y=Tensor(numpy.int64, ())
    )
    z1, z2 = op.if_(b, then_branch=lambda: (x, x), else_branch=lambda: (y, y))
    graph = results(z1=z1, z2=z2)
    onnx_helper.assert_close(
        onnx_helper.run(
            graph, "z1", b=numpy.array(True), x=numpy.array(0), y=numpy.array(1)
        ),
        [0],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            graph, "z2", b=numpy.array(True), x=numpy.array(0), y=numpy.array(1)
        ),
        [0],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            graph, "z1", b=numpy.array(False), x=numpy.array(0), y=numpy.array(1)
        ),
        [1],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            graph, "z2", b=numpy.array(False), x=numpy.array(0), y=numpy.array(1)
        ),
        [1],
    )


def test_subgraph_non_copy_repeated_result(op, onnx_helper):
    b, x, y = arguments(
        b=Tensor(numpy.bool_, ()), x=Tensor(numpy.int64, ()), y=Tensor(numpy.int64, ())
    )
    x = op.mul(x, op.const(2))
    y = op.mul(y, op.const(2))
    z1, z2 = op.if_(b, then_branch=lambda: (x, x), else_branch=lambda: (y, y))
    graph = results(z1=z1, z2=z2)
    onnx_helper.assert_close(
        onnx_helper.run(
            graph, "z1", b=numpy.array(True), x=numpy.array(0), y=numpy.array(1)
        ),
        [0],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            graph, "z2", b=numpy.array(True), x=numpy.array(0), y=numpy.array(1)
        ),
        [0],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            graph, "z1", b=numpy.array(False), x=numpy.array(0), y=numpy.array(1)
        ),
        [2],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            graph, "z2", b=numpy.array(False), x=numpy.array(0), y=numpy.array(1)
        ),
        [2],
    )


def test_outer_scope_arguments(op, onnx_helper):
    b, x = arguments(b=Tensor(numpy.bool_, ()), x=Tensor(numpy.float32, (None,)))
    (r,) = op.if_(
        b, else_branch=lambda: [op.add(x, x)], then_branch=lambda: [op.mul(x, x)]
    )
    graph = results(r=r)
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=numpy.array(False),
            x=numpy.array([1, 2, 3], dtype=numpy.float32),
        ),
        [2, 4, 6],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=numpy.array(True),
            x=numpy.array([1, 2, 3], dtype=numpy.float32),
        ),
        [1, 4, 9],
    )


def test_outer_scope_arguments_nested(op, onnx_helper):
    b, c, x, y = arguments(
        b=Tensor(numpy.bool_, ()),
        c=Tensor(numpy.bool_, ()),
        x=Tensor(numpy.float32, (None,)),
        y=Tensor(numpy.float32, (None,)),
    )
    (r,) = op.if_(
        b,
        else_branch=lambda: [x],
        then_branch=lambda: op.if_(
            c, else_branch=lambda: [y], then_branch=lambda: [op.add(y, y)]
        ),
    )
    graph = results(r=r)
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=numpy.array(True),
            c=numpy.array(False),
            x=numpy.array([1, 2, 3], dtype=numpy.float32),
            y=numpy.array([0, 2, 1], dtype=numpy.float32),
        ),
        [0, 2, 1],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=numpy.array(True),
            c=numpy.array(True),
            x=numpy.array([1, 2, 3], dtype=numpy.float32),
            y=numpy.array([0, 2, 1], dtype=numpy.float32),
        ),
        [0, 4, 2],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=numpy.array(False),
            c=numpy.array(False),
            x=numpy.array([1, 2, 3], dtype=numpy.float32),
            y=numpy.array([0, 2, 1], dtype=numpy.float32),
        ),
        [1, 2, 3],
    )


def test_outer_scope_arguments_nested_used_in_both(op, onnx_helper):
    b, c, x, y = arguments(
        b=Tensor(numpy.bool_, ()),
        c=Tensor(numpy.bool_, ()),
        x=Tensor(numpy.float32, (None,)),
        y=Tensor(numpy.float32, (None,)),
    )
    # An argument is used only in a nested context, and then also afterwards.
    (r,) = op.if_(
        b,
        else_branch=lambda: [y],
        then_branch=lambda: op.if_(
            c, else_branch=lambda: [y], then_branch=lambda: [op.add(x, y)]
        ),
    )
    r = op.add(r, x)
    graph = results(r=r)
    graph.to_onnx_model(check_model=2)


def test_subgraph_argument_used_only_in_subsubgraph(op, onnx_helper):
    (r,) = op.loop(
        M=op.const([5]),
        v_initial=[],
        body=lambda i, _: (
            op.const(numpy.array(True)),
            op.if_(
                op.const(numpy.array(True)),
                else_branch=lambda: [op.const([0])],
                then_branch=lambda: [i],
            )[0],
        ),
    )
    graph = results(r=r)
    graph.to_onnx_model(check_model=2)


def test_copied_outer_argument(op, onnx_helper):
    b, x = arguments(b=Tensor(numpy.bool_, ()), x=Tensor(numpy.float32, (None,)))
    (r,) = op.if_(b, else_branch=lambda: [x], then_branch=lambda: [x])
    graph = results(r=op.add(x, r))
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=numpy.array(False),
            x=numpy.array([1, 2, 3], dtype=numpy.float32),
        ),
        [2, 4, 6],
    )


def test_outer_scope_argument_used_only_inner(op, onnx_helper):
    b, x = arguments(b=Tensor(numpy.bool_, ()), x=Tensor(numpy.float32, (None,)))
    (r,) = op.if_(
        b, else_branch=lambda: [x], then_branch=lambda: [op.mul(op.const(2.0), x)]
    )
    graph = results(r=r)
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=numpy.array(False),
            x=numpy.array([1, 2, 3], dtype=numpy.float32),
        ),
        [1, 2, 3],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=numpy.array(True),
            x=numpy.array([1, 2, 3], dtype=numpy.float32),
        ),
        [2, 4, 6],
    )


def test_subgraph_arguments_kept_separate(op, onnx_helper):
    a, b = arguments(a=Tensor(numpy.int64, ()), b=Tensor(numpy.int64, ()))

    x, y = op.loop(
        v_initial=[a, b],  # _a = a, _b = b
        body=lambda _i, _c, _a, _b: (
            op.const(False),
            *op.loop(
                v_initial=[_b, _a],  # _A = _b, _B = _a
                body=lambda _I, _C, _A, _B: (op.const(False), _a, _b),
            ),
        ),  # -> (_a, _b) = (a, b)
    )  # (a, b)

    x = op.reshape(x, op.const(numpy.array([], dtype=numpy.int64)))
    y = op.reshape(y, op.const(numpy.array([], dtype=numpy.int64)))

    onnx_helper.assert_close(
        onnx_helper.run(
            results(x=x, y=y),
            "x",
            a=numpy.array(0),
            b=numpy.array(1),
        ),
        0,
    )


def test_scan_for_product(op, onnx_helper):
    (x,) = arguments(x=Tensor(numpy.int32, ("N",)))
    one = op.const(numpy.array(1, dtype=numpy.int32))
    prod, _x = op.scan([one, x], body=lambda a, p: [op.mul(a, p), a], num_scan_inputs=1)
    onnx_helper.assert_close(
        onnx_helper.run(
            results(prod=prod),
            "prod",
            x=numpy.array([1, 2, 3, 4, 5], dtype=numpy.int32),
        ),
        numpy.array(1 * 2 * 3 * 4 * 5, numpy.int32),
    )


def test_sequence_map_for_zip_mul(op, onnx_helper):
    xs, ys = arguments(
        xs=Sequence(Tensor(numpy.int32)), ys=Sequence(Tensor(numpy.int32))
    )
    (zs,) = op.sequence_map(xs, [ys], body=lambda x, y: (op.mul(x, y),))
    onnx_helper.assert_close(
        onnx_helper.run(
            results(zs=zs),
            "zs",
            xs=[numpy.array([1]), numpy.array([2]), numpy.array([3])],
            ys=[numpy.array([-1]), numpy.array([0]), numpy.array([1])],
        ),
        [numpy.array([-1]), numpy.array([0]), numpy.array([3])],
    )


def test_graph_inherits_subgraph_opset_req(op, onnx_helper):
    import spox.opset.ai.onnx.ml.v3 as ml

    xs, ys = arguments(
        xs=Sequence(Tensor(numpy.int64)), ys=Sequence(Tensor(numpy.int64))
    )
    (zs,) = op.sequence_map(
        xs,
        [ys],
        body=lambda x, y: (
            op.mul(
                ml.label_encoder(x, keys_int64s=[1, 2, 3], values_int64s=[2, 4, 6]), y
            ),
        ),
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            results(zs=zs),
            "zs",
            xs=[numpy.array([1]), numpy.array([2]), numpy.array([3])],
            ys=[numpy.array([-1]), numpy.array([0]), numpy.array([1])],
        ),
        [numpy.array([-2]), numpy.array([0]), numpy.array([6])],
    )


def test_subgraph_not_callback_raises(op):
    with pytest.raises(TypeError):
        op.if_(op.const(True), then_branch=[op.const(0)], else_branch=[op.const(1)])


def test_subgraph_not_iterable_raises(op):
    with pytest.raises(TypeError):
        op.if_(
            op.const(True),
            then_branch=lambda: op.const(0),
            else_branch=lambda: op.const(1),
        )


def test_subgraph_not_var_iterable_raises(op):
    with pytest.raises(TypeError):
        op.if_(
            op.const(True), then_branch=lambda: [0], else_branch=lambda: [op.const(1)]
        )
