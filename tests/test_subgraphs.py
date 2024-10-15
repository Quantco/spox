# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable

import numpy as np
import pytest

import spox.opset.ai.onnx.v17 as op
from spox import Var
from spox._exceptions import BuildError
from spox._future import initializer
from spox._graph import arguments, results
from spox._type_system import Sequence, Tensor


def test_subgraph(onnx_helper):
    (e,) = arguments(e=Tensor(np.int64, ()))
    lp, sc = op.loop(
        v_initial=[op.constant(value_floats=[0.0])],  # Initial carry
        body=lambda iter_num, cond_in, carry_in: (
            # Stop condition: iter_num < e
            op.less(iter_num, e),
            # Carried: carry_in + iter_num
            op.add(carry_in, op.cast(iter_num, to=np.float32)),
            # Scanned: iter_num
            iter_num,
        ),
    )
    lp = op.reshape(lp, op.const(np.array([], dtype=np.int64)))
    sc = op.squeeze(sc, op.const([-1]))
    graph = results(lp=lp, sc=sc)
    onnx_helper.assert_close(onnx_helper.run(graph, "lp", e=np.array(9)), [45])
    onnx_helper.assert_close(onnx_helper.run(graph, "lp", e=np.array(10)), [55])
    onnx_helper.assert_close(
        onnx_helper.run(graph, "sc", e=np.array(5)), [0, 1, 2, 3, 4, 5]
    )


def test_subgraph_copy_result(onnx_helper):
    b, x, y = arguments(
        b=Tensor(np.bool_, ()), x=Tensor(np.int64, ()), y=Tensor(np.int64, ())
    )
    z1, z2 = op.if_(b, then_branch=lambda: (x, x), else_branch=lambda: (y, y))
    graph = results(z1=z1, z2=z2)
    onnx_helper.assert_close(
        onnx_helper.run(graph, "z1", b=np.array(True), x=np.array(0), y=np.array(1)),
        [0],
    )
    onnx_helper.assert_close(
        onnx_helper.run(graph, "z2", b=np.array(True), x=np.array(0), y=np.array(1)),
        [0],
    )
    onnx_helper.assert_close(
        onnx_helper.run(graph, "z1", b=np.array(False), x=np.array(0), y=np.array(1)),
        [1],
    )
    onnx_helper.assert_close(
        onnx_helper.run(graph, "z2", b=np.array(False), x=np.array(0), y=np.array(1)),
        [1],
    )


def test_subgraph_non_copy_repeated_result(onnx_helper):
    b, x, y = arguments(
        b=Tensor(np.bool_, ()), x=Tensor(np.int64, ()), y=Tensor(np.int64, ())
    )
    x = op.mul(x, op.const(2))
    y = op.mul(y, op.const(2))
    z1, z2 = op.if_(b, then_branch=lambda: (x, x), else_branch=lambda: (y, y))
    graph = results(z1=z1, z2=z2)
    onnx_helper.assert_close(
        onnx_helper.run(graph, "z1", b=np.array(True), x=np.array(0), y=np.array(1)),
        [0],
    )
    onnx_helper.assert_close(
        onnx_helper.run(graph, "z2", b=np.array(True), x=np.array(0), y=np.array(1)),
        [0],
    )
    onnx_helper.assert_close(
        onnx_helper.run(graph, "z1", b=np.array(False), x=np.array(0), y=np.array(1)),
        [2],
    )
    onnx_helper.assert_close(
        onnx_helper.run(graph, "z2", b=np.array(False), x=np.array(0), y=np.array(1)),
        [2],
    )


def test_outer_scope_arguments(onnx_helper):
    b, x = arguments(b=Tensor(np.bool_, ()), x=Tensor(np.float32, (None,)))
    (r,) = op.if_(
        b, else_branch=lambda: [op.add(x, x)], then_branch=lambda: [op.mul(x, x)]
    )
    graph = results(r=r)
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=np.array(False),
            x=np.array([1, 2, 3], dtype=np.float32),
        ),
        [2, 4, 6],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=np.array(True),
            x=np.array([1, 2, 3], dtype=np.float32),
        ),
        [1, 4, 9],
    )


def test_outer_scope_arguments_nested(onnx_helper):
    b, c, x, y = arguments(
        b=Tensor(np.bool_, ()),
        c=Tensor(np.bool_, ()),
        x=Tensor(np.float32, (None,)),
        y=Tensor(np.float32, (None,)),
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
            b=np.array(True),
            c=np.array(False),
            x=np.array([1, 2, 3], dtype=np.float32),
            y=np.array([0, 2, 1], dtype=np.float32),
        ),
        [0, 2, 1],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=np.array(True),
            c=np.array(True),
            x=np.array([1, 2, 3], dtype=np.float32),
            y=np.array([0, 2, 1], dtype=np.float32),
        ),
        [0, 4, 2],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=np.array(False),
            c=np.array(False),
            x=np.array([1, 2, 3], dtype=np.float32),
            y=np.array([0, 2, 1], dtype=np.float32),
        ),
        [1, 2, 3],
    )


def test_outer_scope_arguments_nested_used_in_both(onnx_helper):
    b, c, x, y = arguments(
        b=Tensor(np.bool_, ()),
        c=Tensor(np.bool_, ()),
        x=Tensor(np.float32, (None,)),
        y=Tensor(np.float32, (None,)),
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


def test_subgraph_argument_used_only_in_subsubgraph(onnx_helper):
    (r,) = op.loop(
        M=op.const([5]),
        v_initial=[],
        body=lambda i, _: (
            op.const(np.array(True)),
            op.if_(
                op.const(np.array(True)),
                else_branch=lambda: [op.const([0])],
                then_branch=lambda: [i],
            )[0],
        ),
    )
    graph = results(r=r)
    graph.to_onnx_model(check_model=2)


def test_copied_outer_argument(onnx_helper):
    b, x = arguments(b=Tensor(np.bool_, ()), x=Tensor(np.float32, (None,)))
    (r,) = op.if_(b, else_branch=lambda: [x], then_branch=lambda: [x])
    graph = results(r=op.add(x, r))
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=np.array(False),
            x=np.array([1, 2, 3], dtype=np.float32),
        ),
        [2, 4, 6],
    )


def test_outer_scope_argument_used_only_inner(onnx_helper):
    b, x = arguments(b=Tensor(np.bool_, ()), x=Tensor(np.float32, (None,)))
    two = op.const(np.float32(2.0))
    (r,) = op.if_(b, else_branch=lambda: [x], then_branch=lambda: [op.mul(two, x)])
    graph = results(r=r)
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=np.array(False),
            x=np.array([1, 2, 3], dtype=np.float32),
        ),
        [1, 2, 3],
    )
    onnx_helper.assert_close(
        onnx_helper.run(
            graph,
            "r",
            b=np.array(True),
            x=np.array([1, 2, 3], dtype=np.float32),
        ),
        [2, 4, 6],
    )


def test_subgraph_arguments_kept_separate(onnx_helper):
    a, b = arguments(a=Tensor(np.int64, ()), b=Tensor(np.int64, ()))

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

    x = op.reshape(x, op.const(np.array([], dtype=np.int64)))
    y = op.reshape(y, op.const(np.array([], dtype=np.int64)))

    onnx_helper.assert_close(
        onnx_helper.run(
            results(x=x, y=y),
            "x",
            a=np.array(0),
            b=np.array(1),
        ),
        0,
    )


def test_scan_for_product(onnx_helper):
    (x,) = arguments(x=Tensor(np.int32, ("N",)))
    one = op.const(np.array(1, dtype=np.int32))
    prod, _x = op.scan([one, x], body=lambda a, p: [op.mul(a, p), a], num_scan_inputs=1)
    onnx_helper.assert_close(
        onnx_helper.run(
            results(prod=prod),
            "prod",
            x=np.array([1, 2, 3, 4, 5], dtype=np.int32),
        ),
        np.array(1 * 2 * 3 * 4 * 5, np.int32),
    )


def test_sequence_map_for_zip_mul(onnx_helper):
    xs, ys = arguments(xs=Sequence(Tensor(np.int64)), ys=Sequence(Tensor(np.int64)))
    (zs,) = op.sequence_map(xs, [ys], body=lambda x, y: (op.mul(x, y),))
    onnx_helper.assert_close(
        onnx_helper.run(
            results(zs=zs),
            "zs",
            xs=[np.array([1]), np.array([2]), np.array([3])],
            ys=[np.array([-1]), np.array([0]), np.array([1])],
        ),
        [np.array([-1]), np.array([0]), np.array([3])],
    )


def test_graph_inherits_subgraph_opset_req(onnx_helper):
    import spox.opset.ai.onnx.ml.v3 as ml

    xs, ys = arguments(xs=Sequence(Tensor(np.int64)), ys=Sequence(Tensor(np.int64)))
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
            xs=[np.array([1]), np.array([2]), np.array([3])],
            ys=[np.array([-1]), np.array([0]), np.array([1])],
        ),
        [np.array([-2]), np.array([0]), np.array([6])],
    )


def test_subgraph_result_depends_on_result(onnx_helper):
    b, x = arguments(b=Tensor(np.bool_, ()), x=Tensor(np.int64, ()))
    x = op.mul(x, op.const(2))
    x1 = op.add(x, op.const(1))
    z1, z2 = op.if_(b, then_branch=lambda: (x, x1), else_branch=lambda: (x1, x))
    graph = results(z1=z1, z2=z2)
    onnx_helper.assert_close(
        onnx_helper.run(graph, "z1", b=np.array(True), x=np.array(5)),
        [10],
    )
    onnx_helper.assert_close(
        onnx_helper.run(graph, "z2", b=np.array(True), x=np.array(5)),
        [11],
    )


@pytest.mark.parametrize("f3", ["fwd", "nest", "tee", "clone"])
@pytest.mark.parametrize("f2", ["fwd", "nest", "tee", "clone"])
@pytest.mark.parametrize("f1", ["fwd", "nest", "tee", "clone"])
def test_complex_scope_trees_on_subgraph_argument(onnx_helper, f1, f2, f3):
    def ident(arg: Callable[[], Var], fork) -> Var:
        if fork == "clone":
            fst, snd = arg(), arg()
        else:
            fst = snd = arg()
        (ret,) = (
            op.if_(
                op.const(True),
                then_branch=lambda: (fst,),
                else_branch=lambda: (snd,) if fork != "nest" else (op.const(-1),),
            )
            if fork != "fwd"
            else (fst,)
        )
        return ret

    (_a,) = op.loop(
        v_initial=[op.const(0)],
        body=lambda _i, _c, a: (
            op.const(False),
            ident(
                lambda: ident(
                    lambda: ident(
                        lambda: a,
                        fork=f3,
                    ),
                    fork=f2,
                ),
                fork=f1,
            ),
        ),
    )

    (x,) = arguments(x=Tensor(int, ()))

    results(_=_a).with_arguments(x).to_onnx_model()


def test_subgraph_not_callback_raises():
    with pytest.raises(TypeError):
        op.if_(
            op.const(True),
            then_branch=[op.const(0)],  # type: ignore
            else_branch=[op.const(1)],  # type: ignore
        )


def test_subgraph_not_iterable_raises():
    with pytest.raises(TypeError):
        op.if_(
            op.const(True),
            then_branch=lambda: op.const(0),  # type: ignore
            else_branch=lambda: op.const(1),  # type: ignore
        )


def test_subgraph_not_var_iterable_raises():
    with pytest.raises(TypeError):
        op.if_(
            op.const(True),
            then_branch=lambda: [0],  # type: ignore
            else_branch=lambda: [op.const(1)],
        )


def test_subgraph_basic_initializer(onnx_helper):
    (e,) = arguments(e=Tensor(bool, ()))
    (f,) = op.if_(
        e, then_branch=lambda: [initializer(0)], else_branch=lambda: [op.const(1)]
    )
    graph = results(f=f)
    onnx_helper.assert_close(onnx_helper.run(graph, "f", e=np.array(True)), [0])
    onnx_helper.assert_close(onnx_helper.run(graph, "f", e=np.array(False)), [1])


def test_subgraph_argument_leak_caught():
    ii: Var = None  # type: ignore

    def fun(i, c):
        nonlocal ii
        ii = i
        return [op.const(True), i]

    (m,) = arguments(m=Tensor(int, ()))
    (r,) = op.loop(M=m, v_initial=[], body=fun)
    graph = results(_=op.add(r, ii))

    with pytest.raises(BuildError):
        graph.to_onnx_model()
